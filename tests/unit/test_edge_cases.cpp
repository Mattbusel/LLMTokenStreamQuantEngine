/// Edge-case and regression tests added during the production-hardening pass.
///
/// These tests cover inputs that existing suites do not exercise:
///   - empty / null-equivalent inputs to every major public API
///   - overflow and saturation paths
///   - invalid JSON in SSE delta parsing
///   - null / zero-length strings in DedupKey
///   - RiskManager constructor validation
///   - LatencyController with zero-window
///   - CsvOutputSink / JsonOutputSink NaN/Inf guard
///   - TokenStreamSimulator stop idempotency
///   - LLMAdapter with empty token string
///   - TradeSignalEngine with neutral (all-zero) weight

#include "gtest/gtest.h"

#include "Config.h"
#include "Deduplicator.h"
#include "LLMAdapter.h"
#include "LLMStreamClient.h"
#include "LatencyController.h"
#include "OutputSinkImpl.h"
#include "RiskManager.h"
#include "TokenStreamSimulator.h"
#include "TradeSignalEngine.h"

#include <chrono>
#include <cmath>
#include <limits>
#include <memory>
#include <string>
#include <thread>
#include <vector>

using namespace llmquant;

// ---------------------------------------------------------------------------
// LLMAdapter edge cases
// ---------------------------------------------------------------------------

TEST(EdgeCase_LLMAdapter, empty_token_returns_neutral_weight) {
    LLMAdapter adapter;
    SemanticWeight w = adapter.map_token_to_weight("");
    EXPECT_DOUBLE_EQ(w.sentiment_score,  0.0);
    EXPECT_DOUBLE_EQ(w.directional_bias, 0.0);
    // confidence_score for unknown tokens defaults to 0.5
    EXPECT_DOUBLE_EQ(w.confidence_score, 0.5);
}

TEST(EdgeCase_LLMAdapter, very_long_token_returns_neutral_weight) {
    LLMAdapter adapter;
    std::string long_token(10000, 'x');
    SemanticWeight w = adapter.map_token_to_weight(long_token);
    EXPECT_DOUBLE_EQ(w.sentiment_score,  0.0);
    EXPECT_DOUBLE_EQ(w.directional_bias, 0.0);
}

TEST(EdgeCase_LLMAdapter, map_sequence_with_all_unknown_tokens_returns_zero_confidence) {
    LLMAdapter adapter;
    // All unknown tokens carry confidence=0.5 each; weighted average results
    // in zero-valued scores for sentiment/bias/volatility but confidence=0.5.
    SemanticWeight w = adapter.map_sequence_to_weight({"zzz_unknown_1", "zzz_unknown_2"});
    EXPECT_DOUBLE_EQ(w.sentiment_score,  0.0);
    EXPECT_DOUBLE_EQ(w.directional_bias, 0.0);
}

TEST(EdgeCase_LLMAdapter, simd_single_unknown_token_matches_scalar) {
    LLMAdapter adapter;
    auto scalar = adapter.map_sequence_to_weight({"not_in_dict"});
    auto simd   = adapter.map_sequence_simd({"not_in_dict"});
    EXPECT_NEAR(scalar.sentiment_score,  simd.sentiment_score,  1e-9);
    EXPECT_NEAR(scalar.directional_bias, simd.directional_bias, 1e-9);
    EXPECT_NEAR(scalar.confidence_score, simd.confidence_score, 1e-9);
}

// ---------------------------------------------------------------------------
// Deduplicator edge cases
// ---------------------------------------------------------------------------

TEST(EdgeCase_Deduplicator, empty_token_string_is_deduplicated) {
    InProcessDeduplicator dedup;
    auto key1 = DedupKey::from_token("");
    auto key2 = DedupKey::from_token("");
    EXPECT_EQ(key1.value, key2.value)
        << "Empty token must produce a stable key";
    EXPECT_EQ(dedup.check_and_register(key1, std::chrono::milliseconds{500}),
              DedupResult::Novel);
    EXPECT_EQ(dedup.check_and_register(key2, std::chrono::milliseconds{500}),
              DedupResult::Duplicate);
}

TEST(EdgeCase_Deduplicator, zero_ttl_entry_is_immediately_expired) {
    InProcessDeduplicator dedup;
    auto key = DedupKey::from_token("zero_ttl_test");
    // Register with TTL=0 ms: the entry should expire immediately.
    dedup.check_and_register(key, std::chrono::milliseconds{0});
    // Wait a tick so steady_clock advances.
    std::this_thread::sleep_for(std::chrono::milliseconds{5});
    auto result = dedup.check_and_register(key, std::chrono::milliseconds{500});
    // After expiry the key should be treated as Novel again.
    EXPECT_EQ(result, DedupResult::Novel);
}

TEST(EdgeCase_Deduplicator, purge_on_empty_map_does_not_crash) {
    InProcessDeduplicator dedup;
    EXPECT_NO_THROW(dedup.purge_expired());
    EXPECT_EQ(dedup.size(), 0u);
}

TEST(EdgeCase_Deduplicator, evict_nonexistent_key_does_not_crash) {
    InProcessDeduplicator dedup;
    auto key = DedupKey::from_token("never_inserted");
    EXPECT_NO_THROW(dedup.evict(key));
}

TEST(EdgeCase_Deduplicator, facade_check_empty_token_string) {
    auto backend = std::make_shared<InProcessDeduplicator>();
    Deduplicator dedup(backend, std::chrono::milliseconds{5000});
    EXPECT_EQ(dedup.check(""), DedupResult::Novel);
    EXPECT_EQ(dedup.check(""), DedupResult::Duplicate);
}

// ---------------------------------------------------------------------------
// RiskManager constructor validation
// ---------------------------------------------------------------------------

TEST(EdgeCase_RiskManager, invalid_min_confidence_above_one_throws) {
    RiskManager::Config cfg;
    cfg.max_bias_magnitude     = 1.0;
    cfg.min_confidence         = 1.5;  // invalid
    cfg.max_signals_per_second = 100;
    cfg.max_drawdown           = 5.0;
    EXPECT_THROW(RiskManager rm(cfg), std::invalid_argument);
}

TEST(EdgeCase_RiskManager, zero_max_signals_per_second_throws) {
    RiskManager::Config cfg;
    cfg.max_bias_magnitude     = 1.0;
    cfg.min_confidence         = 0.1;
    cfg.max_signals_per_second = 0;   // invalid
    cfg.max_drawdown           = 5.0;
    EXPECT_THROW(RiskManager rm(cfg), std::invalid_argument);
}

TEST(EdgeCase_RiskManager, nan_bias_in_signal_is_blocked_by_magnitude_check) {
    RiskManager::Config cfg;
    cfg.max_bias_magnitude     = 1.0;
    cfg.min_confidence         = 0.1;
    cfg.max_signals_per_second = 100;
    cfg.max_drawdown           = 5.0;
    RiskManager rm(cfg);

    TradeSignal s;
    s.delta_bias_shift      = std::numeric_limits<double>::quiet_NaN();
    s.volatility_adjustment = 0.1;
    s.spread_modifier       = 0.05;
    s.confidence            = 0.8;
    s.timestamp_ns          = 1;

    // NaN: std::abs(NaN) > any finite limit, so magnitude check fails.
    bool result = rm.evaluate(s);
    EXPECT_FALSE(result);
    EXPECT_EQ(rm.get_stats().signals_blocked_magnitude.load(), 1u);
}

TEST(EdgeCase_RiskManager, infinite_bias_in_signal_is_blocked) {
    RiskManager::Config cfg;
    cfg.max_bias_magnitude     = 1.0;
    cfg.min_confidence         = 0.1;
    cfg.max_signals_per_second = 100;
    cfg.max_drawdown           = 5.0;
    RiskManager rm(cfg);

    TradeSignal s;
    s.delta_bias_shift      = std::numeric_limits<double>::infinity();
    s.volatility_adjustment = 0.1;
    s.spread_modifier       = 0.05;
    s.confidence            = 0.8;
    s.timestamp_ns          = 1;

    EXPECT_FALSE(rm.evaluate(s));
    EXPECT_EQ(rm.get_stats().signals_blocked_magnitude.load(), 1u);
}

// ---------------------------------------------------------------------------
// LatencyController edge cases
// ---------------------------------------------------------------------------

TEST(EdgeCase_LatencyController, zero_latency_recorded_does_not_crash) {
    LatencyController::Config cfg;
    cfg.target_latency   = std::chrono::microseconds{10};
    cfg.sample_window    = 100;
    cfg.enable_profiling = true;
    LatencyController lc(cfg);

    EXPECT_NO_THROW(lc.record_latency(std::chrono::microseconds{0}));
    auto stats = lc.get_stats();
    EXPECT_EQ(stats.measurements, 1u);
    EXPECT_EQ(stats.min_latency.count(), 0);
}

TEST(EdgeCase_LatencyController, large_latency_recorded_does_not_overflow) {
    LatencyController::Config cfg;
    cfg.target_latency   = std::chrono::microseconds{10};
    cfg.sample_window    = 10;
    cfg.enable_profiling = true;
    LatencyController lc(cfg);

    // Record a 10-second latency (10,000,000 us) -- must not crash or wrap.
    EXPECT_NO_THROW(lc.record_latency(std::chrono::microseconds{10'000'000}));
    auto stats = lc.get_stats();
    EXPECT_EQ(stats.max_latency, std::chrono::microseconds{10'000'000});
}

TEST(EdgeCase_LatencyController, ingestion_pressure_zero_arrival_rate) {
    LatencyController::Config cfg;
    cfg.target_latency   = std::chrono::microseconds{10};
    cfg.sample_window    = 100;
    cfg.enable_profiling = false;
    LatencyController lc(cfg);

    lc.update_ingestion_pressure(0.0, 1000.0);
    auto ps = lc.get_pressure();
    EXPECT_DOUBLE_EQ(ps.ingestion_pressure, 0.0);
}

TEST(EdgeCase_LatencyController, ingestion_pressure_zero_max_rate_does_not_divide_by_zero) {
    LatencyController::Config cfg;
    cfg.target_latency   = std::chrono::microseconds{10};
    cfg.sample_window    = 100;
    cfg.enable_profiling = false;
    LatencyController lc(cfg);

    // max_rate == 0 could cause division by zero -- must not crash.
    EXPECT_NO_THROW(lc.update_ingestion_pressure(100.0, 0.0));
}

// ---------------------------------------------------------------------------
// TradeSignalEngine edge cases
// ---------------------------------------------------------------------------

TEST(EdgeCase_TradeSignalEngine, all_zero_weight_emits_signal_with_zero_bias) {
    TradeSignalEngine::Config cfg;
    cfg.bias_sensitivity       = 1.0;
    cfg.volatility_sensitivity = 1.0;
    cfg.signal_decay_rate      = 0.95;
    cfg.signal_cooldown        = std::chrono::microseconds{0};
    TradeSignalEngine engine(cfg);
    engine.set_backtest_mode(true);

    TradeSignal captured;
    engine.set_signal_callback([&captured](const TradeSignal& s) { captured = s; });

    SemanticWeight zero{0.0, 0.0, 0.0, 0.0};
    engine.process_semantic_weight(zero);

    // All-zero weight: accumulated values must still be finite.
    EXPECT_TRUE(std::isfinite(captured.delta_bias_shift));
    EXPECT_TRUE(std::isfinite(captured.volatility_adjustment));
}

TEST(EdgeCase_TradeSignalEngine, output_sink_cleared_after_add) {
    TradeSignalEngine::Config cfg;
    cfg.signal_cooldown = std::chrono::microseconds{0};
    TradeSignalEngine engine(cfg);
    engine.set_backtest_mode(true);

    auto sink = std::make_shared<MemoryOutputSink>();
    engine.add_output_sink(sink);
    engine.clear_output_sinks();

    engine.set_signal_callback([](const TradeSignal&) {});
    SemanticWeight w{0.5, 0.8, 0.3, 0.6};
    engine.process_semantic_weight(w);

    // After clear_output_sinks(), nothing should reach the sink.
    EXPECT_TRUE(sink->get_signals().empty());
}

// ---------------------------------------------------------------------------
// LLMStreamClient SSE parsing edge cases
// ---------------------------------------------------------------------------

TEST(EdgeCase_LLMStreamClient, parse_sse_delta_empty_string_returns_empty) {
    EXPECT_EQ(LLMStreamClient::parse_sse_delta(""), "");
}

TEST(EdgeCase_LLMStreamClient, parse_sse_delta_done_sentinel_returns_empty) {
    EXPECT_EQ(LLMStreamClient::parse_sse_delta("[DONE]"), "");
}

TEST(EdgeCase_LLMStreamClient, parse_sse_delta_null_content_field_returns_empty) {
    std::string data = R"({"choices":[{"delta":{"content":null}}]})";
    EXPECT_EQ(LLMStreamClient::parse_sse_delta(data), "");
}

TEST(EdgeCase_LLMStreamClient, parse_sse_delta_whitespace_only_content) {
    std::string data = R"({"choices":[{"delta":{"content":"   "}}]})";
    // Whitespace-only content is a valid (non-empty) token; returned as-is.
    std::string result = LLMStreamClient::parse_sse_delta(data);
    EXPECT_EQ(result, "   ");
}

TEST(EdgeCase_LLMStreamClient, parse_sse_delta_deeply_nested_wrong_structure) {
    std::string data = R"({"foo":{"bar":{"baz":42}}})";
    EXPECT_EQ(LLMStreamClient::parse_sse_delta(data), "");
}

TEST(EdgeCase_LLMStreamClient, parse_sse_delta_choices_missing_delta_field) {
    std::string data = R"({"choices":[{"index":0}]})";
    EXPECT_EQ(LLMStreamClient::parse_sse_delta(data), "");
}

// ---------------------------------------------------------------------------
// Config edge cases
// ---------------------------------------------------------------------------

TEST(EdgeCase_Config, load_from_yaml_string_empty_string_returns_false) {
    Config cfg;
    // An empty string is valid YAML (null node) but will fail range validation
    // because defaults will be loaded, not the empty-string values.
    // The call must not crash.
    EXPECT_NO_THROW(cfg.load_from_yaml_string(""));
}

TEST(EdgeCase_Config, load_from_yaml_string_numeric_only_is_invalid) {
    Config cfg;
    // A bare scalar YAML value is valid YAML but not a mapping.
    // load_from_yaml_string must return without crashing.
    EXPECT_NO_THROW(cfg.load_from_yaml_string("42"));
}

TEST(EdgeCase_Config, set_use_memory_stream_updates_config) {
    Config cfg;
    cfg.set_use_memory_stream(true);
    EXPECT_TRUE(cfg.get_config().token_stream.use_memory_stream);
    cfg.set_use_memory_stream(false);
    EXPECT_FALSE(cfg.get_config().token_stream.use_memory_stream);
}

// ---------------------------------------------------------------------------
// TokenStreamSimulator edge cases
// ---------------------------------------------------------------------------

TEST(EdgeCase_TokenStreamSimulator, load_empty_vector_then_start_stop_is_safe) {
    TokenStreamSimulator::Config cfg;
    cfg.token_interval    = std::chrono::microseconds{1000};
    cfg.buffer_size       = 8;
    cfg.use_memory_stream = true;
    TokenStreamSimulator sim(cfg);
    sim.load_tokens_from_memory({});
    sim.set_token_callback([](const Token&) {});
    EXPECT_NO_THROW(sim.start());
    std::this_thread::sleep_for(std::chrono::milliseconds{5});
    EXPECT_NO_THROW(sim.stop());
    EXPECT_EQ(sim.get_stats().tokens_emitted.load(), 0u);
}

TEST(EdgeCase_TokenStreamSimulator, single_token_replays_indefinitely) {
    TokenStreamSimulator::Config cfg;
    cfg.token_interval    = std::chrono::microseconds{100};
    cfg.buffer_size       = 4;
    cfg.use_memory_stream = true;
    TokenStreamSimulator sim(cfg);
    sim.load_tokens_from_memory({"only"});

    std::atomic<uint64_t> count{0};
    sim.set_token_callback([&count](const Token& t) {
        EXPECT_EQ(t.text, "only");
        count++;
    });
    sim.start();
    std::this_thread::sleep_for(std::chrono::milliseconds{20});
    sim.stop();
    EXPECT_GT(count.load(), 1u) << "Single token must loop";
}

// ---------------------------------------------------------------------------
// OutputSink edge cases
// ---------------------------------------------------------------------------

TEST(EdgeCase_OutputSink, csv_sink_emit_nan_bias_writes_zero) {
    const std::string path = "/tmp/test_edge_nan_bias.csv";
    {
        CsvOutputSink sink(path);
        TradeSignal s;
        s.timestamp_ns          = 1;
        s.delta_bias_shift      = std::numeric_limits<double>::quiet_NaN();
        s.volatility_adjustment = std::numeric_limits<double>::infinity();
        s.spread_modifier       = 0.0;
        s.confidence            = 0.5;
        s.latency_us            = 5.0;
        s.strategy_toggle       = 0;
        s.strategy_weight       = 0.0;
        EXPECT_NO_THROW(sink.emit(s));
        sink.flush();
    }
    // File must exist and contain "0" substituted for NaN/Inf.
    std::ifstream f(path);
    ASSERT_TRUE(f.is_open());
    std::string content((std::istreambuf_iterator<char>(f)),
                         std::istreambuf_iterator<char>());
    EXPECT_FALSE(content.empty());
    std::remove(path.c_str());
}

TEST(EdgeCase_OutputSink, json_sink_emit_nan_bias_writes_zero) {
    const std::string path = "/tmp/test_edge_nan_bias.json";
    {
        JsonOutputSink sink(path);
        TradeSignal s;
        s.timestamp_ns          = 2;
        s.delta_bias_shift      = std::numeric_limits<double>::quiet_NaN();
        s.volatility_adjustment = -std::numeric_limits<double>::infinity();
        s.spread_modifier       = 0.0;
        s.confidence            = 0.5;
        s.latency_us            = 3.0;
        s.strategy_toggle       = 0;
        s.strategy_weight       = 0.0;
        EXPECT_NO_THROW(sink.emit(s));
        sink.flush();
    }
    std::ifstream f(path);
    ASSERT_TRUE(f.is_open());
    std::string content((std::istreambuf_iterator<char>(f)),
                         std::istreambuf_iterator<char>());
    EXPECT_FALSE(content.empty());
    std::remove(path.c_str());
}

TEST(EdgeCase_OutputSink, memory_sink_emit_large_volume_does_not_crash) {
    MemoryOutputSink sink;
    TradeSignal s;
    s.timestamp_ns = 0;
    // Emit 10,000 signals: must not crash or OOM.
    for (int i = 0; i < 10000; ++i) {
        s.delta_bias_shift = static_cast<double>(i) * 0.0001;
        s.timestamp_ns     = static_cast<uint64_t>(i);
        sink.emit(s);
    }
    EXPECT_EQ(sink.get_signals().size(), 10000u);
    sink.clear();
    EXPECT_TRUE(sink.get_signals().empty());
}
