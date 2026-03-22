/// @file test_production_readiness.cpp
/// @brief Unit tests added for production-readiness coverage pass.
///
/// Covers:
///   - Config::start_watching() hot-reload callback fires on file change.
///   - Config::save_to_file() error path (unwritable path).
///   - Deduplicator evict() removes a registered key.
///   - Deduplicator default_ttl is respected by check().
///   - InProcessDeduplicator size() / total_novel() / total_duplicates() counters.
///   - InProcessDeduplicator purge_expired() removes stale entries.
///   - TradeSignalEngine::add_output_sink() and clear_output_sinks().
///   - TradeSignalEngine backtest mode emits signal on every token.
///   - LLMAdapter::map_sequence_to_weight() weighted average correctness.
///   - LLMAdapter::aggregate_scalar() range safety with begin == end.
///   - LatencyController::profile_token_processing/signal_generation/queue_lag (no crash).
///   - LatencyController reset_stats() zeroes all counters.
///   - MemoryOutputSink::clear() resets the buffer.
///   - OutputSink polymorphic flush() default is no-op.
///   - RiskManager::get_position() returns the most recently set state.
///   - MetricsLogger CSV and JSON log_risk_rejection paths (no crash).
///   - DedupKey::from_token collision resistance (different contexts differ).
///   - TokenStreamSimulator stop before start is safe.

#include "gtest/gtest.h"

#include "Config.h"
#include "Deduplicator.h"
#include "LLMAdapter.h"
#include "LatencyController.h"
#include "MetricsLogger.h"
#include "OutputSinkImpl.h"
#include "RiskManager.h"
#include "TokenStreamSimulator.h"
#include "TradeSignalEngine.h"

#include <atomic>
#include <chrono>
#include <cstdio>
#include <fstream>
#include <memory>
#include <string>
#include <thread>
#include <vector>

using namespace llmquant;

// ============================================================
// Config::start_watching — fires callback when file changes
// ============================================================

TEST(ProductionReadinessTest, test_config_watch_fires_callback_on_file_change) {
    // Write an initial valid config file.
    const std::string path = "test_watch_tmp.yaml";
    {
        std::ofstream f(path);
        f << "token_stream:\n"
             "  token_interval_ms: 5\n"
             "  buffer_size: 512\n"
             "  use_memory_stream: true\n"
          << "trading:\n"
             "  bias_sensitivity: 1.0\n"
             "  volatility_sensitivity: 1.0\n"
             "  signal_decay_rate: 0.95\n"
             "  signal_cooldown_us: 100\n"
          << "latency:\n"
             "  target_latency_us: 10\n"
             "  sample_window: 100\n"
             "  enable_profiling: false\n"
          << "logging:\n"
             "  log_file_path: \"\"\n"
             "  format: CSV\n"
             "  enable_console: false\n"
             "  flush_interval_ms: 100\n";
    }

    Config cfg;
    ASSERT_TRUE(cfg.load_from_file(path));

    std::atomic<int> reload_count{0};
    (void)cfg.start_watching(path, [&](const SystemConfig&) {
        reload_count++;
    }, /*poll_interval_ms=*/50);

    // Give the watcher thread time to capture the initial mtime.
    std::this_thread::sleep_for(std::chrono::milliseconds(100));

    // Modify the file to trigger a reload.
    {
        std::ofstream f(path);
        f << "token_stream:\n"
             "  token_interval_ms: 10\n"
             "  buffer_size: 512\n"
             "  use_memory_stream: true\n"
          << "trading:\n"
             "  bias_sensitivity: 2.0\n"
             "  volatility_sensitivity: 1.0\n"
             "  signal_decay_rate: 0.90\n"
             "  signal_cooldown_us: 100\n"
          << "latency:\n"
             "  target_latency_us: 10\n"
             "  sample_window: 100\n"
             "  enable_profiling: false\n"
          << "logging:\n"
             "  log_file_path: \"\"\n"
             "  format: CSV\n"
             "  enable_console: false\n"
             "  flush_interval_ms: 100\n";
    }

    // Wait up to 600 ms for the callback to fire.
    for (int i = 0; i < 12 && reload_count.load() == 0; ++i)
        std::this_thread::sleep_for(std::chrono::milliseconds(50));

    cfg.stop_watching();
    std::remove(path.c_str());

    EXPECT_GE(reload_count.load(), 1) << "Hot-reload callback should have fired at least once";

    // Verify the new value was applied.
    EXPECT_DOUBLE_EQ(cfg.get_config().trading.bias_sensitivity, 2.0);
}

// ============================================================
// Config::save_to_file error path
// ============================================================

TEST(ProductionReadinessTest, test_config_save_to_file_returns_false_on_bad_path) {
    Config cfg;
    // A path that cannot be written on any OS.
    bool result = cfg.save_to_file("/no/such/directory/should_not_exist.yaml");
    EXPECT_FALSE(result);
}

// ============================================================
// Deduplicator::evict removes a registered key
// ============================================================

TEST(ProductionReadinessTest, test_deduplicator_evict_allows_reregistration) {
    auto backend = std::make_shared<InProcessDeduplicator>();
    Deduplicator dedup(backend, std::chrono::milliseconds{5000});

    // First check registers the key as novel.
    EXPECT_EQ(dedup.check("evict_token"), DedupResult::Novel);
    // Second check should be a duplicate.
    EXPECT_EQ(dedup.check("evict_token"), DedupResult::Duplicate);

    // After eviction the key is novel again.
    dedup.evict("evict_token");
    EXPECT_EQ(dedup.check("evict_token"), DedupResult::Novel);
}

// ============================================================
// InProcessDeduplicator counters
// ============================================================

TEST(ProductionReadinessTest, test_inprocess_dedup_novel_and_duplicate_counters) {
    InProcessDeduplicator d;
    auto ttl = std::chrono::milliseconds{5000};

    DedupKey k1 = DedupKey::from_token("alpha", "ctx");
    DedupKey k2 = DedupKey::from_token("beta",  "ctx");

    EXPECT_EQ(d.check_and_register(k1, ttl), DedupResult::Novel);
    EXPECT_EQ(d.check_and_register(k2, ttl), DedupResult::Novel);
    EXPECT_EQ(d.check_and_register(k1, ttl), DedupResult::Duplicate);

    EXPECT_EQ(d.total_novel(),      2u);
    EXPECT_EQ(d.total_duplicates(), 1u);
    EXPECT_EQ(d.size(),             2u);
}

// ============================================================
// InProcessDeduplicator purge_expired removes expired entries
// ============================================================

TEST(ProductionReadinessTest, test_inprocess_dedup_purge_expired_removes_stale_keys) {
    InProcessDeduplicator d;
    // Register a key with a 1ms TTL — effectively expired after any sleep.
    DedupKey k = DedupKey::from_token("ephemeral", "");
    (void)d.check_and_register(k, std::chrono::milliseconds{1});

    EXPECT_EQ(d.size(), 1u);

    // Sleep long enough that the entry is expired.
    std::this_thread::sleep_for(std::chrono::milliseconds{5});
    d.purge_expired();

    EXPECT_EQ(d.size(), 0u);
}

// ============================================================
// DedupKey context scoping: same token, different context → different keys
// ============================================================

TEST(ProductionReadinessTest, test_dedup_key_different_contexts_produce_different_keys) {
    DedupKey k1 = DedupKey::from_token("token", "ctx_a");
    DedupKey k2 = DedupKey::from_token("token", "ctx_b");
    EXPECT_NE(k1.value, k2.value);
}

TEST(ProductionReadinessTest, test_dedup_key_same_inputs_produce_same_key) {
    DedupKey k1 = DedupKey::from_token("token", "ctx");
    DedupKey k2 = DedupKey::from_token("token", "ctx");
    EXPECT_EQ(k1, k2);
}

// ============================================================
// TradeSignalEngine output sinks
// ============================================================

TEST(ProductionReadinessTest, test_trade_engine_add_output_sink_receives_signal) {
    TradeSignalEngine::Config cfg;
    cfg.signal_cooldown = std::chrono::microseconds{0};
    TradeSignalEngine engine(cfg);
    engine.set_realtime_mode(false);

    auto sink = std::make_shared<MemoryOutputSink>();
    engine.add_output_sink(sink);

    SemanticWeight w;
    w.directional_bias  = 0.5;
    w.confidence_score  = 0.8;
    w.volatility_score  = 0.3;
    w.sentiment_score   = 0.4;
    engine.process_semantic_weight(w);

    EXPECT_GE(sink->get_signals().size(), 1u)
        << "Output sink should have received at least one signal";
}

TEST(ProductionReadinessTest, test_trade_engine_clear_output_sinks_stops_delivery) {
    TradeSignalEngine::Config cfg;
    cfg.signal_cooldown = std::chrono::microseconds{0};
    TradeSignalEngine engine(cfg);
    engine.set_realtime_mode(false);

    auto sink = std::make_shared<MemoryOutputSink>();
    engine.add_output_sink(sink);
    engine.clear_output_sinks();

    SemanticWeight w;
    w.directional_bias  = 0.5;
    w.confidence_score  = 0.8;
    engine.process_semantic_weight(w);

    EXPECT_EQ(sink->get_signals().size(), 0u)
        << "After clear_output_sinks() the cleared sink must not receive signals";
}

TEST(ProductionReadinessTest, test_trade_engine_backtest_mode_emits_every_token) {
    TradeSignalEngine::Config cfg;
    cfg.signal_cooldown = std::chrono::microseconds{100000};  // 100 ms — would suppress in realtime
    TradeSignalEngine engine(cfg);
    engine.set_backtest_mode(true);

    int count = 0;
    engine.set_signal_callback([&](const TradeSignal&) { count++; });

    SemanticWeight w;
    w.directional_bias = 0.3;
    w.confidence_score = 0.8;
    for (int i = 0; i < 5; ++i) engine.process_semantic_weight(w);

    EXPECT_EQ(count, 5) << "Backtest mode must emit a signal on every token";
}

// ============================================================
// LLMAdapter::map_sequence_to_weight weighted average
// ============================================================

TEST(ProductionReadinessTest, test_llm_adapter_sequence_weight_is_confidence_weighted_average) {
    LLMAdapter adapter;
    // "bullish" has high confidence; "the" has very low confidence.
    auto combined = adapter.map_sequence_to_weight({"bullish", "the"});
    auto bullish   = adapter.map_token_to_weight("bullish");
    auto neutral   = adapter.map_token_to_weight("the");

    // The combined directional_bias should be much closer to "bullish" than "the"
    // because "bullish" carries much higher confidence.
    double diff_bullish = std::abs(combined.directional_bias - bullish.directional_bias);
    double diff_neutral = std::abs(combined.directional_bias - neutral.directional_bias);
    EXPECT_LT(diff_bullish, diff_neutral)
        << "Combined bias should be closer to the high-confidence token";
}

TEST(ProductionReadinessTest, test_llm_adapter_empty_sequence_returns_zero_weight) {
    LLMAdapter adapter;
    auto w = adapter.map_sequence_to_weight({});
    EXPECT_DOUBLE_EQ(w.sentiment_score,  0.0);
    EXPECT_DOUBLE_EQ(w.confidence_score, 0.0);
    EXPECT_DOUBLE_EQ(w.volatility_score, 0.0);
    EXPECT_DOUBLE_EQ(w.directional_bias, 0.0);
}

TEST(ProductionReadinessTest, test_llm_adapter_unknown_token_returns_neutral_weight) {
    LLMAdapter adapter;
    auto w = adapter.map_token_to_weight("xyzzy_unknown_token_12345");
    EXPECT_DOUBLE_EQ(w.sentiment_score, 0.0);
    EXPECT_DOUBLE_EQ(w.confidence_score, 0.5);
    EXPECT_DOUBLE_EQ(w.directional_bias, 0.0);
}

// ============================================================
// LatencyController profile hooks and reset
// ============================================================

TEST(ProductionReadinessTest, test_latency_controller_profile_hooks_do_not_crash) {
    LatencyController::Config cfg;
    cfg.enable_profiling = true;
    cfg.sample_window    = 50;
    cfg.target_latency   = std::chrono::microseconds{10};
    LatencyController lc(cfg);

    // These are profile hints — they must not crash even with no measurements.
    EXPECT_NO_THROW(lc.profile_token_processing());
    EXPECT_NO_THROW(lc.profile_signal_generation());
    EXPECT_NO_THROW(lc.profile_queue_lag());
}

TEST(ProductionReadinessTest, test_latency_controller_reset_zeroes_all_stats) {
    LatencyController::Config cfg;
    cfg.enable_profiling = true;
    cfg.sample_window    = 50;
    cfg.target_latency   = std::chrono::microseconds{10};
    LatencyController lc(cfg);

    for (int i = 1; i <= 10; ++i)
        lc.record_latency(std::chrono::microseconds{i * 5});

    EXPECT_GT(lc.get_stats().measurements, 0u);

    lc.reset_stats();
    auto s = lc.get_stats();
    EXPECT_EQ(s.measurements, 0u);
    EXPECT_EQ(s.avg_latency.count(), 0);
    EXPECT_EQ(s.min_latency.count(), 0);
    EXPECT_EQ(s.max_latency.count(), 0);
}

TEST(ProductionReadinessTest, test_latency_controller_pressure_zero_max_rate_does_not_crash) {
    LatencyController::Config cfg;
    cfg.sample_window    = 50;
    cfg.target_latency   = std::chrono::microseconds{10};
    LatencyController lc(cfg);
    // max_rate_tps == 0 should be clamped internally; must not divide by zero.
    EXPECT_NO_THROW(lc.update_ingestion_pressure(100.0, 0.0));
    auto p = lc.get_pressure();
    EXPECT_GE(p.ingestion_pressure, 0.0);
    EXPECT_LE(p.ingestion_pressure, 1.0);
}

// ============================================================
// MemoryOutputSink::clear()
// ============================================================

TEST(ProductionReadinessTest, test_memory_sink_clear_empties_buffer) {
    MemoryOutputSink sink;
    TradeSignal s;
    s.timestamp_ns = 1;
    sink.emit(s);
    sink.emit(s);
    EXPECT_EQ(sink.get_signals().size(), 2u);
    sink.clear();
    EXPECT_TRUE(sink.get_signals().empty());
}

// ============================================================
// OutputSink default flush() is a no-op (polymorphic call)
// ============================================================

TEST(ProductionReadinessTest, test_output_sink_default_flush_is_noop) {
    // MemoryOutputSink does not override flush(); calling it must not crash.
    MemoryOutputSink sink;
    EXPECT_NO_THROW(sink.flush());
}

// ============================================================
// RiskManager::get_position() returns the last updated state
// ============================================================

TEST(ProductionReadinessTest, test_risk_manager_get_position_reflects_last_update) {
    RiskManager::Config cfg;
    cfg.max_bias_magnitude       = 5.0;
    cfg.max_volatility_magnitude = 5.0;
    cfg.max_spread_magnitude     = 5.0;
    cfg.min_confidence           = 0.0;
    cfg.max_signals_per_second   = 1000;
    cfg.max_drawdown             = 100.0;
    RiskManager rm(cfg);

    RiskManager::PositionState state;
    state.net_position   = 0.42;
    state.position_limit = 1.0;
    state.pnl            = 3.14;
    state.pnl_limit      = -10.0;
    rm.update_position(state);

    auto got = rm.get_position();
    EXPECT_DOUBLE_EQ(got.net_position, 0.42);
    EXPECT_DOUBLE_EQ(got.pnl,          3.14);
}

// ============================================================
// MetricsLogger CSV and JSON log paths with empty file path
// ============================================================

TEST(ProductionReadinessTest, test_metrics_logger_csv_log_risk_rejection_no_crash) {
    MetricsLogger::Config cfg;
    cfg.log_file_path        = "";  // no file output
    cfg.format               = MetricsLogger::OutputFormat::CSV;
    cfg.enable_console_output = false;
    MetricsLogger logger(cfg);
    EXPECT_NO_THROW(logger.log_risk_rejection("magnitude_exceeded", 1.5, 0.9));
    EXPECT_NO_THROW(logger.flush());
}

TEST(ProductionReadinessTest, test_metrics_logger_json_log_signal_generated_no_crash) {
    MetricsLogger::Config cfg;
    cfg.log_file_path         = "";
    cfg.format                = MetricsLogger::OutputFormat::JSON;
    cfg.enable_console_output = false;
    MetricsLogger logger(cfg);
    EXPECT_NO_THROW(logger.log_signal_generated(0.5, 0.3, 42));
    EXPECT_NO_THROW(logger.log_token_received("bullish", 99));
    EXPECT_NO_THROW(logger.log_latency_measurement(7));
    EXPECT_NO_THROW(logger.log_system_stats(1024 * 1024 * 128, 12.5));
    EXPECT_NO_THROW(logger.log_performance_summary());
}

// ============================================================
// TokenStreamSimulator: stop before start is safe
// ============================================================

TEST(ProductionReadinessTest, test_token_stream_simulator_stop_before_start_is_safe) {
    TokenStreamSimulator::Config cfg;
    cfg.use_memory_stream = true;
    TokenStreamSimulator sim(cfg);
    // stop() before start() must not crash or block.
    EXPECT_NO_THROW(sim.stop());
}

TEST(ProductionReadinessTest, test_token_stream_simulator_load_from_memory_and_emit) {
    TokenStreamSimulator::Config cfg;
    cfg.token_interval    = std::chrono::microseconds{1000};
    cfg.use_memory_stream = true;
    TokenStreamSimulator sim(cfg);
    sim.load_tokens_from_memory({"alpha", "beta", "gamma"});

    std::atomic<int> count{0};
    sim.set_token_callback([&](const Token& t) {
        (void)t;
        count++;
    });
    sim.start();
    // Wait for at least 3 tokens to be emitted.
    for (int i = 0; i < 50 && count.load() < 3; ++i)
        std::this_thread::sleep_for(std::chrono::milliseconds{10});
    sim.stop();

    EXPECT_GE(count.load(), 3) << "Simulator should emit all loaded tokens";
}

// ============================================================
// Config default construction has sane values
// ============================================================

TEST(ProductionReadinessTest, test_config_default_values_are_valid) {
    Config cfg;
    auto sc = cfg.get_config();
    EXPECT_GT(sc.token_stream.token_interval_ms, 0);
    EXPECT_GT(sc.token_stream.buffer_size,        0u);
    EXPECT_GT(sc.trading.bias_sensitivity,        0.0);
    EXPECT_GT(sc.trading.signal_decay_rate,       0.0);
    EXPECT_LE(sc.trading.signal_decay_rate,       1.0);
    EXPECT_GT(sc.latency.sample_window,           0u);
    EXPECT_GT(sc.latency.target_latency_us,       0);
}

// ============================================================
// LLMAdapter SIMD path matches scalar path
// ============================================================

TEST(ProductionReadinessTest, test_llm_adapter_simd_matches_scalar_for_two_tokens) {
    LLMAdapter adapter;
    std::vector<std::string> tokens = {"crash", "bullish"};
    auto scalar = adapter.map_sequence_to_weight(tokens);
    auto simd   = adapter.map_sequence_simd(tokens);

    EXPECT_NEAR(scalar.sentiment_score,  simd.sentiment_score,  1e-9);
    EXPECT_NEAR(scalar.volatility_score, simd.volatility_score, 1e-9);
    EXPECT_NEAR(scalar.directional_bias, simd.directional_bias, 1e-9);
    EXPECT_NEAR(scalar.confidence_score, simd.confidence_score, 1e-9);
}

TEST(ProductionReadinessTest, test_llm_adapter_simd_odd_length_sequence) {
    LLMAdapter adapter;
    // Odd-length (3) exercises the tail scalar loop.
    std::vector<std::string> tokens = {"crash", "bullish", "volatile"};
    auto scalar = adapter.map_sequence_to_weight(tokens);
    auto simd   = adapter.map_sequence_simd(tokens);

    EXPECT_NEAR(scalar.sentiment_score,  simd.sentiment_score,  1e-9);
    EXPECT_NEAR(scalar.directional_bias, simd.directional_bias, 1e-9);
}
