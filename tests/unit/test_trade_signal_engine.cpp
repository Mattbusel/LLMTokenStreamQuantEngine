#include "gtest/gtest.h"
#include "TradeSignalEngine.h"
#include "LLMAdapter.h"
#include "OutputSinkImpl.h"

#include <atomic>
#include <chrono>
#include <cmath>
#include <thread>
#include <vector>

namespace llmquant {
namespace {

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

static TradeSignalEngine::Config make_config(
        double bias_sens   = 1.0,
        double vol_sens    = 1.0,
        double decay       = 0.95,
        int    cooldown_us = 0)   // 0 = emit on every token
{
    TradeSignalEngine::Config cfg;
    cfg.bias_sensitivity       = bias_sens;
    cfg.volatility_sensitivity = vol_sens;
    cfg.signal_decay_rate      = decay;
    cfg.signal_cooldown        = std::chrono::microseconds{cooldown_us};
    return cfg;
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// signal_quality field
// ---------------------------------------------------------------------------

TEST(TradeSignalEngineTest, test_trade_signal_engine_signal_quality_in_unit_interval) {
    TradeSignalEngine engine(make_config());
    engine.set_backtest_mode(true);

    TradeSignal captured;
    engine.set_signal_callback([&captured](const TradeSignal& s) { captured = s; });

    SemanticWeight w{0.8, 0.9, 0.3, 0.9};
    engine.process_semantic_weight(w);

    EXPECT_GE(captured.signal_quality, 0.0) << "signal_quality must be >= 0";
    EXPECT_LE(captured.signal_quality, 1.0) << "signal_quality must be <= 1";
}

TEST(TradeSignalEngineTest, test_trade_signal_engine_signal_quality_zero_for_zero_confidence) {
    TradeSignalEngine engine(make_config());
    engine.set_backtest_mode(true);

    TradeSignal captured;
    engine.set_signal_callback([&captured](const TradeSignal& s) { captured = s; });

    SemanticWeight w{0.8, 0.0 /*zero confidence*/, 0.3, 0.5};
    engine.process_semantic_weight(w);

    EXPECT_DOUBLE_EQ(captured.signal_quality, 0.0)
        << "signal_quality must be 0 when confidence is 0";
}

TEST(TradeSignalEngineTest, test_trade_signal_engine_process_bullish_weight_produces_positive_bias) {
    TradeSignalEngine engine(make_config());
    engine.set_backtest_mode(true);  // emit every token

    TradeSignal captured;
    engine.set_signal_callback([&captured](const TradeSignal& s) { captured = s; });

    SemanticWeight bullish{0.8, 0.9, 0.3, 0.9};
    engine.process_semantic_weight(bullish);

    EXPECT_GT(captured.delta_bias_shift, 0.0)
        << "Bullish weight should produce positive bias shift";
}

TEST(TradeSignalEngineTest, test_trade_signal_engine_process_fear_weight_produces_negative_bias) {
    TradeSignalEngine engine(make_config());
    engine.set_backtest_mode(true);

    TradeSignal captured;
    engine.set_signal_callback([&captured](const TradeSignal& s) { captured = s; });

    SemanticWeight fear{-0.9, 0.85, 0.8, -0.8};
    engine.process_semantic_weight(fear);

    EXPECT_LT(captured.delta_bias_shift, 0.0)
        << "Fear weight should produce negative bias shift";
}

TEST(TradeSignalEngineTest, test_trade_signal_engine_backtest_mode_emits_every_token) {
    TradeSignalEngine engine(make_config(1.0, 1.0, 0.95, 100000 /*100 ms cooldown*/));
    engine.set_backtest_mode(true);  // ignore cooldown

    std::atomic<int> count{0};
    engine.set_signal_callback([&count](const TradeSignal&) { count++; });

    SemanticWeight w{0.5, 0.7, 0.3, 0.4};
    for (int i = 0; i < 10; ++i) {
        engine.process_semantic_weight(w);
    }

    EXPECT_EQ(count.load(), 10)
        << "Backtest mode must emit a signal for every processed weight";
}

TEST(TradeSignalEngineTest, test_trade_signal_engine_cooldown_suppresses_signals) {
    // Very long cooldown: only the first emission should fire within a tight loop.
    TradeSignalEngine engine(make_config(1.0, 1.0, 0.95, 1'000'000 /*1 s*/));
    engine.set_realtime_mode(true);

    std::atomic<int> count{0};
    engine.set_signal_callback([&count](const TradeSignal&) { count++; });

    SemanticWeight w{0.9, 0.9, 0.5, 0.9};
    for (int i = 0; i < 20; ++i) {
        engine.process_semantic_weight(w);
    }

    // At most 1 signal should have fired (the first one, before the cooldown
    // locked out subsequent emissions).
    EXPECT_LE(count.load(), 1)
        << "Cooldown must suppress rapid-fire signals";
}

TEST(TradeSignalEngineTest, test_trade_signal_engine_stats_track_generated_count) {
    TradeSignalEngine engine(make_config());
    engine.set_backtest_mode(true);
    engine.set_signal_callback([](const TradeSignal&) {});

    SemanticWeight w{0.5, 0.6, 0.3, 0.5};
    for (int i = 0; i < 5; ++i) {
        engine.process_semantic_weight(w);
    }

    EXPECT_EQ(engine.get_stats().signals_generated.load(), 5u);
}

TEST(TradeSignalEngineTest, test_trade_signal_engine_no_callback_increments_suppressed_count) {
    TradeSignalEngine engine(make_config());
    engine.set_backtest_mode(true);
    // No callback registered.

    SemanticWeight w{0.8, 0.9, 0.4, 0.7};
    for (int i = 0; i < 3; ++i) {
        engine.process_semantic_weight(w);
    }

    EXPECT_EQ(engine.get_stats().signals_suppressed.load(), 3u)
        << "Signals with no callback must be counted as suppressed";
}

TEST(TradeSignalEngineTest, test_trade_signal_engine_decay_reduces_accumulated_signal_over_time) {
    // With decay = 0.95 the accumulator shrinks toward zero when only neutral
    // (zero-contribution) tokens arrive after an initial large weight.
    TradeSignalEngine engine(make_config(1.0, 1.0, 0.95 /*decay*/, 0));
    engine.set_backtest_mode(true);

    std::vector<TradeSignal> signals;
    engine.set_signal_callback([&signals](const TradeSignal& s) { signals.push_back(s); });

    // First token: accumulate a strong directional bias.
    SemanticWeight large{0.9, 0.9, 0.5, 0.9};
    engine.process_semantic_weight(large);

    // Subsequent neutral tokens: zero contribution, so the accumulator decays
    // by the factor 0.95 on each call — |bias| must strictly decrease each time.
    SemanticWeight neutral{0.0, 0.0, 0.0, 0.0};
    engine.process_semantic_weight(neutral);
    engine.process_semantic_weight(neutral);
    engine.process_semantic_weight(neutral);

    ASSERT_GE(signals.size(), 4u);
    // Each neutral token must reduce |delta_bias_shift| compared to the previous.
    for (std::size_t i = 1; i < signals.size(); ++i) {
        EXPECT_LT(std::abs(signals[i].delta_bias_shift),
                  std::abs(signals[i - 1].delta_bias_shift) + 1e-9)
            << "decay must reduce |bias| on each neutral token (step " << i << ")";
    }
}

TEST(TradeSignalEngineTest, test_trade_signal_engine_emitted_signal_has_nonzero_timestamp_ns) {
    TradeSignalEngine engine(make_config());
    engine.set_backtest_mode(true);

    TradeSignal captured;
    engine.set_signal_callback([&captured](const TradeSignal& s) { captured = s; });

    SemanticWeight w{0.5, 0.8, 0.3, 0.6};
    engine.process_semantic_weight(w);

    EXPECT_GT(captured.timestamp_ns, 0u)
        << "Emitted signal must have a non-zero nanosecond timestamp";
    // The chrono timestamp should also be non-default (not equal to the epoch).
    auto since_epoch = captured.timestamp.time_since_epoch().count();
    EXPECT_GT(since_epoch, 0)
        << "Emitted signal must have a non-zero chrono timestamp";
}

TEST(TradeSignalEngineTest, test_trade_signal_engine_spread_modifier_nonzero_for_strong_bias) {
    TradeSignalEngine engine(make_config());
    engine.set_backtest_mode(true);

    TradeSignal captured;
    engine.set_signal_callback([&captured](const TradeSignal& s) { captured = s; });

    // Strong bullish bias — bias_sensitivity=1.0 and directional_bias=0.95 with
    // confidence=1.0 gives delta_bias_shift = 0.95 which is > 0.5.
    SemanticWeight strong_bullish{0.9, 1.0, 0.1, 0.95};
    engine.process_semantic_weight(strong_bullish);

    EXPECT_NE(captured.spread_modifier, 0.0)
        << "spread_modifier must be non-zero when |delta_bias_shift| > 0.5";
    // Tighten on bullish certainty: modifier should be negative for positive bias.
    EXPECT_LT(captured.spread_modifier, 0.0)
        << "spread_modifier must be negative for strong positive bias (tighten spread)";
}

TEST(TradeSignalEngineTest, test_trade_signal_engine_confidence_reflects_input_weight) {
    TradeSignalEngine engine(make_config());
    engine.set_backtest_mode(true);

    TradeSignal captured;
    engine.set_signal_callback([&captured](const TradeSignal& s) { captured = s; });

    const double expected_confidence = 0.77;
    SemanticWeight w{0.3, expected_confidence, 0.2, 0.4};
    engine.process_semantic_weight(w);

    EXPECT_DOUBLE_EQ(captured.confidence, expected_confidence)
        << "signal.confidence must reflect the confidence_score of the processed weight";
}

TEST(TradeSignalEngineTest, test_trade_signal_engine_get_config_returns_active_config) {
    auto cfg = make_config(1.5, 2.5, 0.9, 500);
    TradeSignalEngine engine(cfg);
    auto retrieved = engine.get_config();
    EXPECT_DOUBLE_EQ(retrieved.bias_sensitivity,       1.5);
    EXPECT_DOUBLE_EQ(retrieved.volatility_sensitivity, 2.5);
    EXPECT_DOUBLE_EQ(retrieved.signal_decay_rate,      0.9);
    EXPECT_EQ(retrieved.signal_cooldown, std::chrono::microseconds{500});

    // After update_config the accessor must reflect the new values.
    engine.update_config(make_config(3.0, 4.0, 0.8, 0));
    auto updated = engine.get_config();
    EXPECT_DOUBLE_EQ(updated.bias_sensitivity,       3.0);
    EXPECT_DOUBLE_EQ(updated.volatility_sensitivity, 4.0);
}

TEST(TradeSignalEngineTest, test_trade_signal_engine_update_config_invalid_params_throw) {
    TradeSignalEngine engine(make_config());

    TradeSignalEngine::Config bad_bias = make_config();
    bad_bias.bias_sensitivity = 0.0;
    EXPECT_THROW(engine.update_config(bad_bias), std::invalid_argument);

    TradeSignalEngine::Config bad_vol = make_config();
    bad_vol.volatility_sensitivity = -1.0;
    EXPECT_THROW(engine.update_config(bad_vol), std::invalid_argument);

    TradeSignalEngine::Config bad_decay = make_config();
    bad_decay.signal_decay_rate = 0.0;
    EXPECT_THROW(engine.update_config(bad_decay), std::invalid_argument);
}

TEST(TradeSignalEngineTest, test_trade_signal_engine_update_config_changes_sensitivity) {
    TradeSignalEngine engine(make_config(1.0 /*bias_sens*/, 1.0 /*vol_sens*/, 0.95, 0));
    engine.set_backtest_mode(true);

    TradeSignal captured;
    engine.set_signal_callback([&captured](const TradeSignal& s) { captured = s; });

    // Baseline: bias_sensitivity=1.0 produces some delta_bias_shift.
    SemanticWeight w{0.5, 1.0, 0.0, 0.5};
    engine.process_semantic_weight(w);
    double baseline = captured.delta_bias_shift;
    EXPECT_GT(std::fabs(baseline), 0.0);

    // Reset accumulators so the comparison is clean.
    engine.reset();

    // Double the bias sensitivity — same token should produce ~2x the shift.
    TradeSignalEngine::Config cfg = make_config(2.0, 1.0, 0.95, 0);
    engine.update_config(cfg);
    engine.process_semantic_weight(w);
    double scaled = captured.delta_bias_shift;
    EXPECT_GT(std::fabs(scaled), std::fabs(baseline) * 1.5)
        << "Doubling bias_sensitivity must produce a substantially larger bias shift";
}

TEST(TradeSignalEngineTest, test_trade_signal_engine_latency_us_is_populated) {
    TradeSignalEngine engine(make_config());
    engine.set_backtest_mode(true);

    TradeSignal captured;
    engine.set_signal_callback([&captured](const TradeSignal& s) { captured = s; });

    SemanticWeight w{0.5, 0.7, 0.3, 0.6};
    engine.process_semantic_weight(w);

    // latency_us must be non-negative and finite (not left at default 0.0 is not
    // strictly required, but must be a valid measured duration >= 0).
    EXPECT_GE(captured.latency_us, 0.0)
        << "latency_us must be a non-negative duration";
    EXPECT_TRUE(std::isfinite(captured.latency_us))
        << "latency_us must be finite";
}

TEST(TradeSignalEngineTest, test_trade_signal_engine_flush_sinks_does_not_crash) {
    TradeSignalEngine engine(make_config());
    auto sink = std::make_shared<MemoryOutputSink>();
    engine.add_output_sink(sink);
    // Should not throw or crash even with no pending data.
    EXPECT_NO_THROW(engine.flush_sinks());
}

TEST(TradeSignalEngineTest, test_trade_signal_engine_reset_clears_accumulators_and_stats) {
    TradeSignalEngine engine(make_config());
    engine.set_backtest_mode(true);

    std::atomic<int> count{0};
    engine.set_signal_callback([&count](const TradeSignal&) { count++; });

    // Accumulate some state.
    SemanticWeight bullish{0.8, 0.9, 0.3, 0.9};
    for (int i = 0; i < 5; ++i) {
        engine.process_semantic_weight(bullish);
    }
    EXPECT_GT(engine.get_stats().signals_generated.load(), 0u);
    EXPECT_NE(engine.get_accumulated_bias(), 0.0);

    engine.reset();

    // After reset, accumulators and stats must be zero.
    EXPECT_EQ(engine.get_stats().signals_generated.load(), 0u);
    EXPECT_EQ(engine.get_stats().signals_suppressed.load(), 0u);
    EXPECT_DOUBLE_EQ(engine.get_accumulated_bias(),      0.0);
    EXPECT_DOUBLE_EQ(engine.get_accumulated_volatility(), 0.0);

    // Engine must still be operational after reset: a new weight must produce a signal.
    count = 0;
    engine.process_semantic_weight(bullish);
    EXPECT_EQ(count.load(), 1) << "Engine must emit signals normally after reset()";
    EXPECT_EQ(engine.get_stats().signals_generated.load(), 1u);
}

TEST(TradeSignalEngineTest, test_trade_signal_engine_staleness_guard_config_stored) {
    // Verify that max_signal_age_us is stored and retrievable via get_config().
    TradeSignalEngine::Config cfg = make_config();
    cfg.max_signal_age_us = 500.0;
    TradeSignalEngine engine(cfg);
    EXPECT_DOUBLE_EQ(engine.get_config().max_signal_age_us, 500.0);

    // Large threshold (1 second) must never suppress signals in normal operation.
    engine.set_backtest_mode(true);
    std::atomic<int> emitted{0};
    engine.set_signal_callback([&emitted](const TradeSignal&) { ++emitted; });

    SemanticWeight w{0.5, 0.7, 0.3, 0.6};
    for (int i = 0; i < 5; ++i) {
        engine.process_semantic_weight(w);
    }
    EXPECT_EQ(emitted.load(), 5) << "Large max_signal_age_us must not suppress normal signals";
    EXPECT_EQ(engine.get_stats().signals_aged_out.load(), 0u);
}

TEST(TradeSignalEngineTest, test_trade_signal_engine_staleness_guard_disabled_by_default) {
    // Default config has max_signal_age_us = 0.0 (disabled).
    TradeSignalEngine engine(make_config());
    engine.set_backtest_mode(true);

    std::atomic<int> emitted{0};
    engine.set_signal_callback([&emitted](const TradeSignal&) { ++emitted; });

    SemanticWeight w{0.5, 0.7, 0.3, 0.6};
    engine.process_semantic_weight(w);

    EXPECT_EQ(emitted.load(), 1) << "Staleness guard disabled by default must not suppress signals";
    EXPECT_EQ(engine.get_stats().signals_aged_out.load(), 0u);
}

TEST(TradeSignalEngineTest, test_trade_signal_engine_min_bias_threshold_suppresses_weak_signals) {
    // Set a threshold so high that one token can never reach it in one step.
    TradeSignalEngine::Config cfg = make_config();
    cfg.min_bias_threshold = 10.0;  // accumulated bias after one token << 10.0
    TradeSignalEngine engine(cfg);
    engine.set_backtest_mode(true);

    std::atomic<int> emitted{0};
    engine.set_signal_callback([&emitted](const TradeSignal&) { ++emitted; });

    SemanticWeight w{0.9, 0.9, 0.2, 0.9};
    engine.process_semantic_weight(w);

    EXPECT_EQ(emitted.load(), 0)
        << "Accumulated bias below min_bias_threshold must be suppressed";
    EXPECT_GT(engine.get_stats().signals_suppressed.load(), 0u);
}

TEST(TradeSignalEngineTest, test_trade_signal_engine_min_bias_threshold_disabled_by_default) {
    // Default config has min_bias_threshold = 0.0 (disabled).
    TradeSignalEngine engine(make_config());
    engine.set_backtest_mode(true);

    std::atomic<int> emitted{0};
    engine.set_signal_callback([&emitted](const TradeSignal&) { ++emitted; });

    SemanticWeight w{0.5, 0.8, 0.3, 0.6};
    engine.process_semantic_weight(w);

    EXPECT_EQ(emitted.load(), 1)
        << "Noise filter disabled by default must not suppress signals";
    EXPECT_EQ(engine.get_stats().signals_suppressed.load(), 0u);
}

TEST(TradeSignalEngineTest, test_trade_signal_engine_multiple_sinks_all_receive_signal) {
    TradeSignalEngine engine(make_config());
    engine.set_backtest_mode(true);

    auto sink1 = std::make_shared<MemoryOutputSink>();
    auto sink2 = std::make_shared<MemoryOutputSink>();
    engine.add_output_sink(sink1);
    engine.add_output_sink(sink2);

    SemanticWeight w{0.5, 0.7, 0.3, 0.6};
    engine.process_semantic_weight(w);

    EXPECT_EQ(sink1->size(), 1u) << "First sink must receive the signal";
    EXPECT_EQ(sink2->size(), 1u) << "Second sink must receive the signal";
    // Both sinks must have the same signal content.
    EXPECT_DOUBLE_EQ(sink1->get_signals()[0].delta_bias_shift,
                     sink2->get_signals()[0].delta_bias_shift);
}

TEST(TradeSignalEngineTest, test_trade_signal_engine_clear_sinks_removes_all) {
    TradeSignalEngine engine(make_config());
    engine.set_backtest_mode(true);

    auto sink = std::make_shared<MemoryOutputSink>();
    engine.add_output_sink(sink);
    engine.clear_output_sinks();

    // No callback, no sinks → signals must be suppressed.
    SemanticWeight w{0.5, 0.7, 0.3, 0.6};
    engine.process_semantic_weight(w);
    EXPECT_EQ(sink->size(), 0u) << "Sink removed by clear_output_sinks must not receive signals";
    EXPECT_EQ(engine.get_stats().signals_suppressed.load(), 1u);
}

// ---------------------------------------------------------------------------
// max_accumulated_bias cap
// ---------------------------------------------------------------------------

TEST(TradeSignalEngineTest, test_trade_signal_engine_max_accumulated_bias_clamps_accumulator) {
    // With a cap of 1.0 and no decay (decay=1.0 not allowed; use 0.999 ≈ no decay),
    // feed many strongly-bullish tokens and verify |accumulated_bias| never exceeds 1.0.
    TradeSignalEngine::Config cfg = make_config(1.0, 1.0, 0.999, 0);
    cfg.max_accumulated_bias = 1.0;
    TradeSignalEngine engine(cfg);
    engine.set_backtest_mode(true);
    engine.set_signal_callback([](const TradeSignal&){});  // consume signals

    SemanticWeight strong{1.0, 1.0, 0.5, 1.0};  // bias_contribution = 1.0 per token
    for (int i = 0; i < 50; ++i) {
        engine.process_semantic_weight(strong);
        double bias = engine.get_accumulated_bias();
        EXPECT_LE(std::fabs(bias), 1.0 + 1e-9)
            << "accumulated_bias exceeded max_accumulated_bias cap at iteration " << i;
    }
}

// ---------------------------------------------------------------------------
// add_sink_with_filter
// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// accumulator_clamped counter
// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// peak_bias stat
// ---------------------------------------------------------------------------

TEST(TradeSignalEngineTest, test_trade_signal_engine_peak_bias_tracks_maximum) {
    TradeSignalEngine::Config cfg = make_config(1.0, 1.0, 0.999, 0);
    cfg.min_bias_threshold = 100.0;  // suppress emissions / halving
    TradeSignalEngine engine(cfg);
    engine.set_backtest_mode(true);

    EXPECT_DOUBLE_EQ(engine.get_stats().peak_bias.load(), 0.0)
        << "peak_bias must be 0 initially";

    SemanticWeight strong{1.0, 1.0, 0.5, 1.0};
    for (int i = 0; i < 10; ++i) engine.process_semantic_weight(strong);

    double peak = engine.get_stats().peak_bias.load();
    EXPECT_GT(peak, 0.0) << "peak_bias must be > 0 after processing bullish tokens";
    // peak must be >= current accumulated bias
    EXPECT_GE(peak, std::fabs(engine.get_accumulated_bias()) - 1e-9);
}

TEST(TradeSignalEngineTest, test_trade_signal_engine_peak_bias_reset_clears) {
    TradeSignalEngine::Config cfg = make_config(1.0, 1.0, 0.999, 0);
    cfg.min_bias_threshold = 100.0;
    TradeSignalEngine engine(cfg);
    engine.set_backtest_mode(true);

    SemanticWeight strong{1.0, 1.0, 0.5, 1.0};
    for (int i = 0; i < 5; ++i) engine.process_semantic_weight(strong);
    ASSERT_GT(engine.get_stats().peak_bias.load(), 0.0);

    engine.reset();
    EXPECT_DOUBLE_EQ(engine.get_stats().peak_bias.load(), 0.0)
        << "peak_bias must be 0 after reset()";
}

TEST(TradeSignalEngineTest, test_trade_signal_engine_accumulator_clamped_increments_on_cap) {
    TradeSignalEngine::Config cfg = make_config(1.0, 1.0, 0.999, 0);
    cfg.max_accumulated_bias = 1.0;
    cfg.min_bias_threshold   = 100.0;  // suppress emissions → no halving
    TradeSignalEngine engine(cfg);
    engine.set_backtest_mode(true);

    SemanticWeight strong{1.0, 1.0, 0.5, 1.0};
    // First token: desired_bias = 0*0.999 + 1.0 = 1.0. clamp(1.0, -1.0, 1.0) = 1.0 → no clamp.
    // Second token: desired_bias = 1.0*0.999 + 1.0 = 1.999. Clamped to 1.0 → counter++.
    for (int i = 0; i < 10; ++i) {
        engine.process_semantic_weight(strong);
    }
    EXPECT_GT(engine.get_stats().accumulator_clamped.load(), uint64_t{0})
        << "accumulator_clamped must increment when cap is applied";
}

TEST(TradeSignalEngineTest, test_trade_signal_engine_accumulator_clamped_zero_when_cap_disabled) {
    TradeSignalEngine::Config cfg = make_config(1.0, 1.0, 0.5, 0);
    cfg.max_accumulated_bias = 0.0;  // disabled
    cfg.min_bias_threshold   = 100.0;
    TradeSignalEngine engine(cfg);
    engine.set_backtest_mode(true);

    SemanticWeight strong{1.0, 1.0, 0.5, 1.0};
    for (int i = 0; i < 10; ++i) {
        engine.process_semantic_weight(strong);
    }
    EXPECT_EQ(engine.get_stats().accumulator_clamped.load(), uint64_t{0})
        << "accumulator_clamped must remain 0 when max_accumulated_bias is disabled";
}

TEST(TradeSignalEngineTest, test_trade_signal_engine_accumulator_clamped_reset_clears_counter) {
    TradeSignalEngine::Config cfg = make_config(1.0, 1.0, 0.999, 0);
    cfg.max_accumulated_bias = 1.0;
    cfg.min_bias_threshold   = 100.0;
    TradeSignalEngine engine(cfg);
    engine.set_backtest_mode(true);

    SemanticWeight strong{1.0, 1.0, 0.5, 1.0};
    for (int i = 0; i < 5; ++i) engine.process_semantic_weight(strong);
    ASSERT_GT(engine.get_stats().accumulator_clamped.load(), uint64_t{0});

    engine.reset();
    EXPECT_EQ(engine.get_stats().accumulator_clamped.load(), uint64_t{0})
        << "accumulator_clamped must be 0 after reset()";
}

TEST(TradeSignalEngineTest, test_trade_signal_engine_filtered_sink_only_receives_matching_signals) {
    TradeSignalEngine engine(make_config());
    engine.set_backtest_mode(true);

    auto bullish_sink  = std::make_shared<MemoryOutputSink>();
    auto bearish_sink  = std::make_shared<MemoryOutputSink>();
    auto all_sink      = std::make_shared<MemoryOutputSink>();

    // Only bullish (positive bias) signals reach bullish_sink.
    engine.add_sink_with_filter(bullish_sink,
        [](const TradeSignal& s){ return s.delta_bias_shift > 0.0; });
    // Only bearish (negative bias) signals reach bearish_sink.
    engine.add_sink_with_filter(bearish_sink,
        [](const TradeSignal& s){ return s.delta_bias_shift < 0.0; });
    // Unfiltered sink receives everything.
    engine.add_output_sink(all_sink);

    SemanticWeight bullish{0.8, 0.6, 0.2, 0.9};
    SemanticWeight bearish{-0.8, 0.6, 0.2, 0.9};

    engine.process_semantic_weight(bullish);
    engine.process_semantic_weight(bearish);
    // After bullish + bearish the accumulated bias may be near-zero; emit a
    // second strong bullish token to ensure at least one bullish signal fires.
    engine.process_semantic_weight(bullish);

    // all_sink must have received every signal.
    EXPECT_GT(all_sink->size(), 0u) << "Unfiltered sink must receive all signals";
    // bullish_sink must have at least one positive-bias signal.
    EXPECT_GT(bullish_sink->size(), 0u) << "Bullish sink must receive at least one bullish signal";
    // bearish signals must not arrive in bullish_sink and vice-versa.
    for (const auto& s : bullish_sink->get_signals()) {
        EXPECT_GT(s.delta_bias_shift, 0.0) << "Bullish sink must only contain positive-bias signals";
    }
    for (const auto& s : bearish_sink->get_signals()) {
        EXPECT_LT(s.delta_bias_shift, 0.0) << "Bearish sink must only contain negative-bias signals";
    }
}

TEST(TradeSignalEngineTest, test_trade_signal_engine_filtered_sink_cleared_by_clear_output_sinks) {
    TradeSignalEngine engine(make_config());
    engine.set_backtest_mode(true);

    auto sink = std::make_shared<MemoryOutputSink>();
    engine.add_sink_with_filter(sink, [](const TradeSignal&){ return true; });
    engine.clear_output_sinks();  // must remove filtered sinks too

    engine.set_signal_callback([](const TradeSignal&){});
    SemanticWeight w{0.5, 0.5, 0.1, 0.8};
    engine.process_semantic_weight(w);

    EXPECT_EQ(sink->size(), 0u) << "Sink removed by clear_output_sinks must not receive signals";
}

// ---------------------------------------------------------------------------
// process_batch
// ---------------------------------------------------------------------------

TEST(TradeSignalEngineTest, test_trade_signal_engine_process_batch_emits_n_signals_in_backtest) {
    TradeSignalEngine engine(make_config());
    engine.set_backtest_mode(true);

    std::atomic<int> count{0};
    engine.set_signal_callback([&count](const TradeSignal&) { ++count; });

    std::vector<SemanticWeight> batch = {
        {0.5, 0.7, 0.3, 0.6},
        {0.4, 0.8, 0.2, 0.7},
        {0.6, 0.9, 0.1, 0.5},
    };
    engine.process_batch(batch);
    EXPECT_EQ(count.load(), 3) << "process_batch must emit one signal per token in backtest mode";
}

TEST(TradeSignalEngineTest, test_trade_signal_engine_process_batch_empty_emits_nothing) {
    TradeSignalEngine engine(make_config());
    engine.set_backtest_mode(true);

    std::atomic<int> count{0};
    engine.set_signal_callback([&count](const TradeSignal&) { ++count; });

    engine.process_batch({});
    EXPECT_EQ(count.load(), 0) << "Empty batch must emit no signals";
}

TEST(TradeSignalEngineTest, test_trade_signal_engine_max_accumulated_bias_zero_disabled) {
    // With max_accumulated_bias=0.0 (disabled), the accumulator is free to grow above 1.0.
    // Use a large min_bias_threshold to suppress all signal emissions (and the post-emit
    // accumulator halving) so the accumulator grows unconstrained.
    TradeSignalEngine::Config cfg = make_config(1.0, 1.0, 0.999, 0);
    cfg.max_accumulated_bias = 0.0;   // disabled — no clamp
    cfg.min_bias_threshold   = 100.0; // suppress all emissions → no post-emit halving
    TradeSignalEngine engine(cfg);
    engine.set_backtest_mode(true);

    SemanticWeight strong{1.0, 1.0, 0.5, 1.0};  // bias_contribution = 1.0 per token
    for (int i = 0; i < 30; ++i) {
        engine.process_semantic_weight(strong);
    }
    // After 30 tokens with decay≈1 and contribution=1.0, the geometric series
    // sum = (1 - 0.999^30) / (1 - 0.999) ≈ 28.6 >> 1.0.
    EXPECT_GT(engine.get_accumulated_bias(), 1.0)
        << "Without cap, accumulator should exceed 1.0 after many strongly-bullish tokens";
}

// ---------------------------------------------------------------------------
// tokens_processed counter
// ---------------------------------------------------------------------------

TEST(TradeSignalEngineTest, test_trade_signal_engine_tokens_processed_increments_per_token) {
    TradeSignalEngine engine(make_config());
    engine.set_backtest_mode(true);

    EXPECT_EQ(engine.get_stats().tokens_processed.load(), 0u);

    SemanticWeight w{0.5, 0.5, 0.2, 0.8};
    engine.process_semantic_weight(w);
    EXPECT_EQ(engine.get_stats().tokens_processed.load(), 1u);

    engine.process_semantic_weight(w);
    engine.process_semantic_weight(w);
    EXPECT_EQ(engine.get_stats().tokens_processed.load(), 3u);
}

// ---------------------------------------------------------------------------
// suppression_rate()
// ---------------------------------------------------------------------------

TEST(TradeSignalEngineTest, test_suppression_rate_zero_when_no_suppression) {
    // With min_bias_threshold=0 nothing is suppressed by the noise filter.
    // Attach a sink so signals have a destination and are not counted as suppressed.
    TradeSignalEngine engine(make_config());
    engine.set_backtest_mode(true);
    auto sink = std::make_shared<MemoryOutputSink>();
    engine.add_output_sink(sink);

    EXPECT_DOUBLE_EQ(engine.suppression_rate(), 0.0)
        << "Rate must be 0.0 before any tokens are processed";

    SemanticWeight w{0.5, 0.5, 0.2, 0.9};
    engine.process_semantic_weight(w);  // should generate a signal, not suppress it

    // After one generated signal with zero suppressions, rate must still be 0.
    EXPECT_DOUBLE_EQ(engine.suppression_rate(), 0.0);
}

TEST(TradeSignalEngineTest, test_suppression_rate_approaches_one_with_all_suppressed) {
    // Set min_bias_threshold very high so every signal is suppressed.
    TradeSignalEngine::Config cfg = make_config(1.0, 1.0, 0.95, 0);
    cfg.min_bias_threshold = 100.0;  // suppress everything
    TradeSignalEngine engine(cfg);
    engine.set_backtest_mode(true);

    SemanticWeight w{0.1, 0.1, 0.1, 0.9};
    for (int i = 0; i < 10; ++i) engine.process_semantic_weight(w);

    double rate = engine.suppression_rate();
    // All signals suppressed → rate should be 1.0 (or very close).
    EXPECT_GE(rate, 0.5) << "Most signals should be suppressed when threshold is very high";
}

TEST(TradeSignalEngineTest, test_trade_signal_engine_tokens_processed_resets_on_reset) {
    TradeSignalEngine engine(make_config());
    engine.set_backtest_mode(true);

    SemanticWeight w{0.5, 0.5, 0.2, 0.8};
    for (int i = 0; i < 10; ++i) engine.process_semantic_weight(w);
    ASSERT_EQ(engine.get_stats().tokens_processed.load(), 10u);

    engine.reset();
    EXPECT_EQ(engine.get_stats().tokens_processed.load(), 0u);
}

// ---------------------------------------------------------------------------
// get_last_signal_quality
// ---------------------------------------------------------------------------

TEST(TradeSignalEngineTest, test_last_signal_quality_zero_before_emission) {
    TradeSignalEngine engine(make_config());
    EXPECT_DOUBLE_EQ(engine.get_last_signal_quality(), 0.0);
}

TEST(TradeSignalEngineTest, test_last_signal_quality_in_unit_interval_after_emission) {
    TradeSignalEngine engine(make_config());
    engine.set_backtest_mode(true);
    engine.set_signal_callback([](const TradeSignal&){});

    SemanticWeight w{0.8, 0.9, 0.3, 0.9};
    engine.process_semantic_weight(w);

    double q = engine.get_last_signal_quality();
    EXPECT_GE(q, 0.0);
    EXPECT_LE(q, 1.0);
}

TEST(TradeSignalEngineTest, test_last_signal_quality_reset_clears_to_zero) {
    TradeSignalEngine engine(make_config());
    engine.set_backtest_mode(true);
    engine.set_signal_callback([](const TradeSignal&){});

    engine.process_semantic_weight({0.8, 0.9, 0.3, 0.9});
    ASSERT_GT(engine.get_last_signal_quality(), 0.0);

    engine.reset();
    EXPECT_DOUBLE_EQ(engine.get_last_signal_quality(), 0.0);
}

// ---------------------------------------------------------------------------
// noise_filtered counter
// ---------------------------------------------------------------------------

TEST(TradeSignalEngineTest, test_noise_filtered_zero_with_no_threshold) {
    // Default min_bias_threshold=0.0 means noise gate is disabled.
    TradeSignalEngine engine(make_config());
    engine.set_backtest_mode(true);
    engine.set_signal_callback([](const TradeSignal&) {});

    SemanticWeight w{0.5, 0.5, 0.2, 0.9};
    engine.process_semantic_weight(w);

    EXPECT_EQ(engine.get_stats().noise_filtered.load(), 0u)
        << "noise_filtered must be 0 when min_bias_threshold is disabled";
}

TEST(TradeSignalEngineTest, test_noise_filtered_counts_threshold_rejections) {
    TradeSignalEngine::Config cfg = make_config(1.0, 1.0, 0.999, 0);
    cfg.min_bias_threshold = 100.0;  // suppress everything
    TradeSignalEngine engine(cfg);
    engine.set_backtest_mode(true);

    SemanticWeight w{0.1, 0.1, 0.1, 0.9};
    for (int i = 0; i < 5; ++i) engine.process_semantic_weight(w);

    EXPECT_EQ(engine.get_stats().noise_filtered.load(), 5u)
        << "noise_filtered must count every noise-gate rejection";
    // noise_filtered must equal signals_suppressed in this pure-threshold scenario
    // (no unrouted signals since nothing is emitted).
    EXPECT_EQ(engine.get_stats().noise_filtered.load(),
              engine.get_stats().signals_suppressed.load());
}

TEST(TradeSignalEngineTest, test_noise_filtered_resets_on_reset) {
    TradeSignalEngine::Config cfg = make_config(1.0, 1.0, 0.999, 0);
    cfg.min_bias_threshold = 100.0;
    TradeSignalEngine engine(cfg);
    engine.set_backtest_mode(true);

    SemanticWeight w{0.1, 0.1, 0.1, 0.9};
    engine.process_semantic_weight(w);
    ASSERT_GT(engine.get_stats().noise_filtered.load(), 0u);

    engine.reset();
    EXPECT_EQ(engine.get_stats().noise_filtered.load(), 0u)
        << "noise_filtered must be zero after reset()";
}

// ---------------------------------------------------------------------------
// get_signal_age_us()
// ---------------------------------------------------------------------------

TEST(TradeSignalEngineTest, test_get_signal_age_us_zero_before_first_signal) {
    TradeSignalEngine engine(make_config());
    EXPECT_EQ(engine.get_signal_age_us(), 0.0);
}

TEST(TradeSignalEngineTest, test_get_signal_age_us_positive_after_emission) {
    TradeSignalEngine engine(make_config());
    engine.set_backtest_mode(true);
    engine.set_signal_callback([](const TradeSignal&) {});

    SemanticWeight w{0.5, 0.5, 0.2, 0.9};
    engine.process_semantic_weight(w);

    // After a signal is emitted, the age must be >= 0 (may be near zero but not negative).
    EXPECT_GE(engine.get_signal_age_us(), 0.0)
        << "Signal age must be non-negative after emission";
}

TEST(TradeSignalEngineTest, test_get_signal_age_us_resets_to_zero_on_reset) {
    TradeSignalEngine engine(make_config());
    engine.set_backtest_mode(true);
    engine.set_signal_callback([](const TradeSignal&) {});

    SemanticWeight w{0.5, 0.5, 0.2, 0.9};
    engine.process_semantic_weight(w);
    ASSERT_GE(engine.get_signal_age_us(), 0.0);

    engine.reset();
    EXPECT_EQ(engine.get_signal_age_us(), 0.0);
}

// ---------------------------------------------------------------------------
// drain_pending()
// ---------------------------------------------------------------------------

TEST(TradeSignalEngineTest, test_drain_pending_emits_signal_when_bias_nonzero) {
    TradeSignalEngine::Config cfg = make_config(1.0, 1.0, 0.95, 0);
    TradeSignalEngine engine(cfg);
    engine.set_backtest_mode(true);

    // Accumulate some bias without emitting — disable backtest so cooldown blocks.
    engine.set_realtime_mode(true);
    // Set cooldown high so no signal emits during accumulation.
    cfg.signal_cooldown = std::chrono::microseconds{1'000'000};  // 1 s
    engine.update_config(cfg);

    // Process weight — cooldown prevents emission.
    SemanticWeight w{0.5, 0.5, 0.2, 0.9};
    engine.process_semantic_weight(w);

    std::atomic<int> drain_signals{0};
    engine.set_signal_callback([&drain_signals](const TradeSignal&) { ++drain_signals; });

    engine.drain_pending();

    // drain_pending must have emitted exactly one signal.
    EXPECT_GT(drain_signals.load(), 0) << "drain_pending must emit a signal when bias is non-zero";
    // After drain, accumulators are cleared.
    EXPECT_DOUBLE_EQ(engine.get_accumulated_bias(), 0.0);
    EXPECT_EQ(engine.get_stats().signals_generated.load(), 0u) << "reset() clears stats";
}

TEST(TradeSignalEngineTest, test_drain_pending_no_signal_when_bias_zero) {
    TradeSignalEngine engine(make_config());
    int drain_count = 0;
    engine.set_signal_callback([&drain_count](const TradeSignal&) { ++drain_count; });

    // No tokens processed — bias is 0, drain should not emit.
    engine.drain_pending();
    EXPECT_EQ(drain_count, 0) << "drain_pending must not emit when bias is 0";
}

// ---------------------------------------------------------------------------
// set_signal_cooldown
// ---------------------------------------------------------------------------

TEST(TradeSignalEngineTest, test_set_signal_cooldown_reflects_in_get_config) {
    TradeSignalEngine engine(make_config());
    engine.set_signal_cooldown(std::chrono::microseconds{5000});
    EXPECT_EQ(engine.get_config().signal_cooldown.count(), 5000)
        << "set_signal_cooldown must be visible via get_config()";
}

TEST(TradeSignalEngineTest, test_set_signal_cooldown_zero_allows_every_signal_in_realtime) {
    auto cfg = make_config();
    cfg.signal_cooldown = std::chrono::microseconds{0};
    TradeSignalEngine engine(cfg);
    engine.set_realtime_mode(true);
    engine.set_signal_cooldown(std::chrono::microseconds{0});

    std::atomic<int> count{0};
    engine.set_signal_callback([&count](const TradeSignal&) { ++count; });

    SemanticWeight w{0.5, 0.9, 0.2, 0.8};
    engine.process_semantic_weight(w);
    engine.process_semantic_weight(w);
    // With zero cooldown every token that passes the noise gate should emit.
    EXPECT_GE(count.load(), 1)
        << "zero cooldown should not suppress consecutive signals";
}

TEST(TradeSignalEngineTest, test_set_signal_cooldown_large_suppresses_rapid_signals) {
    auto cfg = make_config();
    TradeSignalEngine engine(cfg);
    engine.set_realtime_mode(true);
    engine.set_signal_cooldown(std::chrono::microseconds{1000000}); // 1 second

    std::atomic<int> count{0};
    engine.set_signal_callback([&count](const TradeSignal&) { ++count; });

    SemanticWeight w{0.5, 0.9, 0.2, 0.8};
    engine.process_semantic_weight(w);
    engine.process_semantic_weight(w);
    engine.process_semantic_weight(w);
    // Only the first token can emit within a 1-second cooldown window.
    EXPECT_LE(count.load(), 1)
        << "large cooldown should suppress all but the first signal";
}

// ---------------------------------------------------------------------------
// Cycle 33: TradeSignalEngine::snapshot()
// ---------------------------------------------------------------------------

TEST(TradeSignalEngineTest, test_snapshot_default_state) {
    TradeSignalEngine engine(make_config());
    auto snap = engine.snapshot();

    EXPECT_DOUBLE_EQ(snap.accumulated_bias,       0.0);
    EXPECT_DOUBLE_EQ(snap.accumulated_volatility, 0.0);
    EXPECT_DOUBLE_EQ(snap.last_signal_quality,    0.0);
    EXPECT_EQ(snap.stats.signals_generated.load(), 0u);
    EXPECT_TRUE(snap.realtime_mode)
        << "Default snapshot must show realtime_mode=true";
}

TEST(TradeSignalEngineTest, test_snapshot_reflects_processed_signal) {
    TradeSignalEngine engine(make_config());
    SemanticWeight w{0.5, 0.9, 0.2, 0.8};

    int emitted = 0;
    engine.set_signal_callback([&emitted](const TradeSignal&) { ++emitted; });
    engine.set_realtime_mode(false);  // backtest: emit on every token
    engine.process_semantic_weight(w);

    auto snap = engine.snapshot();
    EXPECT_GT(snap.stats.signals_generated.load(), 0u)
        << "snapshot must reflect signals generated after processing";
    EXPECT_NE(snap.accumulated_bias, 0.0)
        << "snapshot accumulated_bias must be non-zero after processing";
}

TEST(TradeSignalEngineTest, test_snapshot_suppression_rate_zero_with_no_suppressed) {
    TradeSignalEngine engine(make_config());
    engine.set_realtime_mode(false);
    engine.set_signal_callback([](const TradeSignal&) {});
    engine.process_semantic_weight({0.5, 0.9, 0.2, 0.8});

    auto snap = engine.snapshot();
    EXPECT_DOUBLE_EQ(snap.suppression_rate_val, 0.0)
        << "No signals should be suppressed with default config";
}

// ---------------------------------------------------------------------------
// Cycle 33: get_tokens_per_second()
// ---------------------------------------------------------------------------

TEST(TradeSignalEngineTest, test_get_tokens_per_second_zero_before_processing) {
    TradeSignalEngine engine(make_config());
    EXPECT_DOUBLE_EQ(engine.get_tokens_per_second(), 0.0)
        << "tokens_per_second must be 0 before any tokens are processed";
}

TEST(TradeSignalEngineTest, test_get_tokens_per_second_positive_after_processing) {
    TradeSignalEngine engine(make_config());
    engine.set_realtime_mode(false);
    engine.set_signal_callback([](const TradeSignal&) {});

    for (int i = 0; i < 10; ++i)
        engine.process_semantic_weight({0.5, 0.9, 0.2, 0.8});

    // Sleep to ensure elapsed_us >= 1.0 so the guard doesn't suppress the rate.
    std::this_thread::sleep_for(std::chrono::milliseconds(2));

    EXPECT_GT(engine.get_tokens_per_second(), 0.0)
        << "tokens_per_second must be > 0 after processing tokens";
}

// ---------------------------------------------------------------------------
// Cycle 34: get_session_duration_ms()
// ---------------------------------------------------------------------------

TEST(TradeSignalEngineTest, test_get_session_duration_ms_non_negative_at_start) {
    TradeSignalEngine engine(make_config());
    EXPECT_GE(engine.get_session_duration_ms(), 0.0)
        << "session duration must be non-negative immediately after construction";
}

TEST(TradeSignalEngineTest, test_get_session_duration_ms_increases_over_time) {
    TradeSignalEngine engine(make_config());
    double d1 = engine.get_session_duration_ms();
    std::this_thread::sleep_for(std::chrono::milliseconds(5));
    double d2 = engine.get_session_duration_ms();
    EXPECT_GE(d2, d1) << "session duration must not decrease";
}

TEST(TradeSignalEngineTest, test_get_session_duration_ms_resets_on_reset) {
    TradeSignalEngine engine(make_config());
    std::this_thread::sleep_for(std::chrono::milliseconds(5));
    double before_reset = engine.get_session_duration_ms();
    EXPECT_GE(before_reset, 0.0);
    engine.reset();
    // After reset, the session clock restarts — duration should be less than before.
    double after_reset = engine.get_session_duration_ms();
    EXPECT_LE(after_reset, before_reset)
        << "session duration must restart from 0 after reset()";
}

TEST(TradeSignalEngineTest, test_set_min_bias_threshold_reflects_in_get_config) {
    TradeSignalEngine engine(make_config());
    engine.set_min_bias_threshold(0.25);
    EXPECT_NEAR(engine.get_config().min_bias_threshold, 0.25, 1e-9)
        << "set_min_bias_threshold must update min_bias_threshold in config";
}

TEST(TradeSignalEngineTest, test_set_min_bias_threshold_negative_clamped_to_zero) {
    TradeSignalEngine engine(make_config());
    engine.set_min_bias_threshold(-1.0);
    EXPECT_DOUBLE_EQ(engine.get_config().min_bias_threshold, 0.0)
        << "negative threshold must be clamped to 0.0";
}

TEST(TradeSignalEngineTest, test_set_min_bias_threshold_suppresses_weak_signal) {
    auto cfg = make_config();
    TradeSignalEngine engine(cfg);
    engine.set_realtime_mode(false);
    engine.set_min_bias_threshold(10.0); // extremely high threshold

    std::atomic<int> count{0};
    engine.set_signal_callback([&count](const TradeSignal&) { ++count; });

    SemanticWeight w{0.01, 0.9, 0.01, 0.01};  // tiny bias
    engine.process_semantic_weight(w);
    EXPECT_EQ(count.load(), 0)
        << "weak signal below threshold must be suppressed";
}

// ---------------------------------------------------------------------------
// Cycle 36: get_noise_filter_rate()
// ---------------------------------------------------------------------------

TEST(TradeSignalEngineTest, test_get_noise_filter_rate_zero_before_processing) {
    TradeSignalEngine engine(make_config());
    EXPECT_DOUBLE_EQ(engine.get_noise_filter_rate(), 0.0)
        << "noise filter rate must be 0 before any tokens are processed";
}

TEST(TradeSignalEngineTest, test_get_noise_filter_rate_zero_when_no_noise_gate) {
    // No min_bias_threshold — nothing should be noise-filtered.
    auto cfg = make_config();
    cfg.min_bias_threshold = 0.0;
    TradeSignalEngine engine(cfg);
    engine.set_realtime_mode(false);
    engine.set_signal_callback([](const TradeSignal&) {});
    for (int i = 0; i < 5; ++i)
        engine.process_semantic_weight({0.5, 0.9, 0.2, 0.8});
    EXPECT_DOUBLE_EQ(engine.get_noise_filter_rate(), 0.0)
        << "noise filter rate must be 0 when min_bias_threshold is disabled";
}

TEST(TradeSignalEngineTest, test_get_noise_filter_rate_one_when_all_filtered) {
    // Set an impossibly high threshold so every token is noise-filtered.
    auto cfg = make_config();
    cfg.min_bias_threshold = 1e9;
    TradeSignalEngine engine(cfg);
    engine.set_realtime_mode(false);
    engine.set_signal_callback([](const TradeSignal&) {});
    for (int i = 0; i < 10; ++i)
        engine.process_semantic_weight({0.01, 0.9, 0.01, 0.01});
    EXPECT_DOUBLE_EQ(engine.get_noise_filter_rate(), 1.0)
        << "noise filter rate must be 1.0 when all tokens are below the threshold";
}

TEST(TradeSignalEngineTest, test_get_peak_bias_zero_initially) {
    TradeSignalEngine engine(make_config());
    EXPECT_DOUBLE_EQ(engine.get_peak_bias(), 0.0)
        << "peak_bias must be 0 before any tokens are processed";
}

TEST(TradeSignalEngineTest, test_get_peak_bias_non_negative_after_tokens) {
    auto cfg = make_config();
    TradeSignalEngine engine(cfg);
    engine.set_realtime_mode(false);
    engine.set_signal_callback([](const TradeSignal&) {});
    engine.process_semantic_weight({0.5, 0.9, 0.2, 0.8});
    EXPECT_GE(engine.get_peak_bias(), 0.0)
        << "peak_bias must be >= 0 after processing tokens";
}

TEST(TradeSignalEngineTest, test_get_accumulator_clamp_rate_zero_no_tokens) {
    TradeSignalEngine engine(make_config());
    EXPECT_DOUBLE_EQ(engine.get_accumulator_clamp_rate(), 0.0)
        << "accumulator_clamp_rate must be 0 before any tokens processed";
}

TEST(TradeSignalEngineTest, test_get_accumulator_clamp_rate_zero_no_cap) {
    auto cfg = make_config();
    cfg.max_accumulated_bias = 0.0;  // disabled
    TradeSignalEngine engine(cfg);
    engine.set_realtime_mode(false);
    engine.set_signal_callback([](const TradeSignal&) {});
    for (int i = 0; i < 5; ++i)
        engine.process_semantic_weight({0.5, 0.9, 0.2, 0.8});
    // No cap applied — clamp rate should be 0.
    EXPECT_DOUBLE_EQ(engine.get_accumulator_clamp_rate(), 0.0)
        << "accumulator_clamp_rate must be 0 when max_accumulated_bias is disabled";
}

TEST(TradeSignalEngineTest, test_get_bias_direction_positive) {
    TradeSignalEngine engine(make_config());
    engine.set_realtime_mode(false);
    engine.set_signal_callback([](const TradeSignal&) {});
    SemanticWeight w{0.5, 0.9, 0.1, 0.8};  // positive directional_bias
    engine.process_semantic_weight(w);
    EXPECT_EQ(engine.get_bias_direction(), 1)
        << "positive accumulated bias must yield direction +1";
}

TEST(TradeSignalEngineTest, test_get_bias_direction_zero_before_processing) {
    TradeSignalEngine engine(make_config());
    EXPECT_EQ(engine.get_bias_direction(), 0)
        << "bias direction must be 0 before any processing";
}

TEST(TradeSignalEngineTest, test_has_pending_bias_false_initially) {
    TradeSignalEngine engine(make_config());
    EXPECT_FALSE(engine.has_pending_bias())
        << "has_pending_bias must be false when no tokens processed";
}

TEST(TradeSignalEngineTest, test_has_pending_bias_true_after_processing) {
    TradeSignalEngine engine(make_config());
    engine.set_realtime_mode(false);
    engine.set_signal_callback([](const TradeSignal&) {});
    SemanticWeight w{0.5, 0.9, 0.2, 0.8};
    engine.process_semantic_weight(w);
    EXPECT_TRUE(engine.has_pending_bias())
        << "has_pending_bias must be true after accumulating non-zero bias";
}

TEST(TradeSignalEngineTest, test_get_avg_signal_quality_zero_before_signals) {
    TradeSignalEngine engine(make_config());
    EXPECT_DOUBLE_EQ(engine.get_avg_signal_quality(), 0.0)
        << "avg_signal_quality must be 0 before any signals are emitted";
}

TEST(TradeSignalEngineTest, test_get_avg_signal_quality_in_range_after_processing) {
    TradeSignalEngine engine(make_config());
    engine.set_realtime_mode(false);
    engine.set_signal_callback([](const TradeSignal&) {});
    SemanticWeight w{0.8, 0.9, 0.3, 0.7};
    engine.process_semantic_weight(w);
    double q = engine.get_avg_signal_quality();
    EXPECT_GE(q, 0.0);
    EXPECT_LE(q, 1.0);
}

TEST(TradeSignalEngineTest, test_get_signals_generated_zero_before_processing) {
    TradeSignalEngine engine(make_config());
    EXPECT_EQ(engine.get_signals_generated(), uint64_t{0});
}

TEST(TradeSignalEngineTest, test_get_signals_generated_increments_after_emit) {
    TradeSignalEngine engine(make_config());
    engine.set_realtime_mode(false);
    engine.set_signal_callback([](const TradeSignal&) {});
    SemanticWeight w{0.9, 0.95, 0.2, 0.8};
    engine.process_semantic_weight(w);
    EXPECT_GE(engine.get_signals_generated(), uint64_t{0});
}

TEST(TradeSignalEngineTest, test_get_signals_suppressed_zero_before_processing) {
    TradeSignalEngine engine(make_config());
    EXPECT_EQ(engine.get_signals_suppressed(), uint64_t{0});
}

TEST(TradeSignalEngineTest, test_get_tokens_processed_zero_before_processing) {
    TradeSignalEngine engine(make_config());
    EXPECT_EQ(engine.get_tokens_processed(), uint64_t{0});
}

TEST(TradeSignalEngineTest, test_get_tokens_processed_increments_per_call) {
    TradeSignalEngine engine(make_config());
    engine.set_realtime_mode(false);
    engine.set_signal_callback([](const TradeSignal&) {});
    SemanticWeight w{0.5, 0.8, 0.1, 0.4};
    engine.process_semantic_weight(w);
    engine.process_semantic_weight(w);
    EXPECT_EQ(engine.get_tokens_processed(), uint64_t{2});
}

TEST(TradeSignalEngineTest, test_generated_plus_suppressed_equals_processed_or_less) {
    TradeSignalEngine engine(make_config());
    engine.set_realtime_mode(false);
    engine.set_signal_callback([](const TradeSignal&) {});
    SemanticWeight w{0.7, 0.9, 0.2, 0.6};
    for (int i = 0; i < 5; ++i)
        engine.process_semantic_weight(w);
    // Each process call counts as a token, signals generated + suppressed <= tokens_processed
    EXPECT_LE(engine.get_signals_generated() + engine.get_signals_suppressed(),
              engine.get_tokens_processed());
}

TEST(TradeSignalEngineTest, test_get_time_since_last_signal_us_non_negative) {
    TradeSignalEngine engine(make_config());
    EXPECT_GE(engine.get_time_since_last_signal_us(), 0.0);
}

TEST(TradeSignalEngineTest, test_get_time_since_last_signal_us_grows_over_time) {
    TradeSignalEngine engine(make_config());
    engine.set_realtime_mode(false);
    engine.set_signal_callback([](const TradeSignal&) {});
    SemanticWeight w{0.9, 0.95, 0.2, 0.8};
    engine.process_semantic_weight(w);
    double t0 = engine.get_time_since_last_signal_us();
    std::this_thread::sleep_for(std::chrono::milliseconds(2));
    double t1 = engine.get_time_since_last_signal_us();
    EXPECT_GE(t1, t0);
}

TEST(TradeSignalEngineTest, test_is_in_cooldown_false_initially) {
    TradeSignalEngine engine(make_config());
    // No signal emitted yet — last_signal_time_ is epoch, so time elapsed is very large.
    EXPECT_FALSE(engine.is_in_cooldown());
}

TEST(TradeSignalEngineTest, test_is_in_cooldown_true_immediately_after_signal) {
    TradeSignalEngine::Config cfg = make_config();
    cfg.signal_cooldown = std::chrono::microseconds{100000}; // 100ms cooldown
    TradeSignalEngine engine(cfg);
    engine.set_realtime_mode(false);
    engine.set_signal_callback([](const TradeSignal&) {});
    SemanticWeight w{0.9, 0.95, 0.2, 0.8};
    engine.process_semantic_weight(w);
    // Immediately after emission, should be in cooldown.
    EXPECT_TRUE(engine.is_in_cooldown());
}

TEST(TradeSignalEngineTest, test_get_signal_efficiency_zero_before_any_tokens) {
    TradeSignalEngine engine(make_config());
    EXPECT_DOUBLE_EQ(engine.get_signal_efficiency(), 0.0);
}

TEST(TradeSignalEngineTest, test_get_signal_efficiency_in_range) {
    TradeSignalEngine engine(make_config());
    engine.set_realtime_mode(false);
    engine.set_signal_callback([](const TradeSignal&) {});
    SemanticWeight w{0.8, 0.9, 0.2, 0.7};
    engine.process_semantic_weight(w);
    double eff = engine.get_signal_efficiency();
    EXPECT_GE(eff, 0.0);
    EXPECT_LE(eff, 1.0);
}

TEST(TradeSignalEngineTest, test_get_signals_aged_out_zero_initially) {
    TradeSignalEngine engine(make_config());
    EXPECT_EQ(engine.get_signals_aged_out(), uint64_t{0});
}

TEST(TradeSignalEngineTest, test_get_aged_out_rate_zero_before_processing) {
    TradeSignalEngine engine(make_config());
    EXPECT_DOUBLE_EQ(engine.get_aged_out_rate(), 0.0);
}

TEST(TradeSignalEngineTest, test_get_aged_out_rate_in_range_after_processing) {
    TradeSignalEngine engine(make_config());
    engine.set_realtime_mode(false);
    engine.set_signal_callback([](const TradeSignal&) {});
    SemanticWeight w{0.6, 0.8, 0.2, 0.5};
    for (int i = 0; i < 5; ++i)
        engine.process_semantic_weight(w);
    double rate = engine.get_aged_out_rate();
    EXPECT_GE(rate, 0.0);
    EXPECT_LE(rate, 1.0);
}

TEST(TradeSignalEngineTest, test_get_signals_aged_out_resets_with_reset) {
    TradeSignalEngine engine(make_config());
    engine.set_realtime_mode(false);
    engine.set_signal_callback([](const TradeSignal&) {});
    SemanticWeight w{0.5, 0.8, 0.1, 0.4};
    for (int i = 0; i < 3; ++i)
        engine.process_semantic_weight(w);
    engine.reset();
    EXPECT_EQ(engine.get_signals_aged_out(), uint64_t{0});
}

TEST(TradeSignalEngineTest, test_get_avg_bias_per_token_zero_before_processing) {
    TradeSignalEngine engine(make_config());
    EXPECT_DOUBLE_EQ(engine.get_avg_bias_per_token(), 0.0);
}

TEST(TradeSignalEngineTest, test_get_avg_bias_per_token_non_negative_after_processing) {
    TradeSignalEngine engine(make_config());
    engine.set_realtime_mode(false);
    engine.set_signal_callback([](const TradeSignal&) {});
    SemanticWeight w{0.8, 0.9, 0.2, 0.7};
    engine.process_semantic_weight(w);
    EXPECT_GE(engine.get_avg_bias_per_token(), 0.0);
}

TEST(TradeSignalEngineTest, test_format_stats_no_data_before_processing) {
    TradeSignalEngine engine(make_config());
    std::string s = engine.format_stats();
    EXPECT_NE(s.find("tokens=0"), std::string::npos)
        << "format_stats must indicate zero tokens before any processing";
}

TEST(TradeSignalEngineTest, test_format_stats_contains_key_fields_after_processing) {
    TradeSignalEngine engine(make_config());
    engine.set_realtime_mode(false);
    engine.set_signal_callback([](const TradeSignal&) {});
    SemanticWeight w{0.8, 0.9, 0.2, 0.7};
    engine.process_semantic_weight(w);
    std::string s = engine.format_stats();
    EXPECT_NE(s.find("tokens="),     std::string::npos);
    EXPECT_NE(s.find("generated="),  std::string::npos);
    EXPECT_NE(s.find("suppressed="), std::string::npos);
    EXPECT_NE(s.find("efficiency="), std::string::npos);
}

TEST(TradeSignalEngineTest, test_format_stats_token_count_matches) {
    TradeSignalEngine engine(make_config());
    engine.set_realtime_mode(false);
    engine.set_signal_callback([](const TradeSignal&) {});
    SemanticWeight w{0.6, 0.8, 0.2, 0.5};
    engine.process_semantic_weight(w);
    engine.process_semantic_weight(w);
    std::string s = engine.format_stats();
    EXPECT_NE(s.find("tokens=2"), std::string::npos)
        << "format_stats token count must match processed count";
}

TEST(TradeSignalEngineTest, test_get_signal_velocity_zero_before_signals) {
    TradeSignalEngine engine(make_config());
    EXPECT_DOUBLE_EQ(engine.get_signal_velocity(), 0.0);
}

TEST(TradeSignalEngineTest, test_get_signal_velocity_positive_after_signal) {
    TradeSignalEngine engine(make_config());
    engine.set_realtime_mode(false);
    engine.set_signal_callback([](const TradeSignal&) {});
    SemanticWeight w{0.9, 0.95, 0.2, 0.8};
    engine.process_semantic_weight(w);
    EXPECT_GE(engine.get_signal_velocity(), 0.0);
}

// ---------------------------------------------------------------------------
// format_stats — noise_filtered, peak_bias, avg_quality fields added
// ---------------------------------------------------------------------------

TEST(TradeSignalEngineTest, test_format_stats_contains_noise_filtered_field) {
    TradeSignalEngine::Config cfg = make_config(1.0, 1.0, 0.999, 0);
    cfg.min_bias_threshold = 100.0;  // suppress all to force noise_filtered > 0
    TradeSignalEngine engine(cfg);
    engine.set_backtest_mode(true);
    engine.set_signal_callback([](const TradeSignal&) {});
    for (int i = 0; i < 3; ++i)
        engine.process_semantic_weight({0.1, 0.5, 0.1, 0.1});
    std::string s = engine.format_stats();
    EXPECT_NE(s.find("noise_filtered="), std::string::npos)
        << "format_stats must include noise_filtered field";
    EXPECT_NE(s.find("noise_filtered=3"), std::string::npos)
        << "format_stats noise_filtered must match actual count";
}

TEST(TradeSignalEngineTest, test_format_stats_contains_peak_bias_field) {
    TradeSignalEngine engine(make_config());
    engine.set_backtest_mode(true);
    engine.set_signal_callback([](const TradeSignal&) {});
    engine.process_semantic_weight({0.8, 0.9, 0.2, 0.7});
    std::string s = engine.format_stats();
    EXPECT_NE(s.find("peak_bias="), std::string::npos)
        << "format_stats must include peak_bias field";
}

TEST(TradeSignalEngineTest, test_format_stats_contains_avg_quality_field) {
    TradeSignalEngine engine(make_config());
    engine.set_backtest_mode(true);
    engine.set_signal_callback([](const TradeSignal&) {});
    engine.process_semantic_weight({0.8, 0.9, 0.2, 0.7});
    std::string s = engine.format_stats();
    EXPECT_NE(s.find("avg_quality="), std::string::npos)
        << "format_stats must include avg_quality field";
}

// ---------------------------------------------------------------------------
// Signal quality histogram
// ---------------------------------------------------------------------------

TEST(TradeSignalEngineTest, test_quality_histogram_initial_all_zero) {
    TradeSignalEngine engine(make_config());
    auto hist = engine.get_quality_histogram();
    ASSERT_EQ(hist.size(), 5u);
    for (const auto& b : hist) {
        EXPECT_EQ(b.count, 0u) << "All histogram buckets must be zero at construction";
    }
}

TEST(TradeSignalEngineTest, test_quality_histogram_five_buckets_correct_bounds) {
    TradeSignalEngine engine(make_config());
    auto hist = engine.get_quality_histogram();
    ASSERT_EQ(hist.size(), 5u);
    EXPECT_DOUBLE_EQ(hist[0].upper_bound, 0.2);
    EXPECT_DOUBLE_EQ(hist[1].upper_bound, 0.4);
    EXPECT_DOUBLE_EQ(hist[2].upper_bound, 0.6);
    EXPECT_DOUBLE_EQ(hist[3].upper_bound, 0.8);
    EXPECT_DOUBLE_EQ(hist[4].upper_bound, 1.0);
}

TEST(TradeSignalEngineTest, test_quality_histogram_low_quality_lands_in_first_bucket) {
    TradeSignalEngine engine(make_config());
    engine.set_backtest_mode(true);
    engine.set_signal_callback([](const TradeSignal&) {});
    // Low confidence + small bias → quality well below 0.2
    engine.process_semantic_weight({0.05, 0.1, 0.0, 0.05});
    auto hist = engine.get_quality_histogram();
    EXPECT_GT(hist[0].count, 0u) << "Low-quality signal must land in bucket [0, 0.2)";
}

TEST(TradeSignalEngineTest, test_quality_histogram_high_quality_lands_in_last_bucket) {
    TradeSignalEngine engine(make_config());
    engine.set_backtest_mode(true);
    engine.set_signal_callback([](const TradeSignal&) {});
    // High confidence + large bias → quality above 0.8
    engine.process_semantic_weight({0.95, 0.99, 0.9, 0.95});
    auto hist = engine.get_quality_histogram();
    EXPECT_GT(hist[4].count, 0u) << "High-quality signal must land in bucket [0.8, 1.0]";
}

TEST(TradeSignalEngineTest, test_quality_histogram_total_matches_signals_generated) {
    TradeSignalEngine engine(make_config());
    engine.set_backtest_mode(true);
    engine.set_signal_callback([](const TradeSignal&) {});
    const int N = 5;
    for (int i = 0; i < N; ++i) {
        engine.process_semantic_weight({0.5, 0.8, 0.3, 0.5});
    }
    auto hist = engine.get_quality_histogram();
    uint64_t total = 0;
    for (const auto& b : hist) total += b.count;
    EXPECT_EQ(total, static_cast<uint64_t>(engine.get_signals_generated()))
        << "Sum of histogram bucket counts must equal signals_generated";
}

// ---------------------------------------------------------------------------
// TradeSignal::to_json()
// ---------------------------------------------------------------------------

TEST(TradeSignalEngineTest, test_trade_signal_to_json_contains_all_fields) {
    TradeSignal sig{};
    sig.timestamp_ns          = 1700000000000000000ULL;
    sig.delta_bias_shift      = 0.5;
    sig.volatility_adjustment = 0.3;
    sig.spread_modifier       = -0.1;
    sig.confidence            = 0.85;
    sig.latency_us            = 7.25;
    sig.strategy_toggle       = 1;
    sig.strategy_weight       = 0.9;
    sig.signal_quality        = 0.75;

    std::string json = sig.to_json();
    EXPECT_FALSE(json.empty());
    EXPECT_NE(json.find("\"timestamp_ns\""),          std::string::npos);
    EXPECT_NE(json.find("\"delta_bias_shift\""),      std::string::npos);
    EXPECT_NE(json.find("\"volatility_adjustment\""), std::string::npos);
    EXPECT_NE(json.find("\"spread_modifier\""),       std::string::npos);
    EXPECT_NE(json.find("\"confidence\""),            std::string::npos);
    EXPECT_NE(json.find("\"latency_us\""),            std::string::npos);
    EXPECT_NE(json.find("\"strategy_toggle\""),       std::string::npos);
    EXPECT_NE(json.find("\"strategy_weight\""),       std::string::npos);
    EXPECT_NE(json.find("\"signal_quality\""),        std::string::npos);
}

TEST(TradeSignalEngineTest, test_trade_signal_to_json_is_valid_object) {
    TradeSignal sig{};
    std::string json = sig.to_json();
    ASSERT_FALSE(json.empty());
    EXPECT_EQ(json.front(), '{') << "JSON must start with {";
    EXPECT_EQ(json.back(),  '}') << "JSON must end with }";
}

TEST(TradeSignalEngineTest, test_trade_signal_to_json_values_reflect_fields) {
    TradeSignal sig{};
    sig.delta_bias_shift = 1.5;
    sig.confidence       = 0.75;
    sig.strategy_toggle  = -1;

    std::string json = sig.to_json();
    EXPECT_NE(json.find("1.500000"), std::string::npos) << "delta_bias_shift not serialised";
    EXPECT_NE(json.find("0.750000"), std::string::npos) << "confidence not serialised";
    EXPECT_NE(json.find("-1"),       std::string::npos) << "strategy_toggle=-1 not serialised";
}

// --- format_stats quality histogram ---

TEST(TradeSignalEngineTest, test_format_stats_contains_quality_hist_field) {
    TradeSignalEngine engine(make_config());
    engine.set_backtest_mode(true);
    engine.set_signal_callback([](const TradeSignal&) {});
    engine.process_semantic_weight({0.8, 0.9, 0.5, 0.7});
    std::string stats = engine.format_stats();
    EXPECT_NE(stats.find("quality_hist="), std::string::npos)
        << "format_stats must include quality_hist field after processing a token";
}

TEST(TradeSignalEngineTest, test_format_stats_quality_hist_five_buckets) {
    TradeSignalEngine engine(make_config());
    engine.set_backtest_mode(true);
    engine.set_signal_callback([](const TradeSignal&) {});
    engine.process_semantic_weight({0.8, 0.9, 0.5, 0.7});
    std::string stats = engine.format_stats();
    // quality_hist=[a,b,c,d,e] — expect exactly 4 commas inside the brackets
    auto start = stats.find("quality_hist=[");
    ASSERT_NE(start, std::string::npos);
    auto end = stats.find("]", start);
    ASSERT_NE(end, std::string::npos);
    std::string hist = stats.substr(start, end - start + 1);
    int commas = static_cast<int>(std::count(hist.begin(), hist.end(), ','));
    EXPECT_EQ(commas, 4) << "quality_hist must have exactly 5 comma-separated bucket counts";
}

TEST(TradeSignalEngineTest, test_format_stats_no_data_returns_early) {
    TradeSignalEngine engine(make_config());
    std::string stats = engine.format_stats();
    EXPECT_EQ(stats, "tokens=0 (no data)")
        << "format_stats must return early message when no tokens processed";
}

TEST(TradeSignalEngineTest, test_format_stats_contains_quality_ema_field) {
    TradeSignalEngine engine(make_config());
    engine.set_backtest_mode(true);
    engine.set_signal_callback([](const TradeSignal&) {});
    engine.process_semantic_weight({0.8, 0.9, 0.5, 0.7});

    std::string stats = engine.format_stats();
    EXPECT_NE(stats.find("quality_ema="), std::string::npos)
        << "format_stats must include quality_ema field after a signal is emitted";
}

TEST(TradeSignalEngineTest, test_format_stats_quality_ema_sentinel_before_signals) {
    TradeSignalEngine engine(make_config());
    // Process a token that is below the noise gate so no signal is emitted,
    // meaning quality_ema stays at the -1 sentinel.
    engine.set_backtest_mode(true);
    engine.set_min_bias_threshold(999.0); // impossibly high threshold
    engine.set_signal_callback([](const TradeSignal&) {});
    engine.process_semantic_weight({0.01, 0.01, 0.01, 0.01});

    std::string stats = engine.format_stats();
    EXPECT_NE(stats.find("quality_ema=-1"), std::string::npos)
        << "format_stats must show quality_ema=-1 when no signal has been emitted";
}

// --- signal quality EMA ---

TEST(TradeSignalEngineTest, test_signal_quality_ema_initial_is_minus_one) {
    TradeSignalEngine engine(make_config());
    EXPECT_DOUBLE_EQ(engine.get_signal_quality_ema(), -1.0)
        << "EMA must be -1.0 (sentinel) before any signal is emitted";
}

TEST(TradeSignalEngineTest, test_signal_quality_ema_seeds_on_first_signal) {
    TradeSignalEngine engine(make_config());
    engine.set_backtest_mode(true);
    engine.set_signal_callback([](const TradeSignal&) {});
    engine.process_semantic_weight({0.8, 0.9, 0.5, 0.7});
    double ema = engine.get_signal_quality_ema();
    EXPECT_GE(ema, 0.0) << "EMA must be seeded with first signal quality";
    EXPECT_LE(ema, 1.0);
}

TEST(TradeSignalEngineTest, test_signal_quality_ema_moves_toward_new_values) {
    TradeSignalEngine engine(make_config());
    engine.set_backtest_mode(true);
    engine.set_signal_callback([](const TradeSignal&) {});
    // Emit a high-quality signal first
    engine.process_semantic_weight({0.9, 0.95, 0.1, 0.9});
    double ema_after_first = engine.get_signal_quality_ema();
    // Emit a low-quality signal; EMA should move toward lower value
    engine.process_semantic_weight({0.01, 0.5, 0.5, 0.01});
    double ema_after_second = engine.get_signal_quality_ema();
    EXPECT_LT(ema_after_second, ema_after_first)
        << "EMA must decrease when a lower-quality signal is emitted";
}

TEST(TradeSignalEngineTest, test_signal_quality_ema_reset_to_sentinel) {
    TradeSignalEngine engine(make_config());
    engine.set_backtest_mode(true);
    engine.set_signal_callback([](const TradeSignal&) {});
    engine.process_semantic_weight({0.8, 0.9, 0.5, 0.7});
    EXPECT_GE(engine.get_signal_quality_ema(), 0.0);

    engine.reset();
    EXPECT_DOUBLE_EQ(engine.get_signal_quality_ema(), -1.0)
        << "EMA must reset to -1.0 sentinel after reset()";
}

// ---------------------------------------------------------------------------
// TradeSignalEngine::to_stats_json() tests
// ---------------------------------------------------------------------------

TEST(TradeSignalEngineTest, test_to_stats_json_returns_valid_json_with_required_fields) {
    TradeSignalEngine engine(make_config());
    engine.set_backtest_mode(true);
    engine.set_signal_callback([](const TradeSignal&) {});
    engine.process_semantic_weight({0.8, 0.9, 0.5, 0.7});

    std::string json = engine.to_stats_json();
    ASSERT_FALSE(json.empty());
    EXPECT_EQ(json.front(), '{');
    EXPECT_EQ(json.back(),  '}');
    EXPECT_NE(json.find("signals_generated"),  std::string::npos);
    EXPECT_NE(json.find("tokens_processed"),   std::string::npos);
    EXPECT_NE(json.find("avg_signal_quality"), std::string::npos);
    EXPECT_NE(json.find("peak_bias"),          std::string::npos);
    EXPECT_NE(json.find("quality_hist"),       std::string::npos);
    EXPECT_NE(json.find("signal_quality_ema"), std::string::npos);
}

TEST(TradeSignalEngineTest, test_to_stats_json_tokens_processed_matches_call_count) {
    TradeSignalEngine engine(make_config());
    engine.set_backtest_mode(true);
    engine.set_signal_callback([](const TradeSignal&) {});
    engine.process_semantic_weight({0.5, 0.5, 0.5, 0.5});
    engine.process_semantic_weight({0.6, 0.6, 0.6, 0.6});
    engine.process_semantic_weight({0.7, 0.7, 0.7, 0.7});

    std::string json = engine.to_stats_json();
    EXPECT_NE(json.find("\"tokens_processed\":3"), std::string::npos);
}

// ---------------------------------------------------------------------------
// Cycle 24: SuppressionBreakdown / get_suppression_breakdown()
// ---------------------------------------------------------------------------

TEST(TradeSignalEngineTest, test_suppression_breakdown_all_zero_initially) {
    TradeSignalEngine engine(make_config());
    auto bd = engine.get_suppression_breakdown();
    EXPECT_EQ(bd.noise_filtered,      0u);
    EXPECT_EQ(bd.aged_out,            0u);
    EXPECT_EQ(bd.cooldown_suppressed, 0u);
    EXPECT_EQ(bd.total,               0u);
}

TEST(TradeSignalEngineTest, test_suppression_breakdown_noise_increments_when_below_threshold) {
    TradeSignalEngine::Config cfg = make_config();
    cfg.min_bias_threshold = 0.5;
    TradeSignalEngine engine(cfg);
    engine.set_backtest_mode(true);
    engine.set_signal_callback([](const TradeSignal&) {});

    // Low-bias token should be filtered by noise gate.
    engine.process_semantic_weight({0.1, 0.5, 0.1, 0.05});
    auto bd = engine.get_suppression_breakdown();
    EXPECT_EQ(bd.noise_filtered, 1u) << "noise gate must count the suppressed token";
    EXPECT_GE(bd.total, 1u);
}

TEST(TradeSignalEngineTest, test_suppression_breakdown_cooldown_increments_in_realtime_mode) {
    TradeSignalEngine::Config cfg = make_config();
    cfg.signal_cooldown = std::chrono::microseconds{1'000'000}; // 1 second
    TradeSignalEngine engine(cfg);
    engine.set_realtime_mode(true);
    engine.set_signal_callback([](const TradeSignal&) {});

    // First token emits (cooldown not yet active), subsequent tokens are in cooldown.
    engine.process_semantic_weight({0.9, 0.9, 0.5, 0.8});
    engine.process_semantic_weight({0.9, 0.9, 0.5, 0.8});
    engine.process_semantic_weight({0.9, 0.9, 0.5, 0.8});

    auto bd = engine.get_suppression_breakdown();
    EXPECT_GE(bd.cooldown_suppressed, 1u)
        << "at least two tokens must be suppressed by the 1-second cooldown";
}

TEST(TradeSignalEngineTest, test_suppression_breakdown_resets_on_reset) {
    TradeSignalEngine::Config cfg = make_config();
    cfg.min_bias_threshold = 0.5;
    TradeSignalEngine engine(cfg);
    engine.set_backtest_mode(true);
    engine.set_signal_callback([](const TradeSignal&) {});

    engine.process_semantic_weight({0.1, 0.5, 0.1, 0.05}); // noise-filtered
    EXPECT_GE(engine.get_suppression_breakdown().noise_filtered, 1u);

    engine.reset();
    auto bd = engine.get_suppression_breakdown();
    EXPECT_EQ(bd.noise_filtered,      0u);
    EXPECT_EQ(bd.cooldown_suppressed, 0u);
    EXPECT_EQ(bd.aged_out,            0u);
    EXPECT_EQ(bd.total,               0u);
}

// ---------------------------------------------------------------------------
// Cycle 28: zero-confidence edge case
// ---------------------------------------------------------------------------

TEST(TradeSignalEngineTest, test_zero_confidence_weight_is_noise_filtered) {
    // A SemanticWeight with all-zero fields should never produce a signal —
    // accumulated bias stays at 0 and the min_bias_threshold noise gate fires.
    TradeSignalEngine::Config cfg = make_config();
    cfg.min_bias_threshold = 0.01;  // any positive threshold
    TradeSignalEngine engine(cfg);
    engine.set_backtest_mode(true);

    std::atomic<int> emitted{0};
    engine.set_signal_callback([&emitted](const TradeSignal&) { ++emitted; });

    engine.process_semantic_weight({0.0, 0.0, 0.0, 0.0});  // all-zero weight
    EXPECT_EQ(emitted.load(), 0)
        << "all-zero SemanticWeight must not produce a signal";
}

TEST(TradeSignalEngineTest, test_signals_suppressed_increments_on_noise_filter) {
    // Verify the signals_suppressed counter increments when a token is noise-filtered.
    TradeSignalEngine::Config cfg = make_config();
    cfg.min_bias_threshold = 0.9;  // very high threshold — zero-weight token is filtered
    TradeSignalEngine engine(cfg);
    engine.set_backtest_mode(true);
    engine.set_signal_callback([](const TradeSignal&) {});

    const uint64_t before = engine.get_stats().signals_suppressed.load();
    engine.process_semantic_weight({0.0, 0.0, 0.0, 0.0});
    EXPECT_GT(engine.get_stats().signals_suppressed.load(), before)
        << "signals_suppressed must increment after a noise-filtered token";
}

// ---------------------------------------------------------------------------
// Tests for min_vol_threshold (volatility dead-band filter)
// ---------------------------------------------------------------------------

TEST(TradeSignalEngineTest, test_min_vol_threshold_suppresses_low_vol_signals) {
    // A high min_vol_threshold should suppress signals where accumulated_vol
    // is below the threshold, even if bias is above its threshold.
    TradeSignalEngine::Config cfg = make_config();
    cfg.min_bias_threshold = 0.0;    // bias gate disabled
    cfg.min_vol_threshold  = 100.0;  // extremely high — nothing will pass
    TradeSignalEngine engine(cfg);
    engine.set_backtest_mode(true);

    std::atomic<int> emitted{0};
    engine.set_signal_callback([&emitted](const TradeSignal&) { ++emitted; });

    // Feed several tokens with non-zero bias and some vol — but vol << 100.0.
    for (int i = 0; i < 5; ++i)
        engine.process_semantic_weight({0.5, 0.3, 0.1, 0.8});

    EXPECT_EQ(emitted.load(), 0)
        << "min_vol_threshold=100 must suppress all signals regardless of bias";
    EXPECT_GT(engine.get_stats().signals_suppressed.load(), 0u)
        << "signals_suppressed must be > 0 when vol threshold suppresses tokens";
}

TEST(TradeSignalEngineTest, test_min_vol_threshold_zero_is_disabled) {
    // With min_vol_threshold=0 the vol gate is disabled; signals with bias
    // and vol should pass through.
    TradeSignalEngine::Config cfg = make_config();
    cfg.min_bias_threshold = 0.0;
    cfg.min_vol_threshold  = 0.0;
    TradeSignalEngine engine(cfg);
    engine.set_backtest_mode(true);

    std::atomic<int> emitted{0};
    engine.set_signal_callback([&emitted](const TradeSignal&) { ++emitted; });

    engine.process_semantic_weight({0.5, 0.3, 0.1, 0.8});
    EXPECT_GT(emitted.load(), 0)
        << "min_vol_threshold=0 must not suppress any signals";
}

TEST(TradeSignalEngineTest, test_set_min_vol_threshold_setter) {
    TradeSignalEngine::Config cfg = make_config();
    TradeSignalEngine engine(cfg);
    engine.set_min_vol_threshold(5.0);
    EXPECT_DOUBLE_EQ(engine.get_config().min_vol_threshold, 5.0);
}

TEST(TradeSignalEngineTest, test_set_min_vol_threshold_clamps_negative) {
    TradeSignalEngine::Config cfg = make_config();
    TradeSignalEngine engine(cfg);
    engine.set_min_vol_threshold(-3.0);  // must be clamped to 0
    EXPECT_DOUBLE_EQ(engine.get_config().min_vol_threshold, 0.0);
}

// ---------------------------------------------------------------------------
// Cycle 46: bias_reversals counter — momentum zero-crossing detection.
// ---------------------------------------------------------------------------

TEST(TradeSignalEngineTest, test_bias_reversals_zero_on_same_direction) {
    TradeSignalEngine engine(make_config());
    engine.set_backtest_mode(true);
    engine.set_signal_callback([](const TradeSignal&) {});

    for (int i = 0; i < 5; ++i)
        engine.process_semantic_weight({0.5, 0.8, 0.3, 1.0});

    EXPECT_EQ(engine.get_stats().bias_reversals.load(), 0u)
        << "bias_reversals must be 0 when all tokens share the same direction";
}

TEST(TradeSignalEngineTest, test_bias_reversals_increments_on_direction_change) {
    TradeSignalEngine engine(make_config(2.0, 1.0, 0.1, 0));
    engine.set_backtest_mode(true);
    engine.set_signal_callback([](const TradeSignal&) {});

    for (int i = 0; i < 10; ++i)
        engine.process_semantic_weight({0.5, 0.9, 0.3, 1.0});

    uint64_t reversals_before = engine.get_stats().bias_reversals.load();

    for (int i = 0; i < 10; ++i)
        engine.process_semantic_weight({-0.5, 0.9, 0.3, -1.0});

    EXPECT_GT(engine.get_stats().bias_reversals.load(), reversals_before)
        << "bias_reversals must increment when accumulated_bias changes sign";
}

TEST(TradeSignalEngineTest, test_bias_reversals_reset_on_reset) {
    TradeSignalEngine engine(make_config(2.0, 1.0, 0.1, 0));
    engine.set_backtest_mode(true);
    engine.set_signal_callback([](const TradeSignal&) {});

    for (int i = 0; i < 10; ++i)
        engine.process_semantic_weight({0.5, 0.9, 0.3, 1.0});
    for (int i = 0; i < 10; ++i)
        engine.process_semantic_weight({-0.5, 0.9, 0.3, -1.0});

    ASSERT_GT(engine.get_stats().bias_reversals.load(), 0u);
    engine.reset();

    EXPECT_EQ(engine.get_stats().bias_reversals.load(), 0u)
        << "bias_reversals must be zero after reset()";

    // After reset, no reversal should fire on a fresh bullish run.
    for (int i = 0; i < 5; ++i)
        engine.process_semantic_weight({0.5, 0.9, 0.3, 1.0});
    EXPECT_EQ(engine.get_stats().bias_reversals.load(), 0u)
        << "No reversal should be detected immediately after reset";
}

TEST(TradeSignalEngineTest, test_bias_reversals_format_stats_includes_field) {
    TradeSignalEngine engine(make_config());
    engine.set_backtest_mode(true);
    engine.set_signal_callback([](const TradeSignal&) {});
    engine.process_semantic_weight({0.5, 0.8, 0.3, 1.0});

    std::string stats = engine.format_stats();
    EXPECT_NE(stats.find("bias_reversals="), std::string::npos)
        << "format_stats() must include the bias_reversals field";
}

// ---------------------------------------------------------------------------
// Time-based bias decay tests (time_decay_half_life_ms feature)
// ---------------------------------------------------------------------------
TEST(TradeSignalEngineTest, test_time_decay_disabled_zero_half_life) {
    TradeSignalEngine::Config cfg = make_config();
    cfg.signal_decay_rate       = 1.0;   // no per-token decay — isolate time decay
    cfg.time_decay_half_life_ms = 0.0;   // disabled
    cfg.min_bias_threshold      = 0.0;
    cfg.max_accumulated_bias    = 0.0;
    TradeSignalEngine engine(cfg);
    engine.set_backtest_mode(true);
    engine.set_signal_callback([](const TradeSignal&) {});

    engine.process_semantic_weight({0.5, 1.0, 0.5, 0.0});
    double bias_initial = engine.get_accumulated_bias();
    ASSERT_GT(bias_initial, 0.0);

    // Pause 60 ms — with time decay disabled, bias must survive unchanged.
    std::this_thread::sleep_for(std::chrono::milliseconds{60});
    engine.process_semantic_weight({0.0, 0.0, 0.0, 0.0});
    double bias_after_wait = engine.get_accumulated_bias();

    // signal_decay_rate=1.0 means no per-token decay; zero contribution adds 0.
    // time_decay_half_life_ms=0 means no time decay.
    EXPECT_NEAR(bias_after_wait, bias_initial, 1e-6)
        << "With time decay disabled, bias must not change between zero-contribution tokens";
}

TEST(TradeSignalEngineTest, test_time_decay_reduces_bias_over_elapsed_time) {
    TradeSignalEngine::Config cfg = make_config();
    cfg.signal_decay_rate       = 1.0;   // no per-token decay — isolate time decay
    cfg.time_decay_half_life_ms = 20.0;  // 20 ms half-life
    cfg.min_bias_threshold      = 0.0;
    cfg.max_accumulated_bias    = 0.0;
    TradeSignalEngine engine(cfg);
    engine.set_backtest_mode(true);
    engine.set_signal_callback([](const TradeSignal&) {});

    engine.process_semantic_weight({0.8, 1.0, 0.0, 0.0});
    double bias_initial = engine.get_accumulated_bias();
    ASSERT_GT(bias_initial, 0.0);

    // Sleep 60 ms (3× the 20 ms half-life); after 3 half-lives bias < initial/8.
    // Use a loose check (< initial/2) to tolerate OS sleep imprecision.
    std::this_thread::sleep_for(std::chrono::milliseconds{60});
    engine.process_semantic_weight({0.0, 0.0, 0.0, 0.0});
    double bias_decayed = engine.get_accumulated_bias();

    EXPECT_LT(bias_decayed, bias_initial / 2.0)
        << "After ≥1 half-life elapsed, bias must have decayed by at least 50%;"
        << " initial=" << bias_initial << " decayed=" << bias_decayed;
}


} // namespace
} // namespace llmquant
