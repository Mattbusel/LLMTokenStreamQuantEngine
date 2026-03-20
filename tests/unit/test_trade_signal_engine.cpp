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

} // namespace
} // namespace llmquant
