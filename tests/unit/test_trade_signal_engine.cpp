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

} // namespace
} // namespace llmquant
