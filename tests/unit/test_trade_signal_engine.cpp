#include "gtest/gtest.h"
#include "TradeSignalEngine.h"
#include "LLMAdapter.h"

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
    // With decay = 0.0 each token completely erases the previous accumulation.
    TradeSignalEngine engine(make_config(1.0, 1.0, 0.0 /*full decay*/, 0));
    engine.set_backtest_mode(true);

    std::vector<TradeSignal> signals;
    engine.set_signal_callback([&signals](const TradeSignal& s) { signals.push_back(s); });

    // First token: accumulate something.
    SemanticWeight large{0.9, 0.9, 0.5, 0.9};
    engine.process_semantic_weight(large);

    // Second token: neutral (zero directional bias, low confidence).
    SemanticWeight neutral{0.0, 0.5, 0.1, 0.0};
    engine.process_semantic_weight(neutral);

    ASSERT_GE(signals.size(), 2u);
    // With decay = 0.0, the second signal must have |bias| <= |first signal bias|
    // because the accumulator is wiped each tick.
    EXPECT_LE(std::abs(signals[1].delta_bias_shift),
              std::abs(signals[0].delta_bias_shift) + 1e-9);
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

} // namespace
} // namespace llmquant
