#include <gtest/gtest.h>

#ifdef LLMQUANT_SIGNAL_MOMENTUM_OSC_ENABLED

#include "SignalMomentumOscillator.h"

#include <atomic>
#include <cmath>
#include <thread>
#include <vector>

using namespace llmquant;

namespace {

// ---------------------------------------------------------------------------
// Construction / default state
// ---------------------------------------------------------------------------

TEST(SignalMomentumOscillator, DefaultState) {
    SignalMomentumOscillator osc;
    EXPECT_DOUBLE_EQ(osc.macd(), 0.0);
    EXPECT_DOUBLE_EQ(osc.signal_line(), 0.0);
    EXPECT_DOUBLE_EQ(osc.histogram(), 0.0);
    EXPECT_EQ(osc.total_records(), 0u);
    EXPECT_EQ(osc.total_crosses(), 0u);
    EXPECT_FALSE(osc.is_bullish());
    EXPECT_FALSE(osc.is_bearish());
}

// ---------------------------------------------------------------------------
// Single record seeds EMAs to the first value → macd/histogram ≈ 0
// ---------------------------------------------------------------------------

TEST(SignalMomentumOscillator, FirstRecordSeeds) {
    SignalMomentumOscillator osc;
    osc.record(1.0);
    EXPECT_NEAR(osc.macd(), 0.0, 1e-9);
    EXPECT_NEAR(osc.histogram(), 0.0, 1e-9);
    EXPECT_EQ(osc.total_records(), 1u);
}

// ---------------------------------------------------------------------------
// Sustained positive bias → fast EMA converges above slow EMA → macd > 0
// ---------------------------------------------------------------------------

TEST(SignalMomentumOscillator, SustainedPositiveBias) {
    SignalMomentumOscillator osc;
    for (int i = 0; i < 200; ++i) osc.record(1.0);
    // After many records both EMAs ≈ 1.0, so macd ≈ 0.
    // But if we push a step change the fast should respond faster.
    // Here just verify macd is near 0 for constant input.
    EXPECT_NEAR(osc.macd(), 0.0, 0.01);
}

// ---------------------------------------------------------------------------
// Step up → fast EMA rises faster → macd > 0
// ---------------------------------------------------------------------------

TEST(SignalMomentumOscillator, StepUpProducesPositiveMacd) {
    SignalMomentumOscillator osc;
    // Warm up with 0.0.
    for (int i = 0; i < 100; ++i) osc.record(0.0);
    // Step to 1.0.
    for (int i = 0; i < 30; ++i) osc.record(1.0);
    EXPECT_GT(osc.macd(), 0.0);
}

// ---------------------------------------------------------------------------
// Step down → macd < 0
// ---------------------------------------------------------------------------

TEST(SignalMomentumOscillator, StepDownProducesNegativeMacd) {
    SignalMomentumOscillator osc;
    for (int i = 0; i < 100; ++i) osc.record(0.0);
    for (int i = 0; i < 30; ++i) osc.record(-1.0);
    EXPECT_LT(osc.macd(), 0.0);
}

// ---------------------------------------------------------------------------
// Histogram = macd - signal_line always
// ---------------------------------------------------------------------------

TEST(SignalMomentumOscillator, HistogramIdentity) {
    SignalMomentumOscillator osc;
    for (int i = 0; i < 50; ++i) osc.record(static_cast<double>(i % 5) * 0.1);
    EXPECT_NEAR(osc.histogram(), osc.macd() - osc.signal_line(), 1e-9);
}

// ---------------------------------------------------------------------------
// Bullish zero-cross detected when histogram goes from negative to positive
// ---------------------------------------------------------------------------

TEST(SignalMomentumOscillator, BullishCrossDetected) {
    SignalMomentumOscillator::Config cfg;
    cfg.fast_alpha   = 0.3;
    cfg.slow_alpha   = 0.1;
    cfg.signal_alpha = 0.4;
    std::atomic<int> crosses{0};
    std::atomic<bool> got_bullish{false};
    cfg.on_cross = [&](SignalMomentumOscillator::CrossDirection dir, double /*hist*/) {
        ++crosses;
        if (dir == SignalMomentumOscillator::CrossDirection::Bullish)
            got_bullish.store(true);
    };
    SignalMomentumOscillator osc(cfg);

    // Settlement phase: both EMAs converge to +1 (histogram ≈ 0).
    for (int i = 0; i < 50; ++i) osc.record(1.0);
    // Negative phase: fast drops first → histogram goes negative.
    for (int i = 0; i < 50; ++i) osc.record(-1.0);
    // Positive phase: fast rises first → histogram crosses zero upward (bullish).
    for (int i = 0; i < 50; ++i) osc.record(1.0);

    EXPECT_GT(crosses.load(), 0);
    EXPECT_TRUE(got_bullish.load());
}

// ---------------------------------------------------------------------------
// Bearish zero-cross detected when histogram goes from positive to negative
// ---------------------------------------------------------------------------

TEST(SignalMomentumOscillator, BearishCrossDetected) {
    SignalMomentumOscillator::Config cfg;
    cfg.fast_alpha   = 0.3;
    cfg.slow_alpha   = 0.1;
    cfg.signal_alpha = 0.4;
    std::atomic<bool> got_bearish{false};
    cfg.on_cross = [&](SignalMomentumOscillator::CrossDirection dir, double) {
        if (dir == SignalMomentumOscillator::CrossDirection::Bearish)
            got_bearish.store(true);
    };
    SignalMomentumOscillator osc(cfg);

    // Settlement phase: both EMAs converge to -1 (histogram ≈ 0).
    for (int i = 0; i < 50; ++i) osc.record(-1.0);
    // Positive phase: fast rises first → histogram goes positive.
    for (int i = 0; i < 50; ++i) osc.record(1.0);
    // Negative phase: fast drops first → histogram crosses zero downward (bearish).
    for (int i = 0; i < 50; ++i) osc.record(-1.0);

    EXPECT_TRUE(got_bearish.load());
}

// ---------------------------------------------------------------------------
// Divergence callback fires when |histogram| >= threshold
// ---------------------------------------------------------------------------

TEST(SignalMomentumOscillator, DivergenceCallbackFires) {
    SignalMomentumOscillator::Config cfg;
    cfg.fast_alpha          = 0.5;
    cfg.slow_alpha          = 0.05;
    cfg.divergence_threshold = 0.01;  // low threshold
    std::atomic<int> div_count{0};
    cfg.on_divergence = [&](double, double, double) { ++div_count; };
    SignalMomentumOscillator osc(cfg);

    for (int i = 0; i < 60; ++i) osc.record(static_cast<double>(i % 2) == 0 ? 1.0 : -0.5);

    EXPECT_GT(div_count.load(), 0);
}

// ---------------------------------------------------------------------------
// reset() clears all state
// ---------------------------------------------------------------------------

TEST(SignalMomentumOscillator, ResetClearsState) {
    SignalMomentumOscillator osc;
    for (int i = 0; i < 50; ++i) osc.record(static_cast<double>(i));
    osc.reset();
    EXPECT_DOUBLE_EQ(osc.macd(), 0.0);
    EXPECT_DOUBLE_EQ(osc.histogram(), 0.0);
    EXPECT_EQ(osc.total_records(), 0u);
    EXPECT_EQ(osc.total_crosses(), 0u);
}

// ---------------------------------------------------------------------------
// update_config resets state
// ---------------------------------------------------------------------------

TEST(SignalMomentumOscillator, UpdateConfigResetsState) {
    SignalMomentumOscillator osc;
    for (int i = 0; i < 30; ++i) osc.record(1.0);
    SignalMomentumOscillator::Config cfg2;
    cfg2.fast_alpha = 0.2;
    osc.update_config(cfg2);
    EXPECT_DOUBLE_EQ(osc.macd(), 0.0);
    EXPECT_EQ(osc.total_records(), 0u);
}

// ---------------------------------------------------------------------------
// to_stats_json contains expected fields
// ---------------------------------------------------------------------------

TEST(SignalMomentumOscillator, ToStatsJsonContainsFields) {
    SignalMomentumOscillator osc;
    osc.record(0.5);
    std::string j = osc.to_stats_json();
    EXPECT_NE(j.find("macd"), std::string::npos);
    EXPECT_NE(j.find("signal_line"), std::string::npos);
    EXPECT_NE(j.find("histogram"), std::string::npos);
    EXPECT_NE(j.find("last_cross"), std::string::npos);
    EXPECT_NE(j.find("total_crosses"), std::string::npos);
    EXPECT_NE(j.find("total_records"), std::string::npos);
}

// ---------------------------------------------------------------------------
// Concurrent record() is thread-safe
// ---------------------------------------------------------------------------

TEST(SignalMomentumOscillator, ConcurrentRecordSafe) {
    SignalMomentumOscillator osc;
    std::atomic<bool> go{false};
    auto worker = [&] {
        while (!go.load()) {}
        for (int i = 0; i < 500; ++i)
            osc.record(static_cast<double>(i % 10) * 0.1 - 0.5);
    };
    std::vector<std::thread> threads;
    for (int t = 0; t < 4; ++t) threads.emplace_back(worker);
    go.store(true);
    for (auto& t : threads) t.join();

    EXPECT_EQ(osc.total_records(), 2000u);
    // Values should be finite.
    EXPECT_TRUE(std::isfinite(osc.macd()));
    EXPECT_TRUE(std::isfinite(osc.histogram()));
}

// ---------------------------------------------------------------------------
// is_bullish / is_bearish reflect last cross direction
// ---------------------------------------------------------------------------

TEST(SignalMomentumOscillator, DirectionFlagsReflectLastCross) {
    SignalMomentumOscillator::Config cfg;
    cfg.fast_alpha   = 0.4;
    cfg.slow_alpha   = 0.08;
    cfg.signal_alpha = 0.5;
    SignalMomentumOscillator osc(cfg);

    // Settle at +1, then drive bearish (histogram negative), then bullish
    // (histogram crosses zero upward → last cross is Bullish).
    for (int i = 0; i < 60; ++i) osc.record(1.0);
    for (int i = 0; i < 60; ++i) osc.record(-1.0);
    for (int i = 0; i < 60; ++i) osc.record(1.0);

    // After the final bullish phase the last cross should be bullish.
    EXPECT_TRUE(osc.is_bullish());
    EXPECT_FALSE(osc.is_bearish());
}

} // namespace

#else

TEST(SignalMomentumOscillator, DisabledAtBuildTime) { SUCCEED(); }

#endif  // LLMQUANT_SIGNAL_MOMENTUM_OSC_ENABLED
