#include <gtest/gtest.h>

#ifdef LLMQUANT_SIGNAL_CONVEXITY_ENABLED

#include "SignalConvexityMeter.h"

#include <atomic>
#include <cmath>
#include <thread>
#include <vector>

using namespace llmquant;

namespace {

// -----------------------------------------------------------------------
// Default state
// -----------------------------------------------------------------------

TEST(SignalConvexityMeter, DefaultState) {
    SignalConvexityMeter m;
    EXPECT_DOUBLE_EQ(m.first_derivative(), 0.0);
    EXPECT_DOUBLE_EQ(m.second_derivative(), 0.0);
    EXPECT_EQ(m.regime(), SignalConvexityMeter::Regime::STABLE);
    EXPECT_EQ(m.total_records(), 0u);
    EXPECT_EQ(m.regime_changes(), 0u);
}

// -----------------------------------------------------------------------
// Total records increments
// -----------------------------------------------------------------------

TEST(SignalConvexityMeter, TotalRecordsIncrements) {
    SignalConvexityMeter m;
    m.record(1.0);
    m.record(2.0);
    m.record(3.0);
    EXPECT_EQ(m.total_records(), 3u);
}

// -----------------------------------------------------------------------
// Linearly increasing signal → positive first derivative
// -----------------------------------------------------------------------

TEST(SignalConvexityMeter, LinearIncreasePositiveD1) {
    SignalConvexityMeter m;
    for (int i = 0; i < 30; ++i) m.record(static_cast<double>(i) * 0.1);
    EXPECT_GT(m.first_derivative(), 0.0);
}

// -----------------------------------------------------------------------
// Accelerating signal (exponential) → ACCELERATING regime
// -----------------------------------------------------------------------

TEST(SignalConvexityMeter, AcceleratingSignalRegime) {
    SignalConvexityMeter::Config cfg;
    cfg.convex_threshold = 0.01;
    cfg.min_samples = 3;
    SignalConvexityMeter m(cfg);
    // Exponential growth: differences are increasing
    double x = 0.0;
    for (int i = 1; i <= 40; ++i) {
        x += static_cast<double>(i) * 0.05;  // step size increases each tick
        m.record(x);
    }
    EXPECT_GT(m.second_derivative(), 0.0);
    EXPECT_EQ(m.regime(), SignalConvexityMeter::Regime::ACCELERATING);
    EXPECT_TRUE(m.is_accelerating());
}

// -----------------------------------------------------------------------
// Decelerating signal → DECELERATING regime
// -----------------------------------------------------------------------

TEST(SignalConvexityMeter, DeceleratingSignalRegime) {
    SignalConvexityMeter::Config cfg;
    cfg.convex_threshold = 0.01;
    cfg.min_samples = 3;
    SignalConvexityMeter m(cfg);
    // Decreasing steps (decelerating downtrend)
    double x = 100.0;
    for (int i = 1; i <= 40; ++i) {
        x -= static_cast<double>(i) * 0.05;
        m.record(x);
    }
    EXPECT_LT(m.second_derivative(), 0.0);
    EXPECT_EQ(m.regime(), SignalConvexityMeter::Regime::DECELERATING);
    EXPECT_TRUE(m.is_decelerating());
}

// -----------------------------------------------------------------------
// Accelerating callback fires
// -----------------------------------------------------------------------

TEST(SignalConvexityMeter, AcceleratingCallbackFires) {
    SignalConvexityMeter::Config cfg;
    cfg.convex_threshold = 0.01;
    cfg.min_samples = 3;
    std::atomic<int> fires{0};
    cfg.on_accelerating = [&](double) { ++fires; };
    SignalConvexityMeter m(cfg);
    double x = 0.0;
    for (int i = 1; i <= 40; ++i) {
        x += static_cast<double>(i) * 0.05;
        m.record(x);
    }
    EXPECT_GE(fires.load(), 1);
    EXPECT_GT(m.regime_changes(), 0u);
}

// -----------------------------------------------------------------------
// Stabilized callback fires after returning from non-STABLE
// -----------------------------------------------------------------------

TEST(SignalConvexityMeter, StabilizedCallbackFires) {
    SignalConvexityMeter::Config cfg;
    cfg.convex_threshold = 0.01;  // wider stable zone for reliable convergence
    cfg.min_samples = 3;
    std::atomic<int> stabs{0};
    cfg.on_stabilized = [&]() { ++stabs; };
    SignalConvexityMeter m(cfg);

    // Push into accelerating regime
    double x = 0.0;
    for (int i = 1; i <= 30; ++i) { x += static_cast<double>(i) * 0.1; m.record(x); }

    // Return to stable with constant tiny steps (60 steps to ensure d2 fully decays)
    for (int i = 0; i < 60; ++i) m.record(x + static_cast<double>(i) * 0.001);

    EXPECT_GE(stabs.load(), 1);
}

// -----------------------------------------------------------------------
// reset() clears state
// -----------------------------------------------------------------------

TEST(SignalConvexityMeter, ResetClearsState) {
    SignalConvexityMeter m;
    for (int i = 0; i < 20; ++i) m.record(static_cast<double>(i));
    m.reset();
    EXPECT_EQ(m.total_records(), 0u);
    EXPECT_DOUBLE_EQ(m.first_derivative(), 0.0);
    EXPECT_DOUBLE_EQ(m.second_derivative(), 0.0);
    EXPECT_EQ(m.regime(), SignalConvexityMeter::Regime::STABLE);
    EXPECT_EQ(m.regime_changes(), 0u);
}

// -----------------------------------------------------------------------
// update_config resets state
// -----------------------------------------------------------------------

TEST(SignalConvexityMeter, UpdateConfigResetsState) {
    SignalConvexityMeter m;
    for (int i = 0; i < 10; ++i) m.record(static_cast<double>(i));
    SignalConvexityMeter::Config cfg2;
    cfg2.convex_threshold = 0.01;
    m.update_config(cfg2);
    EXPECT_EQ(m.total_records(), 0u);
    EXPECT_DOUBLE_EQ(m.second_derivative(), 0.0);
}

// -----------------------------------------------------------------------
// to_stats_json contains expected keys
// -----------------------------------------------------------------------

TEST(SignalConvexityMeter, ToStatsJsonContainsKeys) {
    SignalConvexityMeter m;
    for (int i = 0; i < 10; ++i) m.record(static_cast<double>(i) * 0.1);
    std::string j = m.to_stats_json();
    EXPECT_NE(j.find("first_derivative"),  std::string::npos);
    EXPECT_NE(j.find("second_derivative"), std::string::npos);
    EXPECT_NE(j.find("regime"),            std::string::npos);
    EXPECT_NE(j.find("regime_changes"),    std::string::npos);
    EXPECT_NE(j.find("total_records"),     std::string::npos);
}

// -----------------------------------------------------------------------
// Concurrent record() is thread-safe
// -----------------------------------------------------------------------

TEST(SignalConvexityMeter, ConcurrentRecordSafe) {
    SignalConvexityMeter m;
    std::vector<std::thread> threads;
    for (int t = 0; t < 4; ++t) {
        threads.emplace_back([&, t] {
            for (int i = 0; i < 250; ++i)
                m.record(static_cast<double>(t % 2 == 0 ? i : -i) * 0.01);
        });
    }
    for (auto& th : threads) th.join();
    EXPECT_EQ(m.total_records(), 1000u);
    EXPECT_TRUE(std::isfinite(m.first_derivative()));
    EXPECT_TRUE(std::isfinite(m.second_derivative()));
}

} // namespace

#else

TEST(SignalConvexityMeter, DisabledAtBuildTime) { SUCCEED(); }

#endif  // LLMQUANT_SIGNAL_CONVEXITY_ENABLED
