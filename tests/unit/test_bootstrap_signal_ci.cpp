#include <gtest/gtest.h>

#ifdef LLMQUANT_BOOTSTRAP_SIGNAL_CI_ENABLED

#include "BootstrapSignalCI.hpp"

#include <cmath>
#include <thread>
#include <vector>

using namespace llmquant;

namespace {

// ============================================================
// Construction / validation
// ============================================================

TEST(BootstrapSignalCI, DefaultConstructs)
{
    BootstrapSignalCI ci;
    EXPECT_EQ(ci.observation_count(), 0u);
    EXPECT_FALSE(ci.is_ready());
    EXPECT_EQ(ci.total_recorded(), 0u);
}

TEST(BootstrapSignalCI, InvalidNBootstrapThrows)
{
    EXPECT_THROW(BootstrapSignalCI(1, 0.95f, 64, 4), std::invalid_argument);
}

TEST(BootstrapSignalCI, InvalidConfidenceLevelThrows)
{
    EXPECT_THROW(BootstrapSignalCI(100, 0.0f, 64, 4), std::invalid_argument);
    EXPECT_THROW(BootstrapSignalCI(100, 1.0f, 64, 4), std::invalid_argument);
}

TEST(BootstrapSignalCI, InvalidWindowSizeThrows)
{
    EXPECT_THROW(BootstrapSignalCI(100, 0.95f, 0, 4), std::invalid_argument);
}

TEST(BootstrapSignalCI, MinObservationsExceedsWindowThrows)
{
    EXPECT_THROW(BootstrapSignalCI(100, 0.95f, 10, 20), std::invalid_argument);
}

// ============================================================
// is_ready
// ============================================================

TEST(BootstrapSignalCI, NotReadyBeforeMinObservations)
{
    BootstrapSignalCI ci(50, 0.95f, 64, 5);
    for (int i = 0; i < 4; ++i) {
        ci.add_observation(static_cast<float>(i) * 0.1f);
        EXPECT_FALSE(ci.is_ready());
    }
}

TEST(BootstrapSignalCI, ReadyAfterMinObservations)
{
    BootstrapSignalCI ci(50, 0.95f, 64, 5);
    for (int i = 0; i < 5; ++i)
        ci.add_observation(0.5f);
    EXPECT_TRUE(ci.is_ready());
}

// ============================================================
// compute() — structural properties
// ============================================================

TEST(BootstrapSignalCI, ComputeBeforeReadyReturnsDegenerateInterval)
{
    BootstrapSignalCI ci(50, 0.95f, 64, 5);
    ci.add_observation(0.3f);
    ci.add_observation(0.7f);
    // Not ready (< 5 observations) — should return lower == upper == median.
    auto interval = ci.compute();
    EXPECT_FLOAT_EQ(interval.lower,  interval.upper);
    EXPECT_FLOAT_EQ(interval.lower,  interval.median);
}

TEST(BootstrapSignalCI, LowerLeqMedianLeqUpper)
{
    BootstrapSignalCI ci(200, 0.95f, 64, 8);
    for (int i = 0; i < 20; ++i)
        ci.add_observation(static_cast<float>(i) * 0.05f);
    ASSERT_TRUE(ci.is_ready());
    auto interval = ci.compute();
    EXPECT_LE(interval.lower,  interval.median);
    EXPECT_LE(interval.median, interval.upper);
}

TEST(BootstrapSignalCI, ConstantSignalNarrowInterval)
{
    // All observations the same → zero variance → CI collapses to a point.
    BootstrapSignalCI ci(200, 0.95f, 64, 8);
    for (int i = 0; i < 20; ++i)
        ci.add_observation(0.5f);
    auto interval = ci.compute();
    EXPECT_NEAR(interval.lower,  0.5f, 1e-3f);
    EXPECT_NEAR(interval.upper,  0.5f, 1e-3f);
    EXPECT_NEAR(interval.median, 0.5f, 1e-3f);
}

TEST(BootstrapSignalCI, PositiveSignalIntervalPositive)
{
    // All strongly bullish signals → both bounds should be positive.
    BootstrapSignalCI ci(300, 0.95f, 64, 8);
    for (int i = 0; i < 30; ++i)
        ci.add_observation(1.0f);
    auto interval = ci.compute();
    EXPECT_GT(interval.lower, 0.0f);
    EXPECT_GT(interval.upper, 0.0f);
}

TEST(BootstrapSignalCI, MixedSignalIntervalSpansZero)
{
    // Equal positive and negative observations → CI should span zero.
    BootstrapSignalCI ci(300, 0.95f, 64, 8);
    for (int i = 0; i < 30; ++i)
        ci.add_observation(i % 2 == 0 ? +1.0f : -1.0f);
    auto interval = ci.compute();
    EXPECT_LT(interval.lower, 0.1f);
    EXPECT_GT(interval.upper, -0.1f);
}

// ============================================================
// Window eviction
// ============================================================

TEST(BootstrapSignalCI, WindowCapRespected)
{
    BootstrapSignalCI ci(50, 0.95f, 10, 4);
    for (int i = 0; i < 30; ++i)
        ci.add_observation(static_cast<float>(i) * 0.1f);
    EXPECT_EQ(ci.observation_count(), 10u);   // capped at window_size=10
    EXPECT_EQ(ci.total_recorded(),    30u);
}

// ============================================================
// reset()
// ============================================================

TEST(BootstrapSignalCI, ResetClearsState)
{
    BootstrapSignalCI ci(50, 0.95f, 64, 4);
    for (int i = 0; i < 10; ++i)
        ci.add_observation(0.5f);
    EXPECT_TRUE(ci.is_ready());
    ci.reset();
    EXPECT_FALSE(ci.is_ready());
    EXPECT_EQ(ci.observation_count(), 0u);
    auto interval = ci.compute();
    EXPECT_FLOAT_EQ(interval.lower,  0.0f);
    EXPECT_FLOAT_EQ(interval.upper,  0.0f);
    EXPECT_FLOAT_EQ(interval.median, 0.0f);
}

// ============================================================
// to_stats_json()
// ============================================================

TEST(BootstrapSignalCI, ToStatsJsonContainsExpectedKeys)
{
    BootstrapSignalCI ci(50, 0.95f, 64, 4);
    for (int i = 0; i < 8; ++i)
        ci.add_observation(static_cast<float>(i) * 0.1f);
    std::string json = ci.to_stats_json();
    EXPECT_NE(json.find("lower"),             std::string::npos);
    EXPECT_NE(json.find("median"),            std::string::npos);
    EXPECT_NE(json.find("upper"),             std::string::npos);
    EXPECT_NE(json.find("confidence_level"),  std::string::npos);
    EXPECT_NE(json.find("n_bootstrap"),       std::string::npos);
    EXPECT_NE(json.find("observation_count"), std::string::npos);
    EXPECT_NE(json.find("total_recorded"),    std::string::npos);
    EXPECT_NE(json.find("is_ready"),          std::string::npos);
}

// ============================================================
// Thread safety
// ============================================================

TEST(BootstrapSignalCI, ConcurrentAddObservationSafe)
{
    BootstrapSignalCI ci(200, 0.95f, 128, 8);
    std::vector<std::thread> threads;
    for (int t = 0; t < 4; ++t) {
        threads.emplace_back([&, t] {
            for (int i = 0; i < 50; ++i)
                ci.add_observation(static_cast<float>(t) * 0.1f + i * 0.01f);
        });
    }
    for (auto& th : threads) th.join();
    EXPECT_EQ(ci.total_recorded(), 200u);
    EXPECT_LE(ci.observation_count(), 128u);
    // compute() must not crash.
    EXPECT_TRUE(ci.is_ready());
    auto interval = ci.compute();
    EXPECT_LE(interval.lower, interval.upper);
}

} // namespace

#else

TEST(BootstrapSignalCI, DisabledAtBuildTime) { SUCCEED(); }

#endif  // LLMQUANT_BOOTSTRAP_SIGNAL_CI_ENABLED
