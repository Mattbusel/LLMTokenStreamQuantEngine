#include <gtest/gtest.h>

#ifdef LLMQUANT_SIGNAL_CI_ENABLED

#include "SignalConfidenceInterval.h"

#include <atomic>
#include <cmath>
#include <thread>
#include <vector>

using namespace llmquant;

namespace {

// ============================================================
// Default construction
// ============================================================

TEST(SignalConfidenceInterval, DefaultConstructs) {
    SignalConfidenceInterval ci;
    EXPECT_NEAR(ci.mean(),       0.0, 1e-9);
    EXPECT_NEAR(ci.half_width(), 0.0, 1e-9);
    EXPECT_EQ(ci.total_records(), 0u);
    EXPECT_EQ(ci.window_fill(), 0u);
}

// ============================================================
// Single record — mean is set, half_width is 0
// ============================================================

TEST(SignalConfidenceInterval, SingleRecord) {
    SignalConfidenceInterval ci;
    ci.record(0.7);
    EXPECT_NEAR(ci.mean(), 0.7, 1e-9);
    EXPECT_NEAR(ci.half_width(), 0.0, 1e-9);
}

// ============================================================
// Mean is correct
// ============================================================

TEST(SignalConfidenceInterval, MeanCorrect) {
    SignalConfidenceInterval::Config cfg;
    cfg.window_size = 8;
    SignalConfidenceInterval ci(cfg);
    for (int i = 1; i <= 4; ++i)
        ci.record(static_cast<double>(i));
    // mean = (1+2+3+4)/4 = 2.5
    EXPECT_NEAR(ci.mean(), 2.5, 1e-9);
}

// ============================================================
// lower = mean - half_width, upper = mean + half_width
// ============================================================

TEST(SignalConfidenceInterval, BoundsConsistent) {
    SignalConfidenceInterval ci;
    for (int i = 0; i < 10; ++i) ci.record(0.5);
    EXPECT_NEAR(ci.lower(), ci.mean() - ci.half_width(), 1e-12);
    EXPECT_NEAR(ci.upper(), ci.mean() + ci.half_width(), 1e-12);
}

// ============================================================
// Constant signal → zero variance → zero half-width
// ============================================================

TEST(SignalConfidenceInterval, ConstantSignalZeroWidth) {
    SignalConfidenceInterval::Config cfg;
    cfg.window_size = 16;
    SignalConfidenceInterval ci(cfg);
    for (int i = 0; i < 16; ++i) ci.record(0.5);
    EXPECT_NEAR(ci.half_width(), 0.0, 1e-9);
    EXPECT_TRUE(ci.is_narrow());
}

// ============================================================
// High variance signal → wide interval
// ============================================================

TEST(SignalConfidenceInterval, HighVarianceIsWide) {
    SignalConfidenceInterval::Config cfg;
    cfg.window_size    = 16;
    cfg.wide_threshold = 0.05;  // low threshold so we can trigger easily
    SignalConfidenceInterval ci(cfg);
    for (int i = 0; i < 16; ++i)
        ci.record(i % 2 == 0 ? 1.0 : -1.0);  // alternating ±1
    EXPECT_TRUE(ci.is_wide());
}

// ============================================================
// on_narrow_interval callback fires
// ============================================================

TEST(SignalConfidenceInterval, NarrowCallbackFires) {
    SignalConfidenceInterval::Config cfg;
    cfg.window_size     = 8;
    cfg.narrow_threshold = 0.1;
    std::atomic<int> fires{0};
    cfg.on_narrow_interval = [&](double, double) { ++fires; };
    SignalConfidenceInterval ci(cfg);

    // Constant signal → zero half-width → narrow.
    for (int i = 0; i < 8; ++i) ci.record(0.5);
    EXPECT_GT(fires.load(), 0);
}

// ============================================================
// on_wide_interval callback fires
// ============================================================

TEST(SignalConfidenceInterval, WideCallbackFires) {
    SignalConfidenceInterval::Config cfg;
    cfg.window_size    = 8;
    cfg.wide_threshold = 0.01;  // very low threshold
    std::atomic<int> fires{0};
    cfg.on_wide_interval = [&](double, double) { ++fires; };
    SignalConfidenceInterval ci(cfg);

    // High variance signal.
    for (int i = 0; i < 8; ++i)
        ci.record(i % 2 == 0 ? 1.0 : -1.0);
    EXPECT_GT(fires.load(), 0);
}

// ============================================================
// reset() clears window
// ============================================================

TEST(SignalConfidenceInterval, ResetClearsWindow) {
    SignalConfidenceInterval ci;
    for (int i = 0; i < 10; ++i) ci.record(0.5);
    ci.reset();
    EXPECT_EQ(ci.window_fill(), 0u);
    EXPECT_NEAR(ci.mean(), 0.0, 1e-9);
    EXPECT_NEAR(ci.half_width(), 0.0, 1e-9);
}

// ============================================================
// to_stats_json contains expected keys
// ============================================================

TEST(SignalConfidenceInterval, ToStatsJsonContainsKeys) {
    SignalConfidenceInterval ci;
    ci.record(0.3);
    std::string json = ci.to_stats_json();
    EXPECT_NE(json.find("mean"),         std::string::npos);
    EXPECT_NE(json.find("half_width"),   std::string::npos);
    EXPECT_NE(json.find("lower"),        std::string::npos);
    EXPECT_NE(json.find("upper"),        std::string::npos);
    EXPECT_NE(json.find("is_narrow"),    std::string::npos);
    EXPECT_NE(json.find("total_records"), std::string::npos);
}

// ============================================================
// Concurrent record() is thread-safe
// ============================================================

TEST(SignalConfidenceInterval, ConcurrentRecordSafe) {
    SignalConfidenceInterval::Config cfg;
    cfg.window_size      = 64;
    cfg.narrow_threshold = 0.0;   // disable callbacks
    cfg.wide_threshold   = 100.0;
    SignalConfidenceInterval ci(cfg);

    std::vector<std::thread> threads;
    for (int t = 0; t < 4; ++t) {
        threads.emplace_back([&, t] {
            for (int i = 0; i < 50; ++i)
                ci.record(static_cast<double>(t) * 0.1 + i * 0.001);
        });
    }
    for (auto& th : threads) th.join();

    EXPECT_EQ(ci.total_records(), 200u);
    EXPECT_GE(ci.half_width(), 0.0);
}

} // namespace

#else

TEST(SignalConfidenceInterval, DisabledAtBuildTime) { SUCCEED(); }

#endif  // LLMQUANT_SIGNAL_CI_ENABLED
