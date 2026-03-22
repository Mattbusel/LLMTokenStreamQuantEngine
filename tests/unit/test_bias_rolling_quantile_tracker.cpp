#include <gtest/gtest.h>

#ifdef LLMQUANT_ROLLING_QUANTILE_ENABLED
#include "BiasRollingQuantileTracker.h"
#include <atomic>
#include <thread>
using namespace llmquant;

TEST(BiasRollingQuantileTracker, DefaultState) {
    BiasRollingQuantileTracker t;
    EXPECT_EQ(t.total_records(), 0u);
    EXPECT_EQ(t.median_shift_events(), 0u);
}

TEST(BiasRollingQuantileTracker, TotalRecordsIncrements) {
    BiasRollingQuantileTracker t;
    for (int i = 0; i < 20; ++i) t.record(static_cast<double>(i) * 0.1);
    EXPECT_EQ(t.total_records(), 20u);
}

TEST(BiasRollingQuantileTracker, QuantilesSorted) {
    BiasRollingQuantileTracker t;
    for (int i = 0; i < 30; ++i)
        t.record(static_cast<double>(i % 10) * 0.1 - 0.5);
    EXPECT_LE(t.p10(), t.p25());
    EXPECT_LE(t.p25(), t.p50());
    EXPECT_LE(t.p50(), t.p75());
    EXPECT_LE(t.p75(), t.p90());
}

TEST(BiasRollingQuantileTracker, ConstantSeriesAllQuantilesEqual) {
    BiasRollingQuantileTracker::Config cfg;
    cfg.window      = 16;
    cfg.min_samples = 4;
    BiasRollingQuantileTracker t(cfg);
    for (int i = 0; i < 20; ++i) t.record(0.5);
    EXPECT_NEAR(t.p10(), 0.5, 1e-6);
    EXPECT_NEAR(t.p50(), 0.5, 1e-6);
    EXPECT_NEAR(t.p90(), 0.5, 1e-6);
}

TEST(BiasRollingQuantileTracker, IQRNonNegative) {
    BiasRollingQuantileTracker t;
    for (int i = 0; i < 30; ++i)
        t.record(static_cast<double>(i % 7) * 0.15 - 0.5);
    EXPECT_GE(t.iqr(), 0.0);
}

TEST(BiasRollingQuantileTracker, MedianShiftCallbackFires) {
    BiasRollingQuantileTracker::Config cfg;
    cfg.window                  = 16;
    cfg.min_samples             = 4;
    cfg.median_change_threshold = 0.05;
    std::atomic<int> fires{0};
    cfg.on_median_shift = [&](double, double) { ++fires; };
    BiasRollingQuantileTracker t(cfg);
    for (int i = 0; i < 20; ++i) t.record(-0.5);
    for (int i = 0; i < 20; ++i) t.record(0.5);
    EXPECT_GE(fires.load(), 1);
    EXPECT_GE(t.median_shift_events(), 1u);
}

TEST(BiasRollingQuantileTracker, SkewRatioPositive) {
    BiasRollingQuantileTracker t;
    for (int i = 0; i < 50; ++i)
        t.record(static_cast<double>(i) * 0.01);
    EXPECT_GE(t.skew_ratio(), 0.0);
}

TEST(BiasRollingQuantileTracker, ResetClearsState) {
    BiasRollingQuantileTracker t;
    for (int i = 0; i < 20; ++i) t.record(0.3);
    t.reset();
    EXPECT_EQ(t.total_records(), 0u);
    EXPECT_EQ(t.median_shift_events(), 0u);
    EXPECT_NEAR(t.p50(), 0.0, 1e-6);
}

TEST(BiasRollingQuantileTracker, UpdateConfigResetsState) {
    BiasRollingQuantileTracker t;
    for (int i = 0; i < 20; ++i) t.record(0.3);
    BiasRollingQuantileTracker::Config cfg2;
    cfg2.window = 32;
    t.update_config(cfg2);
    EXPECT_EQ(t.total_records(), 0u);
}

TEST(BiasRollingQuantileTracker, StatsJsonContainsFields) {
    BiasRollingQuantileTracker t;
    for (int i = 0; i < 20; ++i) t.record(static_cast<double>(i) * 0.05);
    std::string j = t.to_stats_json();
    EXPECT_NE(j.find("p10"),                 std::string::npos);
    EXPECT_NE(j.find("p50"),                 std::string::npos);
    EXPECT_NE(j.find("p90"),                 std::string::npos);
    EXPECT_NE(j.find("skew_ratio"),          std::string::npos);
    EXPECT_NE(j.find("median_shift_events"), std::string::npos);
    EXPECT_NE(j.find("total_records"),       std::string::npos);
}

TEST(BiasRollingQuantileTracker, ConcurrentRecordSafe) {
    BiasRollingQuantileTracker t;
    std::atomic<bool> go{false};
    auto w0 = [&] { while (!go) {} for (int i = 0; i < 200; ++i) t.record(static_cast<double>(i % 10) * 0.1); };
    auto w1 = [&] { while (!go) {} for (int i = 0; i < 200; ++i) t.record(static_cast<double>(i % 5) * 0.2 - 0.5); };
    std::thread t0(w0), t1(w1);
    go = true;
    t0.join(); t1.join();
    EXPECT_EQ(t.total_records(), 400u);
    EXPECT_LE(t.p10(), t.p90() + 1e-9);
}

#else
TEST(BiasRollingQuantileTracker, DisabledAtBuildTime) { SUCCEED(); }
#endif
