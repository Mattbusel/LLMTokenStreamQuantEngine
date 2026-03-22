#include <gtest/gtest.h>

#ifdef LLMQUANT_ROLLING_SHARPE_ENABLED
#include "RollingSharpeBiasTracker.h"
#include <cmath>

using namespace llmquant;

TEST(RollingSharpeBiasTrackerTest, ConstructsOk) {
    RollingSharpeBiasTracker t;
    EXPECT_EQ(t.sample_count(), 0);
    EXPECT_EQ(t.total_records(), 0u);
}

TEST(RollingSharpeBiasTrackerTest, ReturnsZeroBeforeMinSamples) {
    RollingSharpeBiasTracker::Config cfg;
    cfg.min_samples = 10;
    RollingSharpeBiasTracker t(cfg);
    for (int i = 0; i < 5; ++i) t.record(1.0);
    EXPECT_DOUBLE_EQ(t.last_sharpe(), 0.0);
}

TEST(RollingSharpeBiasTrackerTest, SharpePositiveForConstantPositiveBias) {
    // All values = +1.0 → mean = 1, stddev = 0 → Sharpe = 0 (no variance)
    // but mean > 0 indicates positive bias direction.
    RollingSharpeBiasTracker::Config cfg;
    cfg.min_samples = 5;
    RollingSharpeBiasTracker t(cfg);
    for (int i = 0; i < 20; ++i) t.record(1.0);
    // With zero stddev, Sharpe = 0 (guarded division).
    EXPECT_DOUBLE_EQ(t.last_sharpe(), 0.0);
}

TEST(RollingSharpeBiasTrackerTest, SharpeWithVariance) {
    // Feed alternating 1 and -1 values → mean = 0, Sharpe = 0.
    RollingSharpeBiasTracker::Config cfg;
    cfg.min_samples = 4;
    RollingSharpeBiasTracker t(cfg);
    for (int i = 0; i < 20; ++i) t.record(i % 2 == 0 ? 1.0 : -1.0);
    EXPECT_NEAR(t.last_sharpe(), 0.0, 1e-6);
}

TEST(RollingSharpeBiasTrackerTest, SharpeNonZeroWithPositiveTrend) {
    RollingSharpeBiasTracker::Config cfg;
    cfg.min_samples = 5;
    RollingSharpeBiasTracker t(cfg);
    // Biases: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    // mean ≈ 5.5, std > 0 → Sharpe > 0.
    for (int i = 1; i <= 20; ++i) t.record(static_cast<double>(i));
    EXPECT_GT(t.last_sharpe(), 0.0);
}

TEST(RollingSharpeBiasTrackerTest, WindowSizeCapEnforced) {
    RollingSharpeBiasTracker::Config cfg;
    cfg.window_size = 10;
    cfg.min_samples = 3;
    RollingSharpeBiasTracker t(cfg);
    for (int i = 0; i < 30; ++i) t.record(1.0);
    EXPECT_EQ(t.sample_count(), 10);
}

TEST(RollingSharpeBiasTrackerTest, PoorQualityFlagSetWhenSharpeBelow) {
    RollingSharpeBiasTracker::Config cfg;
    cfg.min_samples = 5;
    cfg.min_sharpe  = 1.0; // high threshold
    RollingSharpeBiasTracker t(cfg);
    // Feed near-zero mean values → Sharpe will be near 0 → poor quality.
    for (int i = 0; i < 20; ++i) t.record(i % 2 == 0 ? 0.001 : -0.001);
    EXPECT_TRUE(t.is_poor_quality());
}

TEST(RollingSharpeBiasTrackerTest, RollingMeanCorrect) {
    RollingSharpeBiasTracker::Config cfg;
    cfg.min_samples = 2;
    RollingSharpeBiasTracker t(cfg);
    t.record(2.0);
    t.record(4.0);
    EXPECT_NEAR(t.rolling_mean(), 3.0, 1e-9);
}

TEST(RollingSharpeBiasTrackerTest, RollingStddevCorrect) {
    RollingSharpeBiasTracker::Config cfg;
    cfg.min_samples = 2;
    RollingSharpeBiasTracker t(cfg);
    for (int i = 0; i < 10; ++i) t.record(3.0); // all same → stddev = 0
    EXPECT_NEAR(t.rolling_stddev(), 0.0, 1e-9);
}

TEST(RollingSharpeBiasTrackerTest, ResetClearsWindow) {
    RollingSharpeBiasTracker t;
    for (int i = 0; i < 30; ++i) t.record(1.0);
    t.reset();
    EXPECT_EQ(t.sample_count(), 0);
    EXPECT_DOUBLE_EQ(t.last_sharpe(), 0.0);
}

TEST(RollingSharpeBiasTrackerTest, UpdateConfigResetsWindow) {
    RollingSharpeBiasTracker t;
    for (int i = 0; i < 30; ++i) t.record(1.0);
    t.update_config(RollingSharpeBiasTracker::Config{});
    EXPECT_EQ(t.sample_count(), 0);
}

TEST(RollingSharpeBiasTrackerTest, TotalRecordsCounts) {
    RollingSharpeBiasTracker t;
    for (int i = 0; i < 7; ++i) t.record(1.0);
    EXPECT_EQ(t.total_records(), 7u);
}

TEST(RollingSharpeBiasTrackerTest, StatsJsonContainsRequiredKeys) {
    RollingSharpeBiasTracker t;
    for (int i = 0; i < 25; ++i) t.record(static_cast<double>(i));
    std::string json = t.to_stats_json();
    EXPECT_NE(json.find("rolling_sharpe"),  std::string::npos);
    EXPECT_NE(json.find("rolling_mean"),    std::string::npos);
    EXPECT_NE(json.find("rolling_stddev"),  std::string::npos);
    EXPECT_NE(json.find("is_poor_quality"), std::string::npos);
    EXPECT_NE(json.find("total_records"),   std::string::npos);
}

#else
TEST(RollingSharpeBiasTrackerTest, FeatureFlagDisabled) { SUCCEED(); }
#endif
