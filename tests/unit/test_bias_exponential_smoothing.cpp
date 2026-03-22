#include <gtest/gtest.h>
#include "BiasExponentialSmoothing.h"

using namespace llmquant;

TEST(BiasExponentialSmoothing, DefaultConstruction) {
    BiasExponentialSmoothing e(BiasExponentialSmoothing::Config{});
    EXPECT_EQ(e.total_records(), 0u);
    EXPECT_DOUBLE_EQ(e.level(), 0.0);
    EXPECT_DOUBLE_EQ(e.trend(), 0.0);
}

TEST(BiasExponentialSmoothing, LevelTracksInput) {
    BiasExponentialSmoothing::Config cfg;
    cfg.alpha = 0.5;
    cfg.beta  = 0.1;
    BiasExponentialSmoothing e(cfg);
    for (int i = 0; i < 10; ++i) e.record(1.0);
    EXPECT_NEAR(e.level(), 1.0, 0.01);
}

TEST(BiasExponentialSmoothing, TrendPositiveForUptrend) {
    BiasExponentialSmoothing::Config cfg;
    cfg.alpha = 0.3;
    cfg.beta  = 0.3;
    BiasExponentialSmoothing e(cfg);
    for (int i = 0; i < 15; ++i) e.record(0.1 * i);
    EXPECT_GT(e.trend(), 0.0);
}

TEST(BiasExponentialSmoothing, ForecastIsLevelPlusTrend) {
    BiasExponentialSmoothing e(BiasExponentialSmoothing::Config{});
    for (int i = 0; i < 5; ++i) e.record(0.1 * i);
    EXPECT_NEAR(e.forecast(), e.level() + e.trend(), 1e-9);
}

TEST(BiasExponentialSmoothing, LargeErrorCallback) {
    BiasExponentialSmoothing::Config cfg;
    cfg.alpha = 0.5;
    cfg.beta  = 0.1;
    cfg.min_samples = 2;
    cfg.error_threshold = 0.01;
    int err_count = 0;
    cfg.on_large_error = [&](double, double) { ++err_count; };
    BiasExponentialSmoothing e(cfg);
    // Seed with stable value, then jump
    for (int i = 0; i < 5; ++i) e.record(0.0);
    e.record(10.0); // large jump
    EXPECT_GE(err_count, 1);
}

TEST(BiasExponentialSmoothing, Reset) {
    BiasExponentialSmoothing e(BiasExponentialSmoothing::Config{});
    for (int i = 0; i < 5; ++i) e.record(0.1);
    e.reset();
    EXPECT_EQ(e.total_records(), 0u);
    EXPECT_DOUBLE_EQ(e.level(), 0.0);
    EXPECT_DOUBLE_EQ(e.trend(), 0.0);
    EXPECT_EQ(e.error_events(), 0u);
}

TEST(BiasExponentialSmoothing, ToStatsJson) {
    BiasExponentialSmoothing e(BiasExponentialSmoothing::Config{});
    for (int i = 0; i < 5; ++i) e.record(0.05);
    std::string j = e.to_stats_json();
    EXPECT_NE(j.find("level"), std::string::npos);
    EXPECT_NE(j.find("forecast"), std::string::npos);
    EXPECT_NE(j.find("total_records"), std::string::npos);
}

TEST(BiasExponentialSmoothing, LastErrorNonNegative) {
    BiasExponentialSmoothing::Config cfg;
    cfg.min_samples = 2;
    BiasExponentialSmoothing e(cfg);
    for (int i = 0; i < 5; ++i) e.record(0.1 * i);
    EXPECT_GE(e.last_error(), 0.0);
}
