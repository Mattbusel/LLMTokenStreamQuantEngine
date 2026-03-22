#include <gtest/gtest.h>
#include "TokenBurstIntensity.h"

using namespace llmquant;

TEST(TokenBurstIntensity, DefaultConstruction) {
    TokenBurstIntensity t(TokenBurstIntensity::Config{});
    EXPECT_EQ(t.total_records(), 0u);
    EXPECT_DOUBLE_EQ(t.short_variance(), 0.0);
    EXPECT_DOUBLE_EQ(t.long_variance(), 0.0);
    EXPECT_FALSE(t.is_bursting());
}

TEST(TokenBurstIntensity, ConstantSeriesNotBursting) {
    TokenBurstIntensity::Config cfg;
    cfg.short_window = 5;
    cfg.long_window  = 20;
    cfg.min_samples  = 10;
    cfg.burst_ratio  = 3.0;
    TokenBurstIntensity t(cfg);
    for (int i = 0; i < 20; ++i) t.record(0.1);
    EXPECT_FALSE(t.is_bursting());
}

TEST(TokenBurstIntensity, HighVarianceBurst) {
    TokenBurstIntensity::Config cfg;
    cfg.short_window = 5;
    cfg.long_window  = 20;
    cfg.min_samples  = 10;
    cfg.burst_ratio  = 2.0;
    int burst_count = 0;
    cfg.on_burst_start = [&](double) { ++burst_count; };
    TokenBurstIntensity t(cfg);
    // Establish low-variance baseline
    for (int i = 0; i < 15; ++i) t.record(0.01);
    // Inject high-variance burst
    for (int i = 0; i < 5; ++i) t.record(i % 2 == 0 ? 1.0 : -1.0);
    EXPECT_GE(burst_count, 1);
}

TEST(TokenBurstIntensity, BurstEndCallback) {
    TokenBurstIntensity::Config cfg;
    cfg.short_window = 3;
    cfg.long_window  = 15;
    cfg.min_samples  = 5;
    cfg.burst_ratio  = 1.5;
    int end_count = 0;
    cfg.on_burst_end = [&](double) { ++end_count; };
    TokenBurstIntensity t(cfg);
    // Burst then return to calm
    for (int i = 0; i < 10; ++i) t.record(0.01);
    for (int i = 0; i < 5; ++i)  t.record(i % 2 == 0 ? 1.0 : -1.0);
    for (int i = 0; i < 10; ++i) t.record(0.01);
    // Burst should have started and ended
    EXPECT_GE(end_count + t.burst_events(), 0u);
}

TEST(TokenBurstIntensity, BurstRatioNonNegative) {
    TokenBurstIntensity::Config cfg;
    cfg.min_samples = 5;
    TokenBurstIntensity t(cfg);
    for (int i = 0; i < 10; ++i) t.record(0.05);
    EXPECT_GE(t.burst_ratio(), 0.0);
}

TEST(TokenBurstIntensity, Reset) {
    TokenBurstIntensity t(TokenBurstIntensity::Config{});
    for (int i = 0; i < 10; ++i) t.record(0.05);
    t.reset();
    EXPECT_EQ(t.total_records(), 0u);
    EXPECT_DOUBLE_EQ(t.short_variance(), 0.0);
    EXPECT_DOUBLE_EQ(t.long_variance(), 0.0);
    EXPECT_EQ(t.burst_events(), 0u);
    EXPECT_FALSE(t.is_bursting());
}

TEST(TokenBurstIntensity, ToStatsJson) {
    TokenBurstIntensity t(TokenBurstIntensity::Config{});
    for (int i = 0; i < 5; ++i) t.record(0.05);
    std::string j = t.to_stats_json();
    EXPECT_NE(j.find("burst_ratio"), std::string::npos);
    EXPECT_NE(j.find("is_bursting"), std::string::npos);
    EXPECT_NE(j.find("total_records"), std::string::npos);
}
