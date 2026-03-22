#define LLMQUANT_ADAPTIVE_FILTER_ENABLED
#include <gtest/gtest.h>
#include "SignalAdaptiveThresholdFilter.h"

using namespace llmquant;

TEST(SignalAdaptiveThresholdFilter, DefaultConstruction) {
    SignalAdaptiveThresholdFilter f(SignalAdaptiveThresholdFilter::Config{});
    EXPECT_EQ(f.total_records(), 0u);
    EXPECT_DOUBLE_EQ(f.atr(), 0.0);
}

TEST(SignalAdaptiveThresholdFilter, ThresholdIsMultiplierTimesATR) {
    SignalAdaptiveThresholdFilter::Config cfg;
    cfg.alpha_atr = 0.5;
    cfg.multiplier = 2.0;
    SignalAdaptiveThresholdFilter f(cfg);
    for (int i = 0; i < 5; ++i) f.record(0.1 * i);
    EXPECT_NEAR(f.threshold(), 2.0 * f.atr(), 1e-9);
}

TEST(SignalAdaptiveThresholdFilter, AboveCallback) {
    SignalAdaptiveThresholdFilter::Config cfg;
    cfg.alpha_atr = 0.1;
    cfg.multiplier = 1.0;
    cfg.min_samples = 3;
    int above_count = 0;
    cfg.on_above = [&](double, double) { ++above_count; };
    SignalAdaptiveThresholdFilter f(cfg);
    for (int i = 0; i < 5; ++i) f.record(0.0);
    f.record(100.0); // massively above any threshold
    EXPECT_GE(above_count, 1);
}

TEST(SignalAdaptiveThresholdFilter, BelowCallback) {
    SignalAdaptiveThresholdFilter::Config cfg;
    cfg.alpha_atr = 0.1;
    cfg.multiplier = 1.0;
    cfg.min_samples = 3;
    int below_count = 0;
    cfg.on_below = [&](double, double) { ++below_count; };
    SignalAdaptiveThresholdFilter f(cfg);
    for (int i = 0; i < 5; ++i) f.record(0.0);
    f.record(-100.0);
    EXPECT_GE(below_count, 1);
}

TEST(SignalAdaptiveThresholdFilter, ATRNonNegative) {
    SignalAdaptiveThresholdFilter f(SignalAdaptiveThresholdFilter::Config{});
    for (int i = 0; i < 5; ++i) f.record(0.1 * i);
    EXPECT_GE(f.atr(), 0.0);
    EXPECT_GE(f.threshold(), 0.0);
}

TEST(SignalAdaptiveThresholdFilter, Reset) {
    SignalAdaptiveThresholdFilter f(SignalAdaptiveThresholdFilter::Config{});
    for (int i = 0; i < 5; ++i) f.record(0.1);
    f.reset();
    EXPECT_EQ(f.total_records(), 0u);
    EXPECT_DOUBLE_EQ(f.atr(), 0.0);
    EXPECT_EQ(f.above_events(), 0u);
    EXPECT_EQ(f.below_events(), 0u);
}

TEST(SignalAdaptiveThresholdFilter, ToStatsJson) {
    SignalAdaptiveThresholdFilter f(SignalAdaptiveThresholdFilter::Config{});
    for (int i = 0; i < 5; ++i) f.record(0.05);
    std::string j = f.to_stats_json();
    EXPECT_NE(j.find("atr"), std::string::npos);
    EXPECT_NE(j.find("threshold"), std::string::npos);
    EXPECT_NE(j.find("total_records"), std::string::npos);
}
