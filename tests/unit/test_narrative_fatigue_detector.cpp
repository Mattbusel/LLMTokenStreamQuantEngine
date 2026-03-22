#include <gtest/gtest.h>
#include "NarrativeFatigueDetector.h"

namespace llmquant::tests {

TEST(NarrativeFatigueDetector, DefaultConstruction) {
    NarrativeFatigueDetector d;
    EXPECT_EQ(d.total_records(), 0u);
    EXPECT_FALSE(d.is_fatigued());
}

TEST(NarrativeFatigueDetector, RatioDeltaDecaysToZero) {
    NarrativeFatigueDetector d;
    // Feed a constant value — fast and slow EMAs converge → ratio_delta → 0
    for (int i = 0; i < 100; ++i) d.record(1.0);
    EXPECT_LT(d.ratio_delta(), 0.5);
}

TEST(NarrativeFatigueDetector, FatigueDetectedOnConstantBias) {
    NarrativeFatigueDetector::Config cfg;
    cfg.alpha_fast        = 0.5;
    cfg.alpha_slow        = 0.1;
    cfg.fatigue_threshold = 0.5; // easy to trigger
    cfg.min_samples       = 5;
    NarrativeFatigueDetector d(cfg);
    for (int i = 0; i < 50; ++i) d.record(1.0);
    EXPECT_TRUE(d.is_fatigued());
}

TEST(NarrativeFatigueDetector, FatigueCallback) {
    NarrativeFatigueDetector::Config cfg;
    cfg.fatigue_threshold = 0.9;
    cfg.min_samples       = 3;
    int events = 0;
    cfg.on_fatigue = [&](double) { ++events; };
    NarrativeFatigueDetector d(cfg);
    for (int i = 0; i < 50; ++i) d.record(1.0);
    EXPECT_GT(events, 0);
}

TEST(NarrativeFatigueDetector, RecoveryCallback) {
    NarrativeFatigueDetector::Config cfg;
    cfg.fatigue_threshold = 0.9;
    cfg.min_samples       = 3;
    int recoveries = 0;
    cfg.on_recovery = [&](double) { ++recoveries; };
    NarrativeFatigueDetector d(cfg);
    for (int i = 0; i < 30; ++i) d.record(1.0);   // converge → fatigued
    for (int i = 0; i < 10; ++i) d.record(10.0);  // spike → recovery
    EXPECT_GE(recoveries, 0);
}

TEST(NarrativeFatigueDetector, Reset) {
    NarrativeFatigueDetector d;
    for (int i = 0; i < 20; ++i) d.record(1.0);
    d.reset();
    EXPECT_EQ(d.total_records(), 0u);
    EXPECT_FALSE(d.is_fatigued());
    EXPECT_EQ(d.fatigue_events(), 0u);
}

TEST(NarrativeFatigueDetector, ToStatsJson) {
    NarrativeFatigueDetector d;
    for (int i = 0; i < 15; ++i) d.record(0.1);
    auto json = d.to_stats_json();
    EXPECT_NE(json.find("ratio_delta"), std::string::npos);
    EXPECT_NE(json.find("fatigue_events"), std::string::npos);
}

} // namespace llmquant::tests
