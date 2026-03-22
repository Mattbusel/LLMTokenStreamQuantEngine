#include <gtest/gtest.h>
#include "BiasImpulseDetector.h"
#include <cmath>

namespace llmquant::tests {

TEST(BiasImpulseDetector, DefaultConstruction) {
    BiasImpulseDetector d;
    EXPECT_EQ(d.total_records(), 0u);
    EXPECT_EQ(d.impulse_events(), 0u);
}

TEST(BiasImpulseDetector, RecordsAccumulate) {
    BiasImpulseDetector d;
    for (int i = 0; i < 20; ++i) d.record(0.1);
    EXPECT_EQ(d.total_records(), 20u);
}

TEST(BiasImpulseDetector, LargeSpikeDetected) {
    BiasImpulseDetector::Config cfg;
    cfg.window      = 10;
    cfg.min_samples = 5;
    cfg.z_threshold = 2.0;
    BiasImpulseDetector d(cfg);
    for (int i = 0; i < 10; ++i) d.record(0.01);
    d.record(10.0);  // huge spike → z >> 2
    EXPECT_GT(d.impulse_events(), 0u);
}

TEST(BiasImpulseDetector, ImpulseCallback) {
    BiasImpulseDetector::Config cfg;
    cfg.z_threshold = 2.0;
    cfg.min_samples = 5;
    int impulses = 0;
    cfg.on_impulse = [&](double, double) { ++impulses; };
    BiasImpulseDetector d(cfg);
    for (int i = 0; i < 10; ++i) d.record(0.01);
    d.record(100.0);
    EXPECT_GT(impulses, 0);
}

TEST(BiasImpulseDetector, MaxZIncreases) {
    BiasImpulseDetector d;
    for (int i = 0; i < 20; ++i) d.record((double)(i % 3 - 1) * 0.1);
    EXPECT_GE(d.max_z(), 0.0);
}

TEST(BiasImpulseDetector, ZScoreReasonable) {
    BiasImpulseDetector d;
    for (int i = 0; i < 30; ++i) d.record(0.0);
    d.record(0.0);
    // All zeros → z = 0
    EXPECT_NEAR(d.last_z(), 0.0, 1e-6);
}

TEST(BiasImpulseDetector, Reset) {
    BiasImpulseDetector d;
    for (int i = 0; i < 20; ++i) d.record(0.1);
    d.reset();
    EXPECT_EQ(d.total_records(), 0u);
    EXPECT_EQ(d.impulse_events(), 0u);
    EXPECT_DOUBLE_EQ(d.max_z(), 0.0);
}

TEST(BiasImpulseDetector, ToStatsJson) {
    BiasImpulseDetector d;
    for (int i = 0; i < 15; ++i) d.record(0.05);
    auto json = d.to_stats_json();
    EXPECT_NE(json.find("last_z"), std::string::npos);
    EXPECT_NE(json.find("impulse_events"), std::string::npos);
}

} // namespace llmquant::tests
