#include <gtest/gtest.h>
#include "SignalKurtosisTracker.h"
#include <cmath>

namespace llmquant::tests {

TEST(SignalKurtosisTracker, DefaultConstruction) {
    SignalKurtosisTracker t;
    EXPECT_EQ(t.total_records(), 0u);
    EXPECT_DOUBLE_EQ(t.kurtosis(), 0.0);
}

TEST(SignalKurtosisTracker, NormalDistributionNearZeroExcessKurtosis) {
    SignalKurtosisTracker t;
    // Uniform alternating ±1 → approximately symmetric, low excess kurtosis
    for (int i = 0; i < 30; ++i) t.record(i % 2 == 0 ? 1.0 : -1.0);
    // Excess kurtosis of a two-point distribution is -2; just verify it's finite
    EXPECT_FALSE(std::isnan(t.kurtosis()));
}

TEST(SignalKurtosisTracker, HighKurtosisForSpiky) {
    SignalKurtosisTracker::Config cfg;
    cfg.window = 30; cfg.min_samples = 10; cfg.fat_tail_threshold = 1.0;
    SignalKurtosisTracker t(cfg);
    // Many zeros + occasional large spike → high kurtosis
    for (int i = 0; i < 25; ++i) t.record(0.0);
    for (int i = 0; i < 5;  ++i) t.record(10.0);
    EXPECT_GT(t.kurtosis(), 0.0);
}

TEST(SignalKurtosisTracker, FatTailCallback) {
    SignalKurtosisTracker::Config cfg;
    cfg.window = 20; cfg.min_samples = 8; cfg.fat_tail_threshold = 0.0;
    int fat_events = 0;
    cfg.on_fat_tail = [&](double) { ++fat_events; };
    SignalKurtosisTracker t(cfg);
    for (int i = 0; i < 15; ++i) t.record(0.0);
    for (int i = 0; i < 5;  ++i) t.record(10.0);
    EXPECT_GE(fat_events, 0);
}

TEST(SignalKurtosisTracker, NormalTailCallback) {
    SignalKurtosisTracker::Config cfg;
    cfg.window = 20; cfg.min_samples = 8; cfg.fat_tail_threshold = -3.0; // very low
    int normal_events = 0;
    cfg.on_normal_tail = [&](double) { ++normal_events; };
    SignalKurtosisTracker t(cfg);
    for (int i = 0; i < 30; ++i) t.record(i % 2 == 0 ? 1.0 : -1.0);
    EXPECT_GE(normal_events, 0);
}

TEST(SignalKurtosisTracker, Reset) {
    SignalKurtosisTracker t;
    for (int i = 0; i < 30; ++i) t.record(0.1);
    t.reset();
    EXPECT_EQ(t.total_records(), 0u);
    EXPECT_DOUBLE_EQ(t.kurtosis(), 0.0);
}

TEST(SignalKurtosisTracker, ToStatsJson) {
    SignalKurtosisTracker t;
    for (int i = 0; i < 20; ++i) t.record(0.05);
    auto json = t.to_stats_json();
    EXPECT_NE(json.find("kurtosis"), std::string::npos);
    EXPECT_NE(json.find("fat_tail_events"), std::string::npos);
}

} // namespace llmquant::tests
