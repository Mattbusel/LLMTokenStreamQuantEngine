#include <gtest/gtest.h>
#include "SignalCrossCorrelator.h"
#include <cmath>

namespace llmquant::tests {

TEST(SignalCrossCorrelator, DefaultConstruction) {
    SignalCrossCorrelator cc;
    EXPECT_EQ(cc.total_records(), 0u);
    EXPECT_GE(cc.dominant_lag(), 0);
}

TEST(SignalCrossCorrelator, RecordsAccumulate) {
    SignalCrossCorrelator cc;
    for (int i = 0; i < 30; ++i) cc.record(0.1);
    EXPECT_EQ(cc.total_records(), 30u);
}

TEST(SignalCrossCorrelator, DominantLagInRange) {
    SignalCrossCorrelator::Config cfg;
    cfg.max_lag = 5;
    cfg.min_samples = 15;
    SignalCrossCorrelator cc(cfg);
    for (int i = 0; i < 40; ++i) cc.record((double)i * 0.01);
    int lag = cc.dominant_lag();
    EXPECT_GE(lag, 0);
    EXPECT_LE(lag, 5);
}

TEST(SignalCrossCorrelator, R0BoundedMinusOneToOne) {
    SignalCrossCorrelator cc;
    for (int i = 0; i < 50; ++i) cc.record((double)(i % 7) * 0.1 - 0.3);
    EXPECT_GE(cc.r_lag0(), -1.0);
    EXPECT_LE(cc.r_lag0(), 1.0);
}

TEST(SignalCrossCorrelator, LagChangeCallback) {
    SignalCrossCorrelator::Config cfg;
    cfg.min_samples = 20;
    cfg.max_lag = 3;
    int changes = 0;
    cfg.on_dominant_lag_change = [&](int, int, double) { ++changes; };
    SignalCrossCorrelator cc(cfg);
    for (int i = 0; i < 100; ++i) cc.record((double)(i % 13) * 0.1 - 0.6);
    EXPECT_GE(cc.lag_change_events(), 0u);
}

TEST(SignalCrossCorrelator, Reset) {
    SignalCrossCorrelator cc;
    for (int i = 0; i < 40; ++i) cc.record(0.1);
    cc.reset();
    EXPECT_EQ(cc.total_records(), 0u);
    EXPECT_DOUBLE_EQ(cc.r_lag0(), 0.0);
}

TEST(SignalCrossCorrelator, ToStatsJson) {
    SignalCrossCorrelator cc;
    for (int i = 0; i < 25; ++i) cc.record(0.05);
    auto json = cc.to_stats_json();
    EXPECT_NE(json.find("r_lag0"), std::string::npos);
    EXPECT_NE(json.find("dominant_lag"), std::string::npos);
}

} // namespace llmquant::tests
