#include <gtest/gtest.h>
#include "SignalTrendStrengthIndex.h"
#include <cmath>

namespace llmquant::tests {

TEST(SignalTrendStrengthIndex, DefaultConstruction) {
    SignalTrendStrengthIndex tsi;
    EXPECT_EQ(tsi.total_records(), 0u);
    EXPECT_DOUBLE_EQ(tsi.tsi(), 0.0);
}

TEST(SignalTrendStrengthIndex, TSIRangeAfterRecords) {
    SignalTrendStrengthIndex tsi;
    for (int i = 0; i < 50; ++i) tsi.record(static_cast<double>(i) * 0.01);
    double t = tsi.tsi();
    EXPECT_GE(t, -100.0);
    EXPECT_LE(t, 100.0);
}

TEST(SignalTrendStrengthIndex, PositiveTSIForUptend) {
    SignalTrendStrengthIndex::Config cfg;
    cfg.min_samples = 20;
    SignalTrendStrengthIndex tsi(cfg);
    // Strong consistent uptrend
    for (int i = 0; i < 50; ++i) tsi.record(static_cast<double>(i) * 0.1);
    EXPECT_GT(tsi.tsi(), 0.0);
}

TEST(SignalTrendStrengthIndex, NegativeTSIForDowntrend) {
    SignalTrendStrengthIndex::Config cfg;
    cfg.min_samples = 20;
    SignalTrendStrengthIndex tsi(cfg);
    for (int i = 0; i < 50; ++i) tsi.record(-static_cast<double>(i) * 0.1);
    EXPECT_LT(tsi.tsi(), 0.0);
}

TEST(SignalTrendStrengthIndex, ZeroCrossCallback) {
    SignalTrendStrengthIndex::Config cfg;
    cfg.min_samples = 20;
    int crosses = 0;
    cfg.on_zero_cross = [&](double, double) { ++crosses; };
    SignalTrendStrengthIndex tsi(cfg);
    for (int i = 0; i < 30; ++i) tsi.record(0.1);   // uptrend → positive TSI
    for (int i = 0; i < 30; ++i) tsi.record(-0.1);  // downtrend → negative TSI
    EXPECT_GT(crosses, 0);
}

TEST(SignalTrendStrengthIndex, StrongEventCallback) {
    SignalTrendStrengthIndex::Config cfg;
    cfg.strength_threshold = 10.0;
    cfg.min_samples = 20;
    int strong = 0;
    cfg.on_strong = [&](double) { ++strong; };
    SignalTrendStrengthIndex tsi(cfg);
    for (int i = 0; i < 50; ++i) tsi.record(static_cast<double>(i) * 0.5);
    EXPECT_GT(strong, 0);
}

TEST(SignalTrendStrengthIndex, Reset) {
    SignalTrendStrengthIndex tsi;
    for (int i = 0; i < 40; ++i) tsi.record(0.1);
    tsi.reset();
    EXPECT_EQ(tsi.total_records(), 0u);
    EXPECT_DOUBLE_EQ(tsi.tsi(), 0.0);
}

TEST(SignalTrendStrengthIndex, ToStatsJson) {
    SignalTrendStrengthIndex tsi;
    for (int i = 0; i < 25; ++i) tsi.record(0.05);
    auto json = tsi.to_stats_json();
    EXPECT_NE(json.find("tsi"), std::string::npos);
    EXPECT_NE(json.find("zero_cross_events"), std::string::npos);
}

} // namespace llmquant::tests
