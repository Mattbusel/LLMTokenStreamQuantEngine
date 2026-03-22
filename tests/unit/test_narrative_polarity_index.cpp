#include <gtest/gtest.h>
#include "NarrativePolarityIndex.h"
#include <cmath>

namespace llmquant::tests {

TEST(NarrativePolarityIndex, DefaultConstruction) {
    NarrativePolarityIndex npi;
    EXPECT_DOUBLE_EQ(npi.npi(), 0.0);
    EXPECT_EQ(npi.total_records(), 0u);
}

TEST(NarrativePolarityIndex, PurelyBullishApproachesPositiveOne) {
    NarrativePolarityIndex npi;
    for (int i = 0; i < 200; ++i) npi.record(0.5);
    EXPECT_GT(npi.npi(), 0.9);
}

TEST(NarrativePolarityIndex, PurelyBearishApproachesNegativeOne) {
    NarrativePolarityIndex npi;
    for (int i = 0; i < 200; ++i) npi.record(-0.5);
    EXPECT_LT(npi.npi(), -0.9);
}

TEST(NarrativePolarityIndex, BalancedApproachesZero) {
    NarrativePolarityIndex npi;
    for (int i = 0; i < 200; ++i) npi.record((i % 2 == 0) ? 0.3 : -0.3);
    EXPECT_NEAR(npi.npi(), 0.0, 0.1);
}

TEST(NarrativePolarityIndex, NPIBoundsMinusOneToOne) {
    NarrativePolarityIndex npi;
    for (int i = 0; i < 50; ++i) npi.record(10.0);
    EXPECT_LE(npi.npi(), 1.0);
    EXPECT_GE(npi.npi(), -1.0);
}

TEST(NarrativePolarityIndex, PolarityFlipCallback) {
    NarrativePolarityIndex::Config cfg;
    cfg.min_samples = 3;
    int flips = 0;
    cfg.on_polarity_flip = [&](double, double) { ++flips; };
    NarrativePolarityIndex npi(cfg);
    for (int i = 0; i < 100; ++i) npi.record(0.5);  // bullish
    for (int i = 0; i < 100; ++i) npi.record(-0.5); // bearish
    EXPECT_GT(flips, 0);
}

TEST(NarrativePolarityIndex, ExtremeCallback) {
    NarrativePolarityIndex::Config cfg;
    cfg.extreme_threshold = 0.5;
    cfg.min_samples = 1;
    int extremes = 0;
    cfg.on_extreme = [&](double) { ++extremes; };
    NarrativePolarityIndex npi(cfg);
    for (int i = 0; i < 100; ++i) npi.record(1.0);
    EXPECT_GT(extremes, 0);
}

TEST(NarrativePolarityIndex, Reset) {
    NarrativePolarityIndex npi;
    for (int i = 0; i < 50; ++i) npi.record(0.3);
    npi.reset();
    EXPECT_DOUBLE_EQ(npi.npi(), 0.0);
    EXPECT_EQ(npi.total_records(), 0u);
}

TEST(NarrativePolarityIndex, ToStatsJson) {
    NarrativePolarityIndex npi;
    for (int i = 0; i < 10; ++i) npi.record(0.1);
    auto json = npi.to_stats_json();
    EXPECT_NE(json.find("npi"), std::string::npos);
    EXPECT_NE(json.find("bull_ema"), std::string::npos);
}

} // namespace llmquant::tests
