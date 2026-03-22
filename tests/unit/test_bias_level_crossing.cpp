#include <gtest/gtest.h>
#include "BiasLevelCrossing.h"
#include <cmath>

namespace llmquant::tests {

TEST(BiasLevelCrossing, DefaultConstruction) {
    BiasLevelCrossing lc;
    EXPECT_EQ(lc.total_records(), 0u);
    EXPECT_DOUBLE_EQ(lc.zero_crossing_rate(), 0.0);
}

TEST(BiasLevelCrossing, HighZCRForAlternatingSignal) {
    BiasLevelCrossing::Config cfg;
    cfg.window      = 20;
    cfg.min_samples = 5;
    BiasLevelCrossing lc(cfg);
    // Alternating +/- → every step crosses zero
    for (int i = 0; i < 30; ++i) lc.record((i % 2 == 0) ? 0.5 : -0.5);
    // ZCR should be close to 0.5 (crosses every 2 samples → 0.5 crossings per sample)
    EXPECT_GT(lc.zero_crossing_rate(), 0.3);
}

TEST(BiasLevelCrossing, LowZCRForConstantSignal) {
    BiasLevelCrossing::Config cfg;
    cfg.window      = 20;
    cfg.min_samples = 5;
    BiasLevelCrossing lc(cfg);
    for (int i = 0; i < 30; ++i) lc.record(0.3);  // constant positive
    EXPECT_LT(lc.zero_crossing_rate(), 0.1);
}

TEST(BiasLevelCrossing, ZCRChangeCallback) {
    BiasLevelCrossing::Config cfg;
    cfg.window = 20;
    cfg.min_samples = 5;
    cfg.zcr_change_threshold = 0.1;
    int events = 0;
    cfg.on_zcr_change = [&](double, double) { ++events; };
    BiasLevelCrossing lc(cfg);
    for (int i = 0; i < 20; ++i) lc.record(0.3);          // low ZCR
    for (int i = 0; i < 30; ++i) lc.record((i%2==0)?0.5:-0.5); // high ZCR
    EXPECT_GT(events, 0);
}

TEST(BiasLevelCrossing, Reset) {
    BiasLevelCrossing lc;
    for (int i = 0; i < 20; ++i) lc.record(0.1);
    lc.reset();
    EXPECT_EQ(lc.total_records(), 0u);
    EXPECT_DOUBLE_EQ(lc.zero_crossing_rate(), 0.0);
}

TEST(BiasLevelCrossing, ToStatsJson) {
    BiasLevelCrossing lc;
    for (int i = 0; i < 15; ++i) lc.record(0.1);
    auto json = lc.to_stats_json();
    EXPECT_NE(json.find("zero_crossing_rate"), std::string::npos);
    EXPECT_NE(json.find("total_records"), std::string::npos);
}

} // namespace llmquant::tests
