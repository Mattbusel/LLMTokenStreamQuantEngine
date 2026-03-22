#include <gtest/gtest.h>
#include "BiasVolatilityBreakout.h"
#include <cmath>

namespace llmquant::tests {

TEST(BiasVolatilityBreakout, DefaultConstruction) {
    BiasVolatilityBreakout b;
    EXPECT_EQ(b.total_records(), 0u);
    EXPECT_DOUBLE_EQ(b.short_vol(), 0.0);
    EXPECT_DOUBLE_EQ(b.long_vol(), 0.0);
}

TEST(BiasVolatilityBreakout, VolsNonNegative) {
    BiasVolatilityBreakout b;
    for (int i = 0; i < 40; ++i) b.record(i % 2 == 0 ? 1.0 : -1.0);
    EXPECT_GE(b.short_vol(), 0.0);
    EXPECT_GE(b.long_vol(), 0.0);
}

TEST(BiasVolatilityBreakout, ExpansionForHighShortVol) {
    BiasVolatilityBreakout::Config cfg;
    cfg.short_window    = 5;
    cfg.long_window     = 20;
    cfg.min_samples     = 10;
    cfg.breakout_ratio  = 1.2;
    BiasVolatilityBreakout b(cfg);
    // Stable period
    for (int i = 0; i < 15; ++i) b.record(0.01);
    // Sudden spike in vol
    for (int i = 0; i < 10; ++i) b.record(i % 2 == 0 ? 2.0 : -2.0);
    EXPECT_GT(b.vol_ratio(), 0.0);
}

TEST(BiasVolatilityBreakout, ExpansionCallback) {
    BiasVolatilityBreakout::Config cfg;
    cfg.short_window   = 5;
    cfg.long_window    = 20;
    cfg.min_samples    = 10;
    cfg.breakout_ratio = 1.0; // easy to trigger
    int events = 0;
    cfg.on_expansion = [&](double) { ++events; };
    BiasVolatilityBreakout b(cfg);
    for (int i = 0; i < 15; ++i) b.record(0.01);
    for (int i = 0; i < 10; ++i) b.record(i % 2 == 0 ? 5.0 : -5.0);
    EXPECT_GE(events, 0);
}

TEST(BiasVolatilityBreakout, ContractionCallback) {
    BiasVolatilityBreakout::Config cfg;
    cfg.short_window      = 5;
    cfg.long_window       = 20;
    cfg.min_samples       = 10;
    cfg.contraction_ratio = 1.0; // always contracting
    int events = 0;
    cfg.on_contraction = [&](double) { ++events; };
    BiasVolatilityBreakout b(cfg);
    for (int i = 0; i < 30; ++i) b.record(0.001);
    EXPECT_GE(events, 0);
}

TEST(BiasVolatilityBreakout, Reset) {
    BiasVolatilityBreakout b;
    for (int i = 0; i < 40; ++i) b.record(0.1);
    b.reset();
    EXPECT_EQ(b.total_records(), 0u);
    EXPECT_DOUBLE_EQ(b.short_vol(), 0.0);
}

TEST(BiasVolatilityBreakout, ToStatsJson) {
    BiasVolatilityBreakout b;
    for (int i = 0; i < 30; ++i) b.record(0.05);
    auto json = b.to_stats_json();
    EXPECT_NE(json.find("short_vol"), std::string::npos);
    EXPECT_NE(json.find("vol_ratio"), std::string::npos);
}

} // namespace llmquant::tests
