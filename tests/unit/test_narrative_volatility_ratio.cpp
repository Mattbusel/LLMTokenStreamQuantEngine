#include <gtest/gtest.h>
#include "NarrativeVolatilityRatio.h"
#include <cmath>

namespace llmquant::tests {

TEST(NarrativeVolatilityRatio, DefaultConstruction) {
    NarrativeVolatilityRatio vr;
    EXPECT_EQ(vr.total_records(), 0u);
    EXPECT_GE(vr.vol_ratio(), 0.0);
}

TEST(NarrativeVolatilityRatio, VolRatioPositive) {
    NarrativeVolatilityRatio vr;
    for (int i = 0; i < 50; ++i) vr.record((double)(i%5-2) * 0.1);
    EXPECT_GT(vr.vol_ratio(), 0.0);
    EXPECT_GT(vr.short_vol(), 0.0);
    EXPECT_GT(vr.long_vol(), 0.0);
}

TEST(NarrativeVolatilityRatio, HighVolSpike_RatioAboveOne) {
    NarrativeVolatilityRatio::Config cfg;
    cfg.alpha_short = 0.5;  // very fast EMA
    cfg.alpha_long  = 0.01; // very slow EMA
    NarrativeVolatilityRatio vr(cfg);
    // Start with low vol
    for (int i = 0; i < 50; ++i) vr.record(0.01);
    // Spike in vol → short-term vol > long-term vol
    for (int i = 0; i < 5; ++i) vr.record(5.0);
    EXPECT_GT(vr.vol_ratio(), 1.0);
}

TEST(NarrativeVolatilityRatio, RegimeChangeCallback) {
    NarrativeVolatilityRatio::Config cfg;
    cfg.change_threshold = 0.1;
    cfg.min_samples = 5;
    int changes = 0;
    cfg.on_vol_regime_change = [&](double, double) { ++changes; };
    NarrativeVolatilityRatio vr(cfg);
    for (int i = 0; i < 20; ++i) vr.record(0.01);
    for (int i = 0; i < 20; ++i) vr.record(5.0);
    EXPECT_GT(changes, 0);
}

TEST(NarrativeVolatilityRatio, Reset) {
    NarrativeVolatilityRatio vr;
    for (int i = 0; i < 30; ++i) vr.record(0.2);
    vr.reset();
    EXPECT_EQ(vr.total_records(), 0u);
    EXPECT_NEAR(vr.vol_ratio(), 1.0, 1e-9);
}

TEST(NarrativeVolatilityRatio, ToStatsJson) {
    NarrativeVolatilityRatio vr;
    for (int i = 0; i < 15; ++i) vr.record(0.1);
    auto json = vr.to_stats_json();
    EXPECT_NE(json.find("vol_ratio"), std::string::npos);
    EXPECT_NE(json.find("short_vol"), std::string::npos);
}

} // namespace llmquant::tests
