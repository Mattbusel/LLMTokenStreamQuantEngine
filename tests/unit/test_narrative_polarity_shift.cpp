#include <gtest/gtest.h>
#include "NarrativePolarityShift.h"

using namespace llmquant;

TEST(NarrativePolarityShift, DefaultConstruction) {
    NarrativePolarityShift p(NarrativePolarityShift::Config{});
    EXPECT_EQ(p.total_records(), 0u);
    EXPECT_NEAR(p.positive_fraction() + p.negative_fraction(), 1.0, 1e-9);
}

TEST(NarrativePolarityShift, AllPositiveBulls) {
    NarrativePolarityShift::Config cfg;
    cfg.window = 10;
    cfg.min_samples = 5;
    cfg.bull_threshold = 0.6;
    NarrativePolarityShift p(cfg);
    for (int i = 0; i < 10; ++i) p.record(0.1);
    EXPECT_NEAR(p.positive_fraction(), 1.0, 1e-6);
    EXPECT_TRUE(p.is_bullish());
    EXPECT_FALSE(p.is_bearish());
}

TEST(NarrativePolarityShift, AllNegativeBears) {
    NarrativePolarityShift::Config cfg;
    cfg.window = 10;
    cfg.min_samples = 5;
    cfg.bear_threshold = 0.4;
    NarrativePolarityShift p(cfg);
    for (int i = 0; i < 10; ++i) p.record(-0.1);
    EXPECT_NEAR(p.positive_fraction(), 0.0, 1e-6);
    EXPECT_TRUE(p.is_bearish());
    EXPECT_FALSE(p.is_bullish());
}

TEST(NarrativePolarityShift, BullishCallback) {
    NarrativePolarityShift::Config cfg;
    cfg.window = 5;
    cfg.min_samples = 3;
    cfg.bull_threshold = 0.6;
    int bull_count = 0;
    cfg.on_bullish = [&](double) { ++bull_count; };
    NarrativePolarityShift p(cfg);
    for (int i = 0; i < 5; ++i) p.record(0.1);
    EXPECT_GE(bull_count, 1);
}

TEST(NarrativePolarityShift, BearishCallback) {
    NarrativePolarityShift::Config cfg;
    cfg.window = 5;
    cfg.min_samples = 3;
    cfg.bear_threshold = 0.4;
    int bear_count = 0;
    cfg.on_bearish = [&](double) { ++bear_count; };
    NarrativePolarityShift p(cfg);
    for (int i = 0; i < 5; ++i) p.record(-0.1);
    EXPECT_GE(bear_count, 1);
}

TEST(NarrativePolarityShift, Reset) {
    NarrativePolarityShift p(NarrativePolarityShift::Config{});
    for (int i = 0; i < 10; ++i) p.record(0.1);
    p.reset();
    EXPECT_EQ(p.total_records(), 0u);
    EXPECT_EQ(p.bullish_events(), 0u);
    EXPECT_EQ(p.bearish_events(), 0u);
}

TEST(NarrativePolarityShift, ToStatsJson) {
    NarrativePolarityShift p(NarrativePolarityShift::Config{});
    for (int i = 0; i < 5; ++i) p.record(0.05);
    std::string j = p.to_stats_json();
    EXPECT_NE(j.find("positive_fraction"), std::string::npos);
    EXPECT_NE(j.find("total_records"), std::string::npos);
}
