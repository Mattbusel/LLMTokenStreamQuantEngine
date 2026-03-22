#include <gtest/gtest.h>
#include "NarrativePressureGauge.h"

using namespace llmquant;

TEST(NarrativePressureGauge, DefaultConstruction) {
    NarrativePressureGauge g(NarrativePressureGauge::Config{});
    EXPECT_EQ(g.total_records(), 0u);
    EXPECT_DOUBLE_EQ(g.pressure(), 0.0);
}

TEST(NarrativePressureGauge, PositivePressureForUptrend) {
    NarrativePressureGauge::Config cfg;
    cfg.alpha_fast = 0.5;
    cfg.alpha_slow = 0.05;
    NarrativePressureGauge g(cfg);
    for (int i = 0; i < 20; ++i) g.record(1.0);
    EXPECT_GT(g.pressure(), 0.0);
    EXPECT_GT(g.fast_ema(), g.slow_ema() - 1e-6);
}

TEST(NarrativePressureGauge, NegativePressureForDowntrend) {
    NarrativePressureGauge::Config cfg;
    cfg.alpha_fast = 0.5;
    cfg.alpha_slow = 0.05;
    NarrativePressureGauge g(cfg);
    for (int i = 0; i < 20; ++i) g.record(-1.0);
    EXPECT_LT(g.pressure(), 0.0);
}

TEST(NarrativePressureGauge, PositivePressureCallback) {
    NarrativePressureGauge::Config cfg;
    cfg.alpha_fast = 0.5;
    cfg.alpha_slow = 0.1;
    cfg.pressure_threshold = 0.001;
    cfg.min_samples = 3;
    int pos_count = 0;
    cfg.on_positive_pressure = [&](double) { ++pos_count; };
    NarrativePressureGauge g(cfg);
    for (int i = 0; i < 10; ++i) g.record(0.5);
    EXPECT_GE(pos_count, 1);
}

TEST(NarrativePressureGauge, NegativePressureCallback) {
    NarrativePressureGauge::Config cfg;
    cfg.alpha_fast = 0.5;
    cfg.alpha_slow = 0.1;
    cfg.pressure_threshold = 0.001;
    cfg.min_samples = 3;
    int neg_count = 0;
    cfg.on_negative_pressure = [&](double) { ++neg_count; };
    NarrativePressureGauge g(cfg);
    for (int i = 0; i < 10; ++i) g.record(-0.5);
    EXPECT_GE(neg_count, 1);
}

TEST(NarrativePressureGauge, PressureIsSpread) {
    NarrativePressureGauge::Config cfg;
    NarrativePressureGauge g(cfg);
    for (int i = 0; i < 10; ++i) g.record(0.1);
    EXPECT_NEAR(g.pressure(), g.fast_ema() - g.slow_ema(), 1e-9);
}

TEST(NarrativePressureGauge, Reset) {
    NarrativePressureGauge g(NarrativePressureGauge::Config{});
    for (int i = 0; i < 5; ++i) g.record(0.1);
    g.reset();
    EXPECT_EQ(g.total_records(), 0u);
    EXPECT_DOUBLE_EQ(g.pressure(), 0.0);
    EXPECT_EQ(g.positive_pressure_events(), 0u);
    EXPECT_EQ(g.negative_pressure_events(), 0u);
}

TEST(NarrativePressureGauge, ToStatsJson) {
    NarrativePressureGauge g(NarrativePressureGauge::Config{});
    for (int i = 0; i < 5; ++i) g.record(0.05);
    std::string j = g.to_stats_json();
    EXPECT_NE(j.find("pressure"), std::string::npos);
    EXPECT_NE(j.find("fast_ema"), std::string::npos);
    EXPECT_NE(j.find("total_records"), std::string::npos);
}
