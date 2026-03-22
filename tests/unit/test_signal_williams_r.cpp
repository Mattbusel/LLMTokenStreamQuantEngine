#include <gtest/gtest.h>
#include "SignalWilliamsR.h"

using namespace llmquant;

TEST(SignalWilliamsR, DefaultConstruction) {
    SignalWilliamsR w(SignalWilliamsR::Config{});
    EXPECT_EQ(w.total_records(), 0u);
    EXPECT_DOUBLE_EQ(w.williams_r(), -50.0);
}

TEST(SignalWilliamsR, WilliamsRInRange) {
    SignalWilliamsR::Config cfg;
    cfg.period = 5;
    cfg.min_samples = 3;
    SignalWilliamsR w(cfg);
    for (int i = 0; i < 10; ++i) w.record(0.01 * i);
    double wr = w.williams_r();
    EXPECT_GE(wr, -100.0);
    EXPECT_LE(wr, 0.0);
}

TEST(SignalWilliamsR, AtHighIsOverbought) {
    SignalWilliamsR::Config cfg;
    cfg.period = 5;
    cfg.min_samples = 3;
    cfg.overbought_threshold = -20.0;
    SignalWilliamsR w(cfg);
    // Record increasing values so last is near high
    for (int i = 0; i < 5; ++i) w.record(0.01 * i);
    // The last value equals the high → W%R = 0 → overbought
    EXPECT_TRUE(w.is_overbought());
}

TEST(SignalWilliamsR, AtLowIsOversold) {
    SignalWilliamsR::Config cfg;
    cfg.period = 5;
    cfg.min_samples = 3;
    cfg.oversold_threshold = -80.0;
    SignalWilliamsR w(cfg);
    // Record decreasing values so last is at low
    for (int i = 4; i >= 0; --i) w.record(0.01 * i);
    // The last value equals the low → W%R = -100 → oversold
    EXPECT_TRUE(w.is_oversold());
}

TEST(SignalWilliamsR, OverboughtCallback) {
    SignalWilliamsR::Config cfg;
    cfg.period = 5;
    cfg.min_samples = 3;
    cfg.overbought_threshold = -20.0;
    int ob_count = 0;
    cfg.on_overbought = [&](double) { ++ob_count; };
    SignalWilliamsR w(cfg);
    for (int i = 0; i < 5; ++i) w.record(0.01 * i);
    EXPECT_GE(ob_count, 1);
}

TEST(SignalWilliamsR, OversoldCallback) {
    SignalWilliamsR::Config cfg;
    cfg.period = 5;
    cfg.min_samples = 3;
    cfg.oversold_threshold = -80.0;
    int os_count = 0;
    cfg.on_oversold = [&](double) { ++os_count; };
    SignalWilliamsR w(cfg);
    for (int i = 4; i >= 0; --i) w.record(0.01 * i);
    EXPECT_GE(os_count, 1);
}

TEST(SignalWilliamsR, Reset) {
    SignalWilliamsR w(SignalWilliamsR::Config{});
    for (int i = 0; i < 10; ++i) w.record(0.05);
    w.reset();
    EXPECT_EQ(w.total_records(), 0u);
    EXPECT_DOUBLE_EQ(w.williams_r(), -50.0);
    EXPECT_FALSE(w.is_overbought());
    EXPECT_FALSE(w.is_oversold());
    EXPECT_EQ(w.overbought_events(), 0u);
    EXPECT_EQ(w.oversold_events(), 0u);
}

TEST(SignalWilliamsR, ToStatsJson) {
    SignalWilliamsR w(SignalWilliamsR::Config{});
    for (int i = 0; i < 5; ++i) w.record(0.05);
    std::string j = w.to_stats_json();
    EXPECT_NE(j.find("williams_r"), std::string::npos);
    EXPECT_NE(j.find("is_overbought"), std::string::npos);
    EXPECT_NE(j.find("total_records"), std::string::npos);
}
