#include <gtest/gtest.h>
#include "SignalKeltnerChannel.h"

using namespace llmquant;

TEST(SignalKeltnerChannel, DefaultConstruction) {
    SignalKeltnerChannel k(SignalKeltnerChannel::Config{});
    EXPECT_EQ(k.total_records(), 0u);
    EXPECT_DOUBLE_EQ(k.ema(), 0.0);
    EXPECT_DOUBLE_EQ(k.atr(), 0.0);
}

TEST(SignalKeltnerChannel, UpperGeqLower) {
    SignalKeltnerChannel::Config cfg;
    cfg.min_samples = 3;
    SignalKeltnerChannel k(cfg);
    for (int i = 0; i < 10; ++i) k.record(0.01 * i);
    EXPECT_GE(k.upper_band(), k.lower_band());
}

TEST(SignalKeltnerChannel, EmaPositiveForPositiveBias) {
    SignalKeltnerChannel::Config cfg;
    cfg.alpha_ema = 0.5;
    SignalKeltnerChannel k(cfg);
    for (int i = 0; i < 10; ++i) k.record(0.1);
    EXPECT_GT(k.ema(), 0.0);
}

TEST(SignalKeltnerChannel, UpperBreakCallback) {
    SignalKeltnerChannel::Config cfg;
    cfg.alpha_ema = 0.1;
    cfg.alpha_atr = 0.1;
    cfg.multiplier = 1.0;
    cfg.min_samples = 3;
    int ub_count = 0;
    cfg.on_upper_break = [&](double, double) { ++ub_count; };
    SignalKeltnerChannel k(cfg);
    for (int i = 0; i < 5; ++i) k.record(0.0);
    k.record(100.0); // extreme breakout
    EXPECT_GE(ub_count, 1);
}

TEST(SignalKeltnerChannel, LowerBreakCallback) {
    SignalKeltnerChannel::Config cfg;
    cfg.alpha_ema = 0.1;
    cfg.alpha_atr = 0.1;
    cfg.multiplier = 1.0;
    cfg.min_samples = 3;
    int lb_count = 0;
    cfg.on_lower_break = [&](double, double) { ++lb_count; };
    SignalKeltnerChannel k(cfg);
    for (int i = 0; i < 5; ++i) k.record(0.0);
    k.record(-100.0);
    EXPECT_GE(lb_count, 1);
}

TEST(SignalKeltnerChannel, Reset) {
    SignalKeltnerChannel k(SignalKeltnerChannel::Config{});
    for (int i = 0; i < 10; ++i) k.record(0.05);
    k.reset();
    EXPECT_EQ(k.total_records(), 0u);
    EXPECT_DOUBLE_EQ(k.ema(), 0.0);
    EXPECT_DOUBLE_EQ(k.atr(), 0.0);
    EXPECT_EQ(k.upper_breaks(), 0u);
    EXPECT_EQ(k.lower_breaks(), 0u);
}

TEST(SignalKeltnerChannel, ToStatsJson) {
    SignalKeltnerChannel k(SignalKeltnerChannel::Config{});
    for (int i = 0; i < 5; ++i) k.record(0.05);
    std::string j = k.to_stats_json();
    EXPECT_NE(j.find("ema"), std::string::npos);
    EXPECT_NE(j.find("atr"), std::string::npos);
    EXPECT_NE(j.find("upper_band"), std::string::npos);
    EXPECT_NE(j.find("total_records"), std::string::npos);
}
