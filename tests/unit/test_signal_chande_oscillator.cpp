#include <gtest/gtest.h>
#include "SignalChandeOscillator.h"

using namespace llmquant;

TEST(SignalChandeOscillator, DefaultConstruction) {
    SignalChandeOscillator c(SignalChandeOscillator::Config{});
    EXPECT_EQ(c.total_records(), 0u);
    EXPECT_DOUBLE_EQ(c.value(), 0.0);
}

TEST(SignalChandeOscillator, ValueInRange) {
    SignalChandeOscillator::Config cfg;
    cfg.window = 5;
    cfg.min_samples = 3;
    SignalChandeOscillator c(cfg);
    for (int i = 0; i < 10; ++i) c.record(0.01 * i);
    double v = c.value();
    EXPECT_GE(v, -100.0);
    EXPECT_LE(v, 100.0);
}

TEST(SignalChandeOscillator, UptrendPositiveCMO) {
    SignalChandeOscillator::Config cfg;
    cfg.window = 5;
    cfg.min_samples = 3;
    SignalChandeOscillator c(cfg);
    for (int i = 0; i < 6; ++i) c.record(0.1 * (i + 1));
    EXPECT_GT(c.value(), 0.0);
}

TEST(SignalChandeOscillator, OverboughtCallback) {
    SignalChandeOscillator::Config cfg;
    cfg.window = 5;
    cfg.min_samples = 3;
    cfg.overbought_threshold = 20.0;
    int ob_count = 0;
    cfg.on_overbought = [&](double) { ++ob_count; };
    SignalChandeOscillator c(cfg);
    for (int i = 0; i < 8; ++i) c.record(0.1 * (i + 1));
    EXPECT_GE(ob_count, 1);
}

TEST(SignalChandeOscillator, OversoldCallback) {
    SignalChandeOscillator::Config cfg;
    cfg.window = 5;
    cfg.min_samples = 3;
    cfg.oversold_threshold = -20.0;
    int os_count = 0;
    cfg.on_oversold = [&](double) { ++os_count; };
    SignalChandeOscillator c(cfg);
    for (int i = 7; i >= 0; --i) c.record(-0.1 * (i + 1));
    EXPECT_GE(os_count, 1);
}

TEST(SignalChandeOscillator, Reset) {
    SignalChandeOscillator c(SignalChandeOscillator::Config{});
    for (int i = 0; i < 10; ++i) c.record(0.05);
    c.reset();
    EXPECT_EQ(c.total_records(), 0u);
    EXPECT_DOUBLE_EQ(c.value(), 0.0);
    EXPECT_EQ(c.overbought_events(), 0u);
    EXPECT_EQ(c.oversold_events(), 0u);
}

TEST(SignalChandeOscillator, ToStatsJson) {
    SignalChandeOscillator c(SignalChandeOscillator::Config{});
    for (int i = 0; i < 5; ++i) c.record(0.05);
    std::string j = c.to_stats_json();
    EXPECT_NE(j.find("cmo"), std::string::npos);
    EXPECT_NE(j.find("total_records"), std::string::npos);
}

TEST(SignalChandeOscillator, FlatSeriesZeroCMO) {
    SignalChandeOscillator::Config cfg;
    cfg.window = 5;
    cfg.min_samples = 3;
    SignalChandeOscillator c(cfg);
    for (int i = 0; i < 6; ++i) c.record(0.05);
    // Flat series: no up/down moves after first → CMO = 0
    EXPECT_DOUBLE_EQ(c.value(), 0.0);
}
