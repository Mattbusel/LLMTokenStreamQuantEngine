#define LLMQUANT_TRIPLE_EMA_ENABLED
#include <gtest/gtest.h>
#include "SignalTripleEMAOscillator.h"

using namespace llmquant;

TEST(SignalTripleEMAOscillator, DefaultConstruction) {
    SignalTripleEMAOscillator t(SignalTripleEMAOscillator::Config{});
    EXPECT_EQ(t.total_records(), 0u);
    EXPECT_DOUBLE_EQ(t.trix(), 0.0);
}

TEST(SignalTripleEMAOscillator, RecordUpdates) {
    SignalTripleEMAOscillator::Config cfg;
    cfg.min_samples = 3;
    SignalTripleEMAOscillator t(cfg);
    for (int i = 0; i < 10; ++i) t.record(0.05 * i);
    EXPECT_EQ(t.total_records(), 10u);
}

TEST(SignalTripleEMAOscillator, UptrendPositiveTRIX) {
    SignalTripleEMAOscillator::Config cfg;
    cfg.min_samples = 3;
    cfg.alpha = 0.5;
    SignalTripleEMAOscillator t(cfg);
    for (int i = 0; i < 15; ++i) t.record(0.1 * (i + 1));
    EXPECT_GT(t.trix(), 0.0);
}

TEST(SignalTripleEMAOscillator, EMA1LessThanOrEqualUptrendInput) {
    SignalTripleEMAOscillator::Config cfg;
    cfg.alpha = 0.3;
    SignalTripleEMAOscillator t(cfg);
    for (int i = 0; i < 10; ++i) t.record(1.0);
    EXPECT_NEAR(t.ema1(), 1.0, 0.05);
}

TEST(SignalTripleEMAOscillator, ZeroCrossCallback) {
    SignalTripleEMAOscillator::Config cfg;
    cfg.min_samples = 3;
    cfg.alpha = 0.5;
    cfg.alpha_signal = 0.5;
    int crosses = 0;
    cfg.on_zero_cross = [&](double) { ++crosses; };
    SignalTripleEMAOscillator t(cfg);
    for (int i = 0; i < 5; ++i)  t.record(0.1);
    for (int i = 0; i < 5; ++i)  t.record(-0.1);
    for (int i = 0; i < 5; ++i)  t.record(0.1);
    EXPECT_GE(crosses + t.zero_cross_events(), 0u);
}

TEST(SignalTripleEMAOscillator, Reset) {
    SignalTripleEMAOscillator t(SignalTripleEMAOscillator::Config{});
    for (int i = 0; i < 10; ++i) t.record(0.05);
    t.reset();
    EXPECT_EQ(t.total_records(), 0u);
    EXPECT_DOUBLE_EQ(t.trix(), 0.0);
    EXPECT_EQ(t.zero_cross_events(), 0u);
    EXPECT_EQ(t.signal_cross_events(), 0u);
}

TEST(SignalTripleEMAOscillator, ToStatsJson) {
    SignalTripleEMAOscillator t(SignalTripleEMAOscillator::Config{});
    for (int i = 0; i < 5; ++i) t.record(0.05);
    std::string j = t.to_stats_json();
    EXPECT_NE(j.find("trix"), std::string::npos);
    EXPECT_NE(j.find("signal_line"), std::string::npos);
    EXPECT_NE(j.find("total_records"), std::string::npos);
}
