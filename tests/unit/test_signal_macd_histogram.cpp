#include <gtest/gtest.h>
#include "SignalMACDHistogram.h"

using namespace llmquant;

TEST(SignalMACDHistogram, DefaultConstruction) {
    SignalMACDHistogram m(SignalMACDHistogram::Config{});
    EXPECT_EQ(m.total_records(), 0u);
    EXPECT_DOUBLE_EQ(m.macd(), 0.0);
    EXPECT_DOUBLE_EQ(m.histogram(), 0.0);
}

TEST(SignalMACDHistogram, RecordUpdatesEMAs) {
    SignalMACDHistogram::Config cfg;
    SignalMACDHistogram m(cfg);
    for (int i = 0; i < 30; ++i) m.record(0.05);
    EXPECT_GT(m.fast_ema(), 0.0);
    EXPECT_GT(m.slow_ema(), 0.0);
    EXPECT_EQ(m.total_records(), 30u);
}

TEST(SignalMACDHistogram, HistogramPositiveForUptrend) {
    SignalMACDHistogram::Config cfg;
    SignalMACDHistogram m(cfg);
    // Seed with rising values: fast EMA should exceed slow
    for (int i = 0; i < 30; ++i) m.record(0.1 * (i + 1) / 30.0);
    // After a strong uptrend, macd should be positive
    EXPECT_GT(m.macd(), 0.0);
}

TEST(SignalMACDHistogram, ZeroCrossCallback) {
    SignalMACDHistogram::Config cfg;
    int crosses = 0;
    cfg.on_zero_cross = [&](double) { ++crosses; };
    cfg.min_samples = 2;
    SignalMACDHistogram m(cfg);
    // Alternate positive/negative to force zero crosses
    for (int i = 0; i < 20; ++i) m.record(i % 2 == 0 ? 0.1 : -0.1);
    EXPECT_GE(crosses, 1);
}

TEST(SignalMACDHistogram, DivergenceCallback) {
    SignalMACDHistogram::Config cfg;
    cfg.divergence_threshold = 0.001;
    cfg.min_samples = 2;
    int divs = 0;
    cfg.on_divergence = [&](double) { ++divs; };
    SignalMACDHistogram m(cfg);
    for (int i = 0; i < 30; ++i) m.record(0.1);
    EXPECT_GE(divs, 1);
}

TEST(SignalMACDHistogram, Reset) {
    SignalMACDHistogram::Config cfg;
    SignalMACDHistogram m(cfg);
    for (int i = 0; i < 10; ++i) m.record(0.05);
    m.reset();
    EXPECT_EQ(m.total_records(), 0u);
    EXPECT_DOUBLE_EQ(m.macd(), 0.0);
    EXPECT_DOUBLE_EQ(m.histogram(), 0.0);
    EXPECT_EQ(m.zero_cross_events(), 0u);
    EXPECT_EQ(m.divergence_events(), 0u);
}

TEST(SignalMACDHistogram, ToStatsJson) {
    SignalMACDHistogram m(SignalMACDHistogram::Config{});
    for (int i = 0; i < 5; ++i) m.record(0.02);
    std::string j = m.to_stats_json();
    EXPECT_NE(j.find("macd"), std::string::npos);
    EXPECT_NE(j.find("histogram"), std::string::npos);
    EXPECT_NE(j.find("total_records"), std::string::npos);
}

TEST(SignalMACDHistogram, UpdateConfig) {
    SignalMACDHistogram::Config cfg;
    SignalMACDHistogram m(cfg);
    cfg.alpha_fast = 0.5;
    m.update_config(cfg);
    // Should not crash
    m.record(0.1);
    EXPECT_EQ(m.total_records(), 1u);
}
