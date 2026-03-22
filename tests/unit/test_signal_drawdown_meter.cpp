#include <gtest/gtest.h>
#include "SignalDrawdownMeter.h"
#include <cmath>

namespace llmquant::tests {

TEST(SignalDrawdownMeter, DefaultConstruction) {
    SignalDrawdownMeter m;
    EXPECT_EQ(m.total_records(), 0u);
    EXPECT_DOUBLE_EQ(m.cum_bias(), 0.0);
    EXPECT_DOUBLE_EQ(m.max_drawdown(), 0.0);
}

TEST(SignalDrawdownMeter, CumBiasAccumulates) {
    SignalDrawdownMeter m;
    for (int i = 0; i < 10; ++i) m.record(1.0);
    EXPECT_NEAR(m.cum_bias(), 10.0, 1e-9);
}

TEST(SignalDrawdownMeter, DrawdownAfterDrop) {
    SignalDrawdownMeter m;
    // Build up peak then drop
    for (int i = 0; i < 10; ++i) m.record(1.0);  // peak = 10
    for (int i = 0; i < 5;  ++i) m.record(-1.0); // current = 5
    EXPECT_GT(m.drawdown(), 0.0);
    EXPECT_GT(m.max_drawdown(), 0.0);
}

TEST(SignalDrawdownMeter, MaxDrawdownNonDecreasing) {
    SignalDrawdownMeter m;
    for (int i = 0; i < 5; ++i) m.record(1.0);
    double dd1 = m.max_drawdown();
    m.record(-3.0);
    double dd2 = m.max_drawdown();
    EXPECT_GE(dd2, dd1);
}

TEST(SignalDrawdownMeter, DrawdownCallback) {
    SignalDrawdownMeter::Config cfg;
    cfg.drawdown_threshold = 0.0; // always fires on new max
    int events = 0;
    cfg.on_new_drawdown_high = [&](double) { ++events; };
    SignalDrawdownMeter m(cfg);
    for (int i = 0; i < 5; ++i) m.record(1.0);
    m.record(-3.0);
    EXPECT_GT(events, 0);
}

TEST(SignalDrawdownMeter, RecoveryCallback) {
    SignalDrawdownMeter::Config cfg;
    int recoveries = 0;
    cfg.on_recovery = [&](double) { ++recoveries; };
    SignalDrawdownMeter m(cfg);
    for (int i = 0; i < 5; ++i)  m.record(1.0);  // peak = 5
    for (int i = 0; i < 3; ++i)  m.record(-1.0); // drawdown
    for (int i = 0; i < 10; ++i) m.record(1.0);  // recover past peak
    EXPECT_GT(recoveries, 0);
}

TEST(SignalDrawdownMeter, Reset) {
    SignalDrawdownMeter m;
    for (int i = 0; i < 10; ++i) m.record(1.0);
    m.reset();
    EXPECT_EQ(m.total_records(), 0u);
    EXPECT_DOUBLE_EQ(m.cum_bias(), 0.0);
    EXPECT_DOUBLE_EQ(m.max_drawdown(), 0.0);
}

TEST(SignalDrawdownMeter, ToStatsJson) {
    SignalDrawdownMeter m;
    for (int i = 0; i < 15; ++i) m.record(0.05);
    auto json = m.to_stats_json();
    EXPECT_NE(json.find("cum_bias"), std::string::npos);
    EXPECT_NE(json.find("max_drawdown"), std::string::npos);
}

} // namespace llmquant::tests
