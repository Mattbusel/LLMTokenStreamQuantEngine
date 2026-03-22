#include <gtest/gtest.h>
#include "SignalTailRiskMeter.h"
#include <cmath>

namespace llmquant::tests {

TEST(SignalTailRiskMeter, DefaultConstruction) {
    SignalTailRiskMeter m;
    EXPECT_EQ(m.total_records(), 0u);
    EXPECT_EQ(m.tail_events(), 0u);
}

TEST(SignalTailRiskMeter, RecordsAccumulate) {
    SignalTailRiskMeter m;
    for (int i = 0; i < 30; ++i) m.record(0.1);
    EXPECT_EQ(m.total_records(), 30u);
}

TEST(SignalTailRiskMeter, ESLowerThanVaR) {
    SignalTailRiskMeter m;
    // Mix of positive and negative bias
    for (int i = 0; i < 100; ++i)
        m.record((i % 5 == 0) ? -0.9 : 0.1);
    // ES should be ≤ VaR (ES is conditional mean of the tail)
    EXPECT_LE(m.expected_shortfall(), m.var_alpha() + 1e-9);
}

TEST(SignalTailRiskMeter, TailEventCallback) {
    SignalTailRiskMeter::Config cfg;
    cfg.es_threshold = 0.0;  // ES < 0 fires event
    cfg.min_samples  = 5;
    int events = 0;
    cfg.on_tail_event = [&](double) { ++events; };
    SignalTailRiskMeter m(cfg);
    // Pure negative bias → ES will be negative → fires
    for (int i = 0; i < 30; ++i) m.record(-0.5);
    EXPECT_GT(events, 0);
}

TEST(SignalTailRiskMeter, Reset) {
    SignalTailRiskMeter m;
    for (int i = 0; i < 50; ++i) m.record(-0.2);
    m.reset();
    EXPECT_EQ(m.total_records(), 0u);
    EXPECT_DOUBLE_EQ(m.var_alpha(), 0.0);
}

TEST(SignalTailRiskMeter, ToStatsJson) {
    SignalTailRiskMeter m;
    for (int i = 0; i < 25; ++i) m.record(0.05);
    auto json = m.to_stats_json();
    EXPECT_NE(json.find("expected_shortfall"), std::string::npos);
    EXPECT_NE(json.find("var_alpha"), std::string::npos);
}

TEST(SignalTailRiskMeter, PurePositiveBiasHasPositiveES) {
    SignalTailRiskMeter m;
    for (int i = 0; i < 50; ++i) m.record(0.5);
    // All bias values are positive → ES (tail below quantile) still ≤ positive
    EXPECT_GE(m.expected_shortfall(), -1e-9);
}

} // namespace llmquant::tests
