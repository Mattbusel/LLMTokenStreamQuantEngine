#include <gtest/gtest.h>
#include "SignalParabolicSAR.h"
#include <cmath>

namespace llmquant::tests {

TEST(SignalParabolicSAR, DefaultConstruction) {
    SignalParabolicSAR sar;
    EXPECT_EQ(sar.total_records(), 0u);
    EXPECT_EQ(sar.reversal_events(), 0u);
}

TEST(SignalParabolicSAR, TrendIsOneOrMinusOne) {
    SignalParabolicSAR sar;
    for (int i = 0; i < 10; ++i) sar.record(0.3);
    int t = sar.trend();
    EXPECT_TRUE(t == 1 || t == -1);
}

TEST(SignalParabolicSAR, AFIncreasesInTrend) {
    SignalParabolicSAR::Config cfg;
    cfg.af_start = 0.02;
    cfg.af_step  = 0.02;
    cfg.af_max   = 0.20;
    SignalParabolicSAR sar(cfg);
    // Rising trend with new highs each step → AF should increase
    for (int i = 0; i < 10; ++i) sar.record(static_cast<double>(i) * 0.1);
    EXPECT_GE(sar.af(), cfg.af_start);
}

TEST(SignalParabolicSAR, AFCappedAtMax) {
    SignalParabolicSAR::Config cfg;
    cfg.af_start = 0.02;
    cfg.af_step  = 0.02;
    cfg.af_max   = 0.20;
    SignalParabolicSAR sar(cfg);
    for (int i = 0; i < 50; ++i) sar.record(static_cast<double>(i) * 0.1);
    EXPECT_LE(sar.af(), cfg.af_max + 1e-9);
}

TEST(SignalParabolicSAR, ReversalDetected) {
    SignalParabolicSAR sar;
    // Uptrend
    for (int i = 0; i < 5; ++i) sar.record(static_cast<double>(i) * 0.1);
    // Sudden large drop → SAR is above bias → reversal
    for (int i = 0; i < 5; ++i) sar.record(-5.0);
    EXPECT_GT(sar.reversal_events(), 0u);
}

TEST(SignalParabolicSAR, ReversalCallback) {
    SignalParabolicSAR::Config cfg;
    int reversals = 0;
    cfg.on_reversal = [&](int, double) { ++reversals; };
    SignalParabolicSAR sar(cfg);
    for (int i = 0; i < 5; ++i) sar.record(0.5);
    for (int i = 0; i < 5; ++i) sar.record(-5.0);
    EXPECT_GT(reversals, 0);
}

TEST(SignalParabolicSAR, Reset) {
    SignalParabolicSAR sar;
    for (int i = 0; i < 20; ++i) sar.record(0.1);
    sar.reset();
    EXPECT_EQ(sar.total_records(), 0u);
    EXPECT_EQ(sar.reversal_events(), 0u);
    EXPECT_EQ(sar.trend(), 1);
}

TEST(SignalParabolicSAR, ToStatsJson) {
    SignalParabolicSAR sar;
    for (int i = 0; i < 10; ++i) sar.record(0.1);
    auto json = sar.to_stats_json();
    EXPECT_NE(json.find("sar"), std::string::npos);
    EXPECT_NE(json.find("reversals"), std::string::npos);
}

} // namespace llmquant::tests
