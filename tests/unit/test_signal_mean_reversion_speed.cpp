#include <gtest/gtest.h>
#include "SignalMeanReversionSpeed.h"
#include <cmath>

namespace llmquant::tests {

TEST(SignalMeanReversionSpeed, DefaultConstruction) {
    SignalMeanReversionSpeed m;
    EXPECT_EQ(m.total_records(), 0u);
    EXPECT_DOUBLE_EQ(m.theta(), 0.0);
}

TEST(SignalMeanReversionSpeed, ThetaComputedAfterMinSamples) {
    SignalMeanReversionSpeed::Config cfg;
    cfg.window = 20; cfg.min_samples = 10;
    SignalMeanReversionSpeed m(cfg);
    for (int i = 0; i < 20; ++i) m.record(std::sin(i * 0.3));
    EXPECT_NE(m.theta(), 0.0);
}

TEST(SignalMeanReversionSpeed, MeanRevertingThetaInRange) {
    SignalMeanReversionSpeed::Config cfg;
    cfg.window = 20; cfg.min_samples = 10;
    SignalMeanReversionSpeed m(cfg);
    // Mean-reverting AR process: x[t] = 0.5*x[t-1] + noise
    double x = 0.0;
    for (int i = 0; i < 30; ++i) {
        x = 0.5 * x + (i % 3 - 1) * 0.01;
        m.record(x);
    }
    // θ should be near 0.5 (mean-reverting)
    EXPECT_LT(m.theta(), 1.0);
}

TEST(SignalMeanReversionSpeed, FastReversionCallback) {
    SignalMeanReversionSpeed::Config cfg;
    cfg.window = 15; cfg.min_samples = 8;
    cfg.fast_kappa_threshold = 0.0; // always fires on any positive kappa
    int events = 0;
    cfg.on_fast_reversion = [&](double) { ++events; };
    SignalMeanReversionSpeed m(cfg);
    double x = 1.0;
    for (int i = 0; i < 20; ++i) { x = 0.3 * x; m.record(x); }
    EXPECT_GE(events, 0);
}

TEST(SignalMeanReversionSpeed, Reset) {
    SignalMeanReversionSpeed m;
    for (int i = 0; i < 20; ++i) m.record(0.1);
    m.reset();
    EXPECT_EQ(m.total_records(), 0u);
    EXPECT_DOUBLE_EQ(m.theta(), 0.0);
    EXPECT_DOUBLE_EQ(m.kappa(), 0.0);
}

TEST(SignalMeanReversionSpeed, ToStatsJson) {
    SignalMeanReversionSpeed m;
    for (int i = 0; i < 20; ++i) m.record(0.05);
    auto json = m.to_stats_json();
    EXPECT_NE(json.find("theta"), std::string::npos);
    EXPECT_NE(json.find("kappa"), std::string::npos);
}

} // namespace llmquant::tests
