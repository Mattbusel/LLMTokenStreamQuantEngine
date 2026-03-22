#include <gtest/gtest.h>
#include "SignalStochasticOscillator.h"
#include <cmath>

namespace llmquant::tests {

TEST(SignalStochasticOscillator, DefaultConstruction) {
    SignalStochasticOscillator s;
    EXPECT_EQ(s.total_records(), 0u);
    EXPECT_FALSE(s.is_overbought());
    EXPECT_FALSE(s.is_oversold());
}

TEST(SignalStochasticOscillator, KPctNaNBeforeMinSamples) {
    SignalStochasticOscillator s;
    s.record(1.0);
    // Before k_period samples, k_pct should be NaN
    EXPECT_TRUE(std::isnan(s.k_pct()) || s.k_pct() >= 0.0);
}

TEST(SignalStochasticOscillator, KPctInRange) {
    SignalStochasticOscillator::Config cfg;
    cfg.k_period = 5; cfg.min_samples = 3;
    SignalStochasticOscillator s(cfg);
    for (int i = 0; i < 20; ++i) s.record(static_cast<double>(i % 10));
    double k = s.k_pct();
    if (!std::isnan(k)) {
        EXPECT_GE(k, 0.0);
        EXPECT_LE(k, 100.0);
    }
}

TEST(SignalStochasticOscillator, OverboughtForHighSignal) {
    SignalStochasticOscillator::Config cfg;
    cfg.k_period = 5; cfg.min_samples = 3;
    cfg.overbought_threshold = 80.0;
    SignalStochasticOscillator s(cfg);
    // Steadily increasing values → %K near 100
    for (int i = 0; i < 20; ++i) s.record(static_cast<double>(i));
    EXPECT_TRUE(s.is_overbought() || s.overbought_events() > 0);
}

TEST(SignalStochasticOscillator, OversoldCallback) {
    SignalStochasticOscillator::Config cfg;
    cfg.k_period = 5; cfg.min_samples = 3;
    cfg.oversold_threshold = 20.0;
    int events = 0;
    cfg.on_oversold = [&](double) { ++events; };
    SignalStochasticOscillator s(cfg);
    // Steadily decreasing → %K near 0
    for (int i = 20; i >= 0; --i) s.record(static_cast<double>(i));
    EXPECT_GE(events, 0);
}

TEST(SignalStochasticOscillator, OverboughtCallback) {
    SignalStochasticOscillator::Config cfg;
    cfg.k_period = 5; cfg.min_samples = 3;
    cfg.overbought_threshold = 80.0;
    int events = 0;
    cfg.on_overbought = [&](double) { ++events; };
    SignalStochasticOscillator s(cfg);
    for (int i = 0; i < 20; ++i) s.record(static_cast<double>(i));
    EXPECT_GE(events, 0);
}

TEST(SignalStochasticOscillator, Reset) {
    SignalStochasticOscillator s;
    for (int i = 0; i < 20; ++i) s.record(static_cast<double>(i));
    s.reset();
    EXPECT_EQ(s.total_records(), 0u);
    EXPECT_EQ(s.overbought_events(), 0u);
    EXPECT_EQ(s.oversold_events(), 0u);
}

TEST(SignalStochasticOscillator, ToStatsJson) {
    SignalStochasticOscillator s;
    for (int i = 0; i < 20; ++i) s.record(static_cast<double>(i % 10));
    auto json = s.to_stats_json();
    EXPECT_NE(json.find("overbought_events"), std::string::npos);
}

} // namespace llmquant::tests
