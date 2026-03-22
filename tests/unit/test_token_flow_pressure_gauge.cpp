#include <gtest/gtest.h>

#ifdef LLMQUANT_FLOW_PRESSURE_ENABLED
#include "TokenFlowPressureGauge.h"
using namespace llmquant;

TEST(TokenFlowPressureGaugeTest, InitialState) {
    TokenFlowPressureGauge g;
    EXPECT_GE(g.pressure(), 0.0);
    EXPECT_LE(g.pressure(), 1.0);
    EXPECT_FALSE(g.is_spiking());
    EXPECT_EQ(g.token_count(), 0u);
}

TEST(TokenFlowPressureGaugeTest, TokenCountIncrement) {
    TokenFlowPressureGauge g;
    g.record_token(1000000ULL);
    g.record_token(2000000ULL);
    EXPECT_EQ(g.token_count(), 2u);
}

TEST(TokenFlowPressureGaugeTest, PressureRangeValid) {
    TokenFlowPressureGauge g;
    uint64_t t = 0;
    for (int i = 0; i < 50; ++i) {
        t += 10'000'000ULL; // 10 ms intervals
        g.record_token(t);
    }
    EXPECT_GE(g.pressure(), 0.0);
    EXPECT_LE(g.pressure(), 1.0);
}

TEST(TokenFlowPressureGaugeTest, HighRateProducesHighPressure) {
    // Target is 33 ms; inject at 5 ms → pressure → 1.0
    TokenFlowPressureGauge::Config cfg;
    cfg.target_interval_ns = 33'000'000ULL;
    cfg.ema_alpha          = 0.50; // fast convergence
    cfg.spike_threshold    = 0.85;
    TokenFlowPressureGauge g(cfg);

    uint64_t t = 0;
    for (int i = 0; i < 30; ++i) {
        t += 5'000'000ULL; // 5 ms — much faster than 33 ms target
        g.record_token(t);
    }
    EXPECT_GT(g.pressure(), 0.85);
    EXPECT_TRUE(g.is_spiking());
}

TEST(TokenFlowPressureGaugeTest, SpikingCallback) {
    TokenFlowPressureGauge::Config cfg;
    cfg.target_interval_ns = 33'000'000ULL;
    cfg.ema_alpha          = 0.50;
    cfg.spike_threshold    = 0.85;
    int fires = 0;
    cfg.on_pressure_spike = [&](double) { ++fires; };
    TokenFlowPressureGauge g(cfg);

    uint64_t t = 0;
    for (int i = 0; i < 30; ++i) {
        t += 5'000'000ULL;
        g.record_token(t);
    }
    EXPECT_GE(fires, 1);
}

TEST(TokenFlowPressureGaugeTest, LowRateProducesLowPressure) {
    // Target is 33 ms; inject at 500 ms → pressure → 0
    TokenFlowPressureGauge::Config cfg;
    cfg.target_interval_ns = 33'000'000ULL;
    cfg.ema_alpha          = 0.50;
    cfg.spike_threshold    = 0.85;
    TokenFlowPressureGauge g(cfg);

    uint64_t t = 0;
    for (int i = 0; i < 20; ++i) {
        t += 500'000'000ULL; // 500 ms — much slower than 33 ms target
        g.record_token(t);
    }
    EXPECT_LT(g.pressure(), 0.2);
    EXPECT_FALSE(g.is_spiking());
}

TEST(TokenFlowPressureGaugeTest, Reset) {
    TokenFlowPressureGauge g;
    g.record_token(1'000'000ULL);
    g.record_token(2'000'000ULL);
    g.reset();
    EXPECT_EQ(g.token_count(), 0u);
    EXPECT_EQ(g.pressure(), 0.0);
}

TEST(TokenFlowPressureGaugeTest, StatsJson) {
    TokenFlowPressureGauge g;
    g.record_token(1'000'000ULL);
    std::string json = g.to_stats_json();
    EXPECT_NE(json.find("pressure"), std::string::npos);
}

#else
TEST(TokenFlowPressureGaugeTest, Disabled) { SUCCEED(); }
#endif
