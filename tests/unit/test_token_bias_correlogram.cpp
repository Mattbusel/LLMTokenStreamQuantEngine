#include <gtest/gtest.h>
#include "TokenBiasCorrelogram.h"
#include <cmath>

namespace llmquant::tests {

TEST(TokenBiasCorrelogram, DefaultConstruction) {
    TokenBiasCorrelogram c;
    EXPECT_EQ(c.total_records(), 0u);
    EXPECT_EQ(c.dominant_lag(), 0);
}

TEST(TokenBiasCorrelogram, AcfAtLag1HighForPeriodicSignal) {
    TokenBiasCorrelogram::Config cfg;
    cfg.window      = 32;
    cfg.max_lag     = 8;
    cfg.min_samples = 16;
    TokenBiasCorrelogram c(cfg);
    // Periodic signal with period 2 → strong ACF at lag 2
    for (int i = 0; i < 50; ++i) c.record(i % 2 == 0 ? 1.0 : -1.0);
    // ACF at lag 2 should be near 1 for alternating signal
    double a2 = c.acf_at(2);
    EXPECT_GT(a2, 0.5);
}

TEST(TokenBiasCorrelogram, PeakAcfInRange) {
    TokenBiasCorrelogram c;
    for (int i = 0; i < 40; ++i) c.record(std::sin(i * 0.5));
    double p = c.peak_acf();
    EXPECT_LE(p, 1.0);
    EXPECT_GE(p, -1.0);
}

TEST(TokenBiasCorrelogram, CycleCallback) {
    TokenBiasCorrelogram::Config cfg;
    cfg.window          = 32;
    cfg.max_lag         = 8;
    cfg.min_samples     = 16;
    cfg.cycle_threshold = 0.3;
    int cycles = 0;
    cfg.on_cycle_detected = [&](int, double) { ++cycles; };
    TokenBiasCorrelogram c(cfg);
    for (int i = 0; i < 60; ++i) c.record(i % 2 == 0 ? 1.0 : -1.0);
    EXPECT_GT(cycles, 0);
}

TEST(TokenBiasCorrelogram, DominantLagPositive) {
    TokenBiasCorrelogram::Config cfg;
    cfg.window = 32; cfg.max_lag = 8; cfg.min_samples = 16;
    TokenBiasCorrelogram c(cfg);
    for (int i = 0; i < 40; ++i) c.record(i % 4 == 0 ? 1.0 : 0.0);
    EXPECT_GE(c.dominant_lag(), 0);
}

TEST(TokenBiasCorrelogram, Reset) {
    TokenBiasCorrelogram c;
    for (int i = 0; i < 30; ++i) c.record(0.1);
    c.reset();
    EXPECT_EQ(c.total_records(), 0u);
    EXPECT_EQ(c.dominant_lag(), 0);
    EXPECT_DOUBLE_EQ(c.peak_acf(), 0.0);
}

TEST(TokenBiasCorrelogram, ToStatsJson) {
    TokenBiasCorrelogram c;
    for (int i = 0; i < 25; ++i) c.record(0.05);
    auto json = c.to_stats_json();
    EXPECT_NE(json.find("dominant_lag"), std::string::npos);
    EXPECT_NE(json.find("peak_acf"), std::string::npos);
}

} // namespace llmquant::tests
