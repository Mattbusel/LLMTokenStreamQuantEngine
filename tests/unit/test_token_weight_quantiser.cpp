#include <gtest/gtest.h>

#ifdef LLMQUANT_TOKEN_QUANTISER_ENABLED
#include "TokenWeightQuantiser.h"
#include <atomic>
#include <cmath>
#include <thread>
using namespace llmquant;

TEST(TokenWeightQuantiserTest, InitialState) {
    TokenWeightQuantiser q;
    EXPECT_EQ(q.total_count(), 0u);
    EXPECT_EQ(q.clamp_count(), 0u);
    EXPECT_NEAR(q.clamp_rate(), 0.0, 1e-9);
}

TEST(TokenWeightQuantiserTest, OutputOnGrid) {
    TokenWeightQuantiser::Config cfg;
    cfg.levels    = 5;
    cfg.range_min = 0.0;
    cfg.range_max = 1.0;
    TokenWeightQuantiser q(cfg);
    // Grid: 0.0, 0.25, 0.5, 0.75, 1.0
    for (int trial = 0; trial < 100; ++trial) {
        double out = q.quantise(0.5);
        // Must be one of the 5 grid points.
        bool valid = (std::abs(out - 0.0) < 1e-9 || std::abs(out - 0.25) < 1e-9 ||
                      std::abs(out - 0.5) < 1e-9 || std::abs(out - 0.75) < 1e-9 ||
                      std::abs(out - 1.0) < 1e-9);
        EXPECT_TRUE(valid);
    }
}

TEST(TokenWeightQuantiserTest, ExpectedValuePreserved) {
    // Stochastic rounding: E[quantise(w)] = w exactly.
    TokenWeightQuantiser::Config cfg;
    cfg.levels    = 100;
    cfg.range_min = -1.0;
    cfg.range_max =  1.0;
    TokenWeightQuantiser q(cfg);
    double w = 0.333;
    double sum = 0.0;
    int N = 10000;
    for (int i = 0; i < N; ++i) sum += q.quantise(w);
    double mean = sum / N;
    EXPECT_NEAR(mean, w, 0.05);  // statistical tolerance
}

TEST(TokenWeightQuantiserTest, ClampCounts) {
    TokenWeightQuantiser::Config cfg;
    cfg.range_min = -1.0;
    cfg.range_max =  1.0;
    cfg.clamp_alert_fraction = 1.0;  // suppress alert
    TokenWeightQuantiser q(cfg);
    q.quantise(2.0);   // clamped
    q.quantise(-2.0);  // clamped
    q.quantise(0.5);   // not clamped
    EXPECT_EQ(q.clamp_count(), 2u);
    EXPECT_EQ(q.total_count(), 3u);
}

TEST(TokenWeightQuantiserTest, ClampRateCorrect) {
    TokenWeightQuantiser::Config cfg;
    cfg.clamp_alert_fraction = 1.0;
    TokenWeightQuantiser q(cfg);
    q.quantise(5.0);  // clamped
    q.quantise(0.0);  // not clamped
    EXPECT_NEAR(q.clamp_rate(), 0.5, 1e-9);
}

TEST(TokenWeightQuantiserTest, QuantiseLevelInBounds) {
    TokenWeightQuantiser::Config cfg;
    cfg.levels = 256;
    TokenWeightQuantiser q(cfg);
    for (int i = 0; i < 100; ++i) {
        int level = q.quantise_level(static_cast<double>(i % 3 - 1) * 0.5);
        EXPECT_GE(level, 0);
        EXPECT_LT(level, cfg.levels);
    }
}

TEST(TokenWeightQuantiserTest, QuantisationErrorNonNegative) {
    TokenWeightQuantiser::Config cfg;
    cfg.levels = 4;
    TokenWeightQuantiser q(cfg);
    for (int i = 0; i < 50; ++i) q.quantise(i * 0.02 - 0.5);
    EXPECT_GE(q.quantisation_error(), 0.0);
}

TEST(TokenWeightQuantiserTest, ClampAlertCallbackFires) {
    TokenWeightQuantiser::Config cfg;
    cfg.clamp_alert_fraction = 0.1;  // alert when >10% clamped
    std::atomic<int> fires{0};
    cfg.on_high_clamp_rate = [&](double, double) { ++fires; };
    TokenWeightQuantiser q(cfg);
    // All weights out of range → clamp rate = 100% > 10%.
    for (int i = 0; i < 10; ++i) q.quantise(5.0);
    EXPECT_GE(fires.load(), 1);
}

TEST(TokenWeightQuantiserTest, ResetClearsState) {
    TokenWeightQuantiser q;
    q.quantise(0.5);
    q.quantise(2.0);
    q.reset();
    EXPECT_EQ(q.total_count(), 0u);
    EXPECT_EQ(q.clamp_count(), 0u);
}

TEST(TokenWeightQuantiserTest, StatsJsonNonEmpty) {
    TokenWeightQuantiser q;
    q.quantise(0.5);
    std::string j = q.to_stats_json();
    EXPECT_NE(j.find("quantisation_error"), std::string::npos);
    EXPECT_NE(j.find("clamp_rate"),         std::string::npos);
}

TEST(TokenWeightQuantiserTest, ConcurrentSafe) {
    TokenWeightQuantiser q;
    std::atomic<bool> go{false};
    auto worker = [&] {
        while (!go.load()) {}
        for (int i = 0; i < 500; ++i) q.quantise(i * 0.004 - 1.0);
    };
    std::thread t1(worker), t2(worker);
    go.store(true);
    t1.join(); t2.join();
    EXPECT_EQ(q.total_count(), 1000u);
}

#else
TEST(TokenWeightQuantiserTest, DisabledAtBuildTime) { SUCCEED(); }
#endif
