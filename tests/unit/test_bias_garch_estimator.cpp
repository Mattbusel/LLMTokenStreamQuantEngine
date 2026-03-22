#include <gtest/gtest.h>
#include "BiasGarchEstimator.h"
#include <cmath>

namespace llmquant::tests {

TEST(BiasGarchEstimator, DefaultConstruction) {
    BiasGarchEstimator est;
    EXPECT_GE(est.conditional_vol(), 0.0);
    EXPECT_EQ(est.total_records(), 0u);
    EXPECT_EQ(est.spike_events(), 0u);
}

TEST(BiasGarchEstimator, VolPositiveAfterRecords) {
    BiasGarchEstimator est;
    for (int i = 0; i < 20; ++i)
        est.record((i % 2 == 0) ? 0.1 : -0.1);
    EXPECT_GT(est.conditional_vol(), 0.0);
    EXPECT_EQ(est.total_records(), 20u);
}

TEST(BiasGarchEstimator, PersistenceOmegaAlphaBeta) {
    BiasGarchEstimator::Config cfg;
    cfg.alpha = 0.10;
    cfg.beta  = 0.85;
    BiasGarchEstimator est(cfg);
    EXPECT_DOUBLE_EQ(est.persistence(), 0.95);
}

TEST(BiasGarchEstimator, VolSpikeCallback) {
    BiasGarchEstimator::Config cfg;
    cfg.sigma2_init   = 0.0001;
    cfg.vol_spike_ratio = 1.2;
    int spikes = 0;
    cfg.on_vol_spike = [&](double) { ++spikes; };
    BiasGarchEstimator est(cfg);
    // Feed large shock after warm-up
    for (int i = 0; i < 10; ++i) est.record(0.01);
    est.record(5.0); // large shock
    EXPECT_GT(spikes, 0);
}

TEST(BiasGarchEstimator, Reset) {
    BiasGarchEstimator est;
    for (int i = 0; i < 30; ++i) est.record(0.05);
    est.reset();
    EXPECT_EQ(est.total_records(), 0u);
    EXPECT_EQ(est.spike_events(), 0u);
}

TEST(BiasGarchEstimator, ToStatsJson) {
    BiasGarchEstimator est;
    for (int i = 0; i < 10; ++i) est.record(0.02);
    auto json = est.to_stats_json();
    EXPECT_NE(json.find("sigma"), std::string::npos);
    EXPECT_NE(json.find("persistence"), std::string::npos);
}

TEST(BiasGarchEstimator, UpdateConfig) {
    BiasGarchEstimator est;
    BiasGarchEstimator::Config cfg;
    cfg.alpha = 0.20;
    cfg.beta  = 0.70;
    est.update_config(cfg);
    EXPECT_DOUBLE_EQ(est.persistence(), 0.90);
}

TEST(BiasGarchEstimator, ConditionalVarPositive) {
    BiasGarchEstimator est;
    for (int i = 0; i < 50; ++i) est.record(static_cast<double>(i % 3 - 1) * 0.05);
    EXPECT_GT(est.conditional_var(), 0.0);
}

} // namespace llmquant::tests
