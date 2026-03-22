#include <gtest/gtest.h>

#ifdef LLMQUANT_CAUSAL_IMPACT_ENABLED
#include "CausalImpactEstimator.h"
using namespace llmquant;

TEST(CausalImpactEstimatorTest, InitialState) {
    CausalImpactEstimator est;
    EXPECT_NEAR(est.cusum_stat(), 0.0, 1e-9);
    EXPECT_FALSE(est.break_detected());
    EXPECT_EQ(est.break_count(), 0u);
    EXPECT_EQ(est.observation_count(), 0u);
}

TEST(CausalImpactEstimatorTest, ObservationCountIncrement) {
    CausalImpactEstimator est;
    est.record_return(0.001);
    est.record_return(-0.002);
    EXPECT_EQ(est.observation_count(), 2u);
}

TEST(CausalImpactEstimatorTest, BreakDetectedOnLargeReturn) {
    CausalImpactEstimator::Config cfg;
    cfg.warmup_window = 5;
    cfg.sensitivity   = 0.001;
    cfg.threshold     = 0.01;
    CausalImpactEstimator est(cfg);
    // Warm up.
    for (int i = 0; i < 10; ++i)
        est.record_return(0.0);
    // Feed a large positive shock to push CUSUM over threshold.
    for (int i = 0; i < 20; ++i)
        est.record_return(0.1);
    EXPECT_TRUE(est.break_detected());
    EXPECT_GE(est.break_count(), 1u);
}

TEST(CausalImpactEstimatorTest, BreakCallbackFires) {
    CausalImpactEstimator::Config cfg;
    cfg.warmup_window = 3;
    cfg.threshold     = 0.005;
    int called = 0;
    cfg.on_break = [&](double, const std::string&, double) { ++called; };
    CausalImpactEstimator est(cfg);
    for (int i = 0; i < 5; ++i) est.record_return(0.0);
    for (int i = 0; i < 20; ++i) est.record_return(0.5);
    EXPECT_GE(called, 1);
}

TEST(CausalImpactEstimatorTest, EventAttributionInWindow) {
    CausalImpactEstimator::Config cfg;
    cfg.warmup_window = 3;
    cfg.threshold     = 0.005;
    cfg.event_lookback = 30;
    std::string attributed;
    cfg.on_break = [&](double, const std::string& label, double) {
        if (!label.empty()) attributed = label;
    };
    CausalImpactEstimator est(cfg);
    for (int i = 0; i < 5; ++i) est.record_return(0.0);
    est.record_event("earnings_beat");
    for (int i = 0; i < 20; ++i) est.record_return(0.5);
    EXPECT_EQ(attributed, "earnings_beat");
}

TEST(CausalImpactEstimatorTest, ResetCusum) {
    CausalImpactEstimator::Config cfg;
    cfg.warmup_window = 3;
    cfg.threshold     = 0.005;
    CausalImpactEstimator est(cfg);
    for (int i = 0; i < 5; ++i) est.record_return(0.0);
    for (int i = 0; i < 20; ++i) est.record_return(0.5);
    EXPECT_TRUE(est.break_detected());
    est.reset_cusum();
    EXPECT_NEAR(est.cusum_stat(), 0.0, 1e-9);
    EXPECT_FALSE(est.break_detected());
}

TEST(CausalImpactEstimatorTest, FullReset) {
    CausalImpactEstimator est;
    est.record_return(0.1);
    est.reset();
    EXPECT_EQ(est.observation_count(), 0u);
    EXPECT_EQ(est.break_count(), 0u);
}

TEST(CausalImpactEstimatorTest, StatsJson) {
    CausalImpactEstimator est;
    est.record_return(0.001);
    std::string json = est.to_stats_json();
    EXPECT_NE(json.find("cusum_stat"), std::string::npos);
    EXPECT_NE(json.find("break_count"), std::string::npos);
}

#else
TEST(CausalImpactEstimatorTest, Disabled) { SUCCEED(); }
#endif
