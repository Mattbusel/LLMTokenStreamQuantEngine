#include <gtest/gtest.h>

#ifdef LLMQUANT_BAYESIAN_FILTER_ENABLED
#include "BayesianSignalFilter.h"
#include <cmath>

using namespace llmquant;

TEST(BayesianSignalFilterTest, ConstructsOk) {
    BayesianSignalFilter f;
    EXPECT_EQ(f.total_signals(), 0u);
    EXPECT_EQ(f.total_outcomes(), 0u);
}

TEST(BayesianSignalFilterTest, DefaultPriorIsUninformative) {
    BayesianSignalFilter f;
    // Prior α=β=2 → posterior mean = 0.5
    EXPECT_NEAR(f.posterior_confidence(true),  0.5, 1e-9);
    EXPECT_NEAR(f.posterior_confidence(false), 0.5, 1e-9);
}

TEST(BayesianSignalFilterTest, WinsIncreaseBullConfidence) {
    BayesianSignalFilter f;
    for (int i = 0; i < 10; ++i) f.record_outcome(true, true);
    EXPECT_GT(f.posterior_confidence(true), 0.5);
}

TEST(BayesianSignalFilterTest, LossesDecreaseBullConfidence) {
    BayesianSignalFilter f;
    for (int i = 0; i < 10; ++i) f.record_outcome(true, false);
    EXPECT_LT(f.posterior_confidence(true), 0.5);
}

TEST(BayesianSignalFilterTest, BullAndBearAreIndependent) {
    BayesianSignalFilter f;
    for (int i = 0; i < 10; ++i) f.record_outcome(true,  true);
    for (int i = 0; i < 10; ++i) f.record_outcome(false, false);
    EXPECT_GT(f.posterior_confidence(true),  0.5);
    EXPECT_LT(f.posterior_confidence(false), 0.5);
}

TEST(BayesianSignalFilterTest, RecordSignalCountsSignals) {
    BayesianSignalFilter f;
    f.record_signal(true);
    f.record_signal(false);
    f.record_signal(true);
    EXPECT_EQ(f.total_signals(), 3u);
}

TEST(BayesianSignalFilterTest, RecordOutcomeCountsOutcomes) {
    BayesianSignalFilter f;
    f.record_outcome(true,  true);
    f.record_outcome(false, false);
    EXPECT_EQ(f.total_outcomes(), 2u);
}

TEST(BayesianSignalFilterTest, PosteriorPrecisionGrowsWithOutcomes) {
    BayesianSignalFilter f;
    double prec_before = f.posterior_precision(true);
    f.record_outcome(true, true);
    EXPECT_GT(f.posterior_precision(true), prec_before);
}

TEST(BayesianSignalFilterTest, CredibleIntervalHalfWidthIsPositive) {
    BayesianSignalFilter f;
    EXPECT_GT(f.credible_interval_half_width(true), 0.0);
}

TEST(BayesianSignalFilterTest, LowConfidenceCallbackFires) {
    BayesianSignalFilter::Config cfg;
    cfg.prior_alpha             = 2.0;
    cfg.prior_beta              = 2.0;
    cfg.forgetting              = 1.0; // no forgetting
    cfg.low_confidence_threshold = 0.4;
    int fires = 0;
    cfg.on_low_confidence = [&](bool, double) { ++fires; };
    BayesianSignalFilter f(cfg);

    // 20 losses → posterior drops well below 0.4
    for (int i = 0; i < 20; ++i) f.record_outcome(true, false);
    EXPECT_GE(fires, 1);
    EXPECT_GE(f.low_confidence_events(), 1u);
}

TEST(BayesianSignalFilterTest, ForgettingDecaysPosteriorTowardPrior) {
    BayesianSignalFilter::Config cfg;
    cfg.forgetting = 0.5; // aggressive forgetting
    BayesianSignalFilter f(cfg);

    // Build up wins, then add losses — forgetting should pull back toward 0.5.
    for (int i = 0; i < 20; ++i) f.record_outcome(true, true);
    double conf_after_wins = f.posterior_confidence(true);
    for (int i = 0; i < 5; ++i) f.record_outcome(true, false);
    // Confidence should drop after losses with aggressive forgetting.
    EXPECT_LT(f.posterior_confidence(true), conf_after_wins);
}

TEST(BayesianSignalFilterTest, ResetRestoresPrior) {
    BayesianSignalFilter f;
    for (int i = 0; i < 20; ++i) f.record_outcome(true, true);
    f.reset();
    EXPECT_NEAR(f.posterior_confidence(true), 0.5, 1e-9);
}

TEST(BayesianSignalFilterTest, UpdateConfigResetsPosteriors) {
    BayesianSignalFilter f;
    for (int i = 0; i < 20; ++i) f.record_outcome(true, true);
    BayesianSignalFilter::Config cfg2;
    cfg2.prior_alpha = 1.0;
    cfg2.prior_beta  = 1.0;
    f.update_config(cfg2);
    EXPECT_NEAR(f.posterior_confidence(true), 0.5, 1e-9);
}

TEST(BayesianSignalFilterTest, StatsJsonContainsRequiredKeys) {
    BayesianSignalFilter f;
    f.record_outcome(true, true);
    std::string json = f.to_stats_json();
    EXPECT_NE(json.find("bull_posterior"), std::string::npos);
    EXPECT_NE(json.find("bear_posterior"), std::string::npos);
    EXPECT_NE(json.find("total_signals"),  std::string::npos);
    EXPECT_NE(json.find("total_outcomes"), std::string::npos);
}

#else
TEST(BayesianSignalFilterTest, FeatureFlagDisabled) { SUCCEED(); }
#endif
