#include <gtest/gtest.h>

#ifdef LLMQUANT_VOLATILITY_FORECASTER_ENABLED
#include "VolatilityForecaster.h"
#include <cmath>

using namespace llmquant;

TEST(VolatilityForecasterTest, ConstructsOk) {
    VolatilityForecaster vf;
    EXPECT_EQ(vf.total_records(), 0u);
    EXPECT_FALSE(vf.is_high_vol());
}

TEST(VolatilityForecasterTest, InitialVarianceFromConfig) {
    VolatilityForecaster::Config cfg;
    cfg.initial_variance = 0.04;
    VolatilityForecaster vf(cfg);
    EXPECT_NEAR(vf.conditional_variance(), 0.04, 1e-9);
    EXPECT_NEAR(vf.conditional_vol(), 0.2, 1e-9);
}

TEST(VolatilityForecasterTest, FirstRecordUsesZeroInnovation) {
    // On first call has_prev_=false → r=0.
    // σ²_1 = ω + α*0 + β*σ²_0 = 0.00001 + 0.84*0.01
    VolatilityForecaster vf;
    vf.record(1.0);
    double expected = 0.00001 + 0.84 * 0.01;
    EXPECT_NEAR(vf.conditional_variance(), expected, 1e-9);
    EXPECT_EQ(vf.total_records(), 1u);
}

TEST(VolatilityForecasterTest, SubsequentRecordUpdatesGARCH) {
    VolatilityForecaster vf;
    vf.record(0.0);
    double sigma2_prev = vf.conditional_variance();
    vf.record(0.1); // r = 0.1 - 0.0 = 0.1
    double expected = 0.00001 + 0.15 * 0.01 + 0.84 * sigma2_prev;
    EXPECT_NEAR(vf.conditional_variance(), expected, 1e-9);
}

TEST(VolatilityForecasterTest, ConditionalVolIsSqrtVariance) {
    VolatilityForecaster vf;
    vf.record(0.0);
    EXPECT_NEAR(vf.conditional_vol(), std::sqrt(vf.conditional_variance()), 1e-9);
}

TEST(VolatilityForecasterTest, LongRunVarianceFormula) {
    VolatilityForecaster::Config cfg;
    cfg.omega = 0.00001;
    cfg.alpha = 0.15;
    cfg.beta  = 0.84;
    VolatilityForecaster vf(cfg);
    // ω / (1 - α - β) = 0.00001 / 0.01 = 0.001
    EXPECT_NEAR(vf.long_run_variance(), 0.001, 1e-9);
}

TEST(VolatilityForecasterTest, LongRunVarianceNonStationaryReturnsCurrent) {
    VolatilityForecaster::Config cfg;
    cfg.omega = 0.0001;
    cfg.alpha = 0.5;
    cfg.beta  = 0.5; // α + β = 1.0 → non-stationary
    VolatilityForecaster vf(cfg);
    // Should return conditional_variance()
    EXPECT_DOUBLE_EQ(vf.long_run_variance(), vf.conditional_variance());
}

TEST(VolatilityForecasterTest, HighVolCallbackFires) {
    VolatilityForecaster::Config cfg;
    cfg.initial_variance    = 0.0001;
    cfg.high_vol_threshold  = 0.001; // very low threshold
    int fires = 0;
    cfg.on_high_vol = [&](double, double) { ++fires; };
    VolatilityForecaster vf(cfg);

    // Inject large innovations to push vol above threshold.
    for (int i = 0; i < 5; ++i) vf.record(static_cast<double>(i) * 0.5);
    EXPECT_GE(fires, 1);
    EXPECT_GE(vf.high_vol_events(), 1u);
}

TEST(VolatilityForecasterTest, HighVolCallbackEdgeTriggered) {
    // Callback fires only on the transition from low→high, not every record while high.
    VolatilityForecaster::Config cfg;
    cfg.initial_variance   = 100.0; // start high
    cfg.high_vol_threshold = 0.001;  // immediately high
    int fires = 0;
    cfg.on_high_vol = [&](double, double) { ++fires; };
    VolatilityForecaster vf(cfg);

    // Once already high, additional records in high-vol state should not re-fire.
    vf.record(0.0);
    vf.record(0.0);
    vf.record(0.0);
    // fires should be exactly 1 (the first transition)
    EXPECT_EQ(fires, 1);
}

TEST(VolatilityForecasterTest, IsHighVolFlag) {
    VolatilityForecaster::Config cfg;
    cfg.initial_variance   = 100.0;
    cfg.high_vol_threshold = 0.001;
    VolatilityForecaster vf(cfg);
    vf.record(0.0);
    EXPECT_TRUE(vf.is_high_vol());
}

TEST(VolatilityForecasterTest, ResetRestoresInitialState) {
    VolatilityForecaster vf;
    for (int i = 0; i < 5; ++i) vf.record(static_cast<double>(i));
    vf.reset();
    EXPECT_NEAR(vf.conditional_variance(), 0.01, 1e-9);
    EXPECT_FALSE(vf.is_high_vol());
}

TEST(VolatilityForecasterTest, UpdateConfigResetsState) {
    VolatilityForecaster vf;
    for (int i = 0; i < 5; ++i) vf.record(static_cast<double>(i));
    VolatilityForecaster::Config cfg2;
    cfg2.initial_variance = 0.05;
    vf.update_config(cfg2);
    EXPECT_NEAR(vf.conditional_variance(), 0.05, 1e-9);
}

TEST(VolatilityForecasterTest, StatsJsonContainsRequiredKeys) {
    VolatilityForecaster vf;
    vf.record(0.1);
    std::string json = vf.to_stats_json();
    EXPECT_NE(json.find("conditional_variance"), std::string::npos);
    EXPECT_NE(json.find("conditional_vol"),      std::string::npos);
    EXPECT_NE(json.find("long_run_variance"),     std::string::npos);
    EXPECT_NE(json.find("is_high_vol"),           std::string::npos);
    EXPECT_NE(json.find("total_records"),         std::string::npos);
}

TEST(VolatilityForecasterTest, TotalRecordsCountsAllCalls) {
    VolatilityForecaster vf;
    for (int i = 0; i < 7; ++i) vf.record(1.0);
    EXPECT_EQ(vf.total_records(), 7u);
}

#else
TEST(VolatilityForecasterTest, FeatureFlagDisabled) { SUCCEED(); }
#endif // LLMQUANT_VOLATILITY_FORECASTER_ENABLED
