#include <gtest/gtest.h>

#ifdef LLMQUANT_CONFIDENCE_CALIBRATOR_ENABLED

#include "ConfidenceCalibrator.h"

#include <cmath>

using namespace llmquant;

namespace {

// ============================================================
// Construction
// ============================================================

TEST(ConfidenceCalibrator, DefaultConstructsNotCalibrated) {
    ConfidenceCalibrator cal;
    EXPECT_FALSE(cal.is_calibrated());
    EXPECT_EQ(cal.total_outcomes(), 0u);
    EXPECT_NEAR(cal.empirical_win_rate(), 0.0, 1e-12);
}

TEST(ConfidenceCalibrator, PlattParamsInitiallyIdentity) {
    ConfidenceCalibrator cal;
    auto [a, b] = cal.platt_params();
    EXPECT_NEAR(a, 1.0, 1e-9);
    EXPECT_NEAR(b, 0.0, 1e-9);
}

// ============================================================
// Pass-through before calibration
// ============================================================

TEST(ConfidenceCalibrator, PassThroughBeforeMinSamples) {
    ConfidenceCalibrator::Config cfg;
    cfg.min_samples = 10;
    ConfidenceCalibrator cal(cfg);

    // Feed 5 outcomes — not yet calibrated.
    for (int i = 0; i < 5; ++i) cal.record_outcome(0.7, true);

    double raw = 0.65;
    EXPECT_NEAR(cal.calibrate(raw), raw, 1e-9);
}

// ============================================================
// Calibration activates after min_samples
// ============================================================

TEST(ConfidenceCalibrator, BecomesActiveAfterMinSamples) {
    ConfidenceCalibrator::Config cfg;
    cfg.min_samples   = 5;
    cfg.learning_rate = 0.1;
    ConfidenceCalibrator cal(cfg);

    for (int i = 0; i < 5; ++i) cal.record_outcome(0.8, true);
    EXPECT_TRUE(cal.is_calibrated());
    EXPECT_EQ(cal.total_outcomes(), 5u);
}

// ============================================================
// empirical_win_rate
// ============================================================

TEST(ConfidenceCalibrator, WinRateReflectsOutcomes) {
    ConfidenceCalibrator::Config cfg;
    cfg.min_samples = 2;
    ConfidenceCalibrator cal(cfg);

    cal.record_outcome(0.8, true);   // win
    cal.record_outcome(0.6, false);  // loss
    cal.record_outcome(0.7, true);   // win

    EXPECT_NEAR(cal.empirical_win_rate(), 2.0 / 3.0, 1e-9);
}

// ============================================================
// Calibrate returns value in [0, 1]
// ============================================================

TEST(ConfidenceCalibrator, CalibrateOutputBounded) {
    ConfidenceCalibrator::Config cfg;
    cfg.min_samples   = 5;
    cfg.learning_rate = 0.01;
    ConfidenceCalibrator cal(cfg);

    for (int i = 0; i < 10; ++i) cal.record_outcome(0.7, i % 2 == 0);

    double out = cal.calibrate(0.8);
    EXPECT_GE(out, 0.0);
    EXPECT_LE(out, 1.0);
    EXPECT_FALSE(std::isnan(out));
}

// ============================================================
// Calibrate shifts over-confident predictions toward win rate
// ============================================================

TEST(ConfidenceCalibrator, ConvergesForHighConfidenceLowWinRate) {
    ConfidenceCalibrator::Config cfg;
    cfg.min_samples   = 5;
    cfg.learning_rate = 0.05;
    cfg.window_size   = 200;
    ConfidenceCalibrator cal(cfg);

    // System is claiming 0.9 confidence but only winning 40% of the time.
    for (int i = 0; i < 100; ++i) cal.record_outcome(0.9, i % 5 < 2);  // 40% win

    // Calibrated confidence for 0.9 should be lower than 0.9.
    double calibrated = cal.calibrate(0.9);
    EXPECT_LT(calibrated, 0.9);
}

// ============================================================
// Sliding window evicts old data
// ============================================================

TEST(ConfidenceCalibrator, WindowEvictsOldOutcomes) {
    ConfidenceCalibrator::Config cfg;
    cfg.min_samples   = 2;
    cfg.window_size   = 5;
    cfg.learning_rate = 0.01;
    ConfidenceCalibrator cal(cfg);

    // Fill with losses.
    for (int i = 0; i < 5; ++i) cal.record_outcome(0.8, false);
    double wr_all_loss = cal.empirical_win_rate();

    // Overwrite with wins.
    for (int i = 0; i < 5; ++i) cal.record_outcome(0.8, true);
    double wr_all_win = cal.empirical_win_rate();

    EXPECT_GT(wr_all_win, wr_all_loss);
}

// ============================================================
// reset
// ============================================================

TEST(ConfidenceCalibrator, ResetClearsHistory) {
    ConfidenceCalibrator::Config cfg;
    cfg.min_samples = 3;
    ConfidenceCalibrator cal(cfg);

    for (int i = 0; i < 5; ++i) cal.record_outcome(0.7, true);
    EXPECT_TRUE(cal.is_calibrated());

    cal.reset();
    EXPECT_FALSE(cal.is_calibrated());
    EXPECT_EQ(cal.total_outcomes(), 0u);
    auto [a, b] = cal.platt_params();
    EXPECT_NEAR(a, 1.0, 1e-9);
    EXPECT_NEAR(b, 0.0, 1e-9);
}

// ============================================================
// update_config
// ============================================================

TEST(ConfidenceCalibrator, UpdateConfigResetsState) {
    ConfidenceCalibrator cal;
    for (int i = 0; i < 30; ++i) cal.record_outcome(0.7, true);

    ConfidenceCalibrator::Config cfg;
    cfg.min_samples = 5;
    cal.update_config(cfg);
    EXPECT_EQ(cal.total_outcomes(), 0u);
    EXPECT_FALSE(cal.is_calibrated());
}

// ============================================================
// to_stats_json
// ============================================================

TEST(ConfidenceCalibrator, ToStatsJsonContainsKeys) {
    ConfidenceCalibrator cal;
    cal.record_outcome(0.6, true);
    std::string json = cal.to_stats_json();
    EXPECT_NE(json.find("calibrated"),         std::string::npos);
    EXPECT_NE(json.find("platt_a"),            std::string::npos);
    EXPECT_NE(json.find("empirical_win_rate"),  std::string::npos);
    EXPECT_NE(json.find("total_outcomes"),      std::string::npos);
}

} // namespace

#else  // LLMQUANT_CONFIDENCE_CALIBRATOR_ENABLED not defined

TEST(ConfidenceCalibrator, DisabledAtBuildTime) {
    SUCCEED();
}

#endif  // LLMQUANT_CONFIDENCE_CALIBRATOR_ENABLED
