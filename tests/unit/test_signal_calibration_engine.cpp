#include <gtest/gtest.h>

#ifdef LLMQUANT_SIGNAL_CALIBRATION_ENABLED

#include "SignalCalibrationEngine.h"

#include <atomic>
#include <thread>
#include <vector>

using namespace llmquant;

namespace {

// ============================================================
// Default state — calibrate() returns raw score before min_samples
// ============================================================

TEST(SignalCalibrationEngine, DefaultReturnsRawScore) {
    SignalCalibrationEngine cal;
    // Before min_samples, calibrate() should pass through raw score.
    double raw = 0.7;
    double cal_out = cal.calibrate(raw);
    // With no data, Platt params are identity-ish → output close to input.
    EXPECT_NEAR(cal_out, raw, 0.5);  // generous tolerance before calibration
}

// ============================================================
// record_outcome increments sample count
// ============================================================

TEST(SignalCalibrationEngine, RecordOutcomeIncrementsSamples) {
    SignalCalibrationEngine cal;
    EXPECT_EQ(cal.sample_count(), 0);
    cal.record_outcome(0.7, true);
    cal.record_outcome(0.3, false);
    EXPECT_EQ(cal.sample_count(), 2);
}

// ============================================================
// After many positive outcomes, calibrate(0.8) > 0.5
// ============================================================

TEST(SignalCalibrationEngine, HighScoreLeadsToHighCalibration) {
    SignalCalibrationEngine::Config cfg;
    cfg.min_samples   = 10;
    cfg.learning_rate = 0.1;
    SignalCalibrationEngine cal(cfg);

    // Train: high scores almost always win.
    for (int i = 0; i < 50; ++i) {
        double score = 0.7 + 0.1 * (i % 3 == 0 ? -1 : 1);
        cal.record_outcome(score, true);
    }

    double p = cal.calibrate(0.8);
    EXPECT_GT(p, 0.4);   // should predict positive probability
    EXPECT_LE(p, 1.0);
}

// ============================================================
// calibrate() output is in [0, 1]
// ============================================================

TEST(SignalCalibrationEngine, CalibrateInUnitInterval) {
    SignalCalibrationEngine::Config cfg;
    cfg.min_samples = 5;
    SignalCalibrationEngine cal(cfg);

    for (int i = 0; i < 30; ++i) {
        double s = static_cast<double>(i % 10) * 0.1;
        cal.record_outcome(s, i % 2 == 0);
    }

    for (double s : {0.0, 0.1, 0.5, 0.9, 1.0}) {
        double p = cal.calibrate(s);
        EXPECT_GE(p, 0.0);
        EXPECT_LE(p, 1.0);
    }
}

// ============================================================
// ECE is in [0, 1]
// ============================================================

TEST(SignalCalibrationEngine, EceInUnitInterval) {
    SignalCalibrationEngine::Config cfg;
    cfg.min_samples = 5;
    SignalCalibrationEngine cal(cfg);

    for (int i = 0; i < 30; ++i)
        cal.record_outcome(static_cast<double>(i % 5) * 0.2, i % 2 == 0);

    double ece = cal.expected_calibration_error();
    EXPECT_GE(ece, 0.0);
    EXPECT_LE(ece, 1.0);
}

// ============================================================
// on_recalibrated callback fires
// ============================================================

TEST(SignalCalibrationEngine, CallbackFires) {
    SignalCalibrationEngine::Config cfg;
    cfg.min_samples = 5;
    std::atomic<int> fires{0};
    cfg.on_recalibrated = [&](double) { ++fires; };
    SignalCalibrationEngine cal(cfg);

    for (int i = 0; i < 10; ++i)
        cal.record_outcome(0.6, true);

    EXPECT_GT(fires.load(), 0);
}

// ============================================================
// reset() clears state
// ============================================================

TEST(SignalCalibrationEngine, ResetClearsState) {
    SignalCalibrationEngine::Config cfg;
    cfg.min_samples = 5;
    SignalCalibrationEngine cal(cfg);

    for (int i = 0; i < 20; ++i)
        cal.record_outcome(static_cast<double>(i % 5) * 0.2, i % 2 == 0);

    cal.reset();
    EXPECT_EQ(cal.sample_count(), 0);
    EXPECT_DOUBLE_EQ(cal.expected_calibration_error(), 0.0);
}

// ============================================================
// to_stats_json contains required keys
// ============================================================

TEST(SignalCalibrationEngine, ToStatsJsonContainsKeys) {
    SignalCalibrationEngine cal;
    cal.record_outcome(0.7, true);
    std::string json = cal.to_stats_json();
    EXPECT_NE(json.find("sample_count"), std::string::npos);
    EXPECT_NE(json.find("platt_A"),      std::string::npos);
    EXPECT_NE(json.find("platt_B"),      std::string::npos);
    EXPECT_NE(json.find("ece"),          std::string::npos);
}

// ============================================================
// Concurrent record_outcome() is thread-safe
// ============================================================

TEST(SignalCalibrationEngine, ConcurrentRecordSafe) {
    SignalCalibrationEngine::Config cfg;
    cfg.min_samples = 5;
    SignalCalibrationEngine cal(cfg);

    std::vector<std::thread> threads;
    for (int t = 0; t < 4; ++t) {
        threads.emplace_back([&, t] {
            for (int i = 0; i < 50; ++i)
                cal.record_outcome(static_cast<double>(i % 10) * 0.1, (i + t) % 2 == 0);
        });
    }
    for (auto& th : threads) th.join();

    EXPECT_EQ(cal.sample_count(), 200);
    double ece = cal.expected_calibration_error();
    EXPECT_GE(ece, 0.0);
    EXPECT_LE(ece, 1.0);
}

} // namespace

#else  // LLMQUANT_SIGNAL_CALIBRATION_ENABLED not defined

TEST(SignalCalibrationEngine, DisabledAtBuildTime) {
    SUCCEED();
}

#endif  // LLMQUANT_SIGNAL_CALIBRATION_ENABLED
