#include <gtest/gtest.h>

#ifdef LLMQUANT_SIGNAL_DRIFT_ENABLED

#include "SignalDriftMonitor.h"

#include <atomic>
#include <cmath>
#include <thread>
#include <vector>

using namespace llmquant;

namespace {

// ---------------------------------------------------------------------------
// Default state
// ---------------------------------------------------------------------------

TEST(SignalDriftMonitor, DefaultState) {
    SignalDriftMonitor sdm;
    EXPECT_EQ(sdm.total_records(), 0u);
    EXPECT_EQ(sdm.drift_events(), 0u);
    EXPECT_FALSE(sdm.is_drifting());
    EXPECT_DOUBLE_EQ(sdm.last_w1(), 0.0);
}

// ---------------------------------------------------------------------------
// Records increment total
// ---------------------------------------------------------------------------

TEST(SignalDriftMonitor, TotalRecordsIncrements) {
    SignalDriftMonitor sdm;
    sdm.record(0.01);
    sdm.record(0.02);
    sdm.record(-0.01);
    EXPECT_EQ(sdm.total_records(), 3u);
}

// ---------------------------------------------------------------------------
// Stable distribution → W1 near zero, no drift
// ---------------------------------------------------------------------------

TEST(SignalDriftMonitor, StableDistributionNoDrift) {
    SignalDriftMonitor::Config cfg;
    cfg.baseline_size   = 64;
    cfg.recent_size     = 16;
    cfg.drift_threshold = 0.20;
    SignalDriftMonitor sdm(cfg);

    // Feed identical distribution to both windows.
    for (int i = 0; i < 120; ++i)
        sdm.record(0.05 * std::sin(static_cast<double>(i) * 0.3));

    EXPECT_FALSE(sdm.is_drifting());
    EXPECT_LT(sdm.last_w1(), cfg.drift_threshold);
}

// ---------------------------------------------------------------------------
// Large distribution shift → drift detected
// ---------------------------------------------------------------------------

TEST(SignalDriftMonitor, LargeShiftTriggersDrift) {
    SignalDriftMonitor::Config cfg;
    cfg.baseline_size   = 64;
    cfg.recent_size     = 16;
    cfg.drift_threshold = 0.05;
    std::atomic<int> fires{0};
    cfg.on_drift_detected = [&](double) { ++fires; };
    SignalDriftMonitor sdm(cfg);

    // Baseline: near-zero values.
    for (int i = 0; i < 80; ++i) sdm.record(0.001);
    // Shift: large positive values.
    for (int i = 0; i < 20; ++i) sdm.record(0.5);

    EXPECT_TRUE(sdm.is_drifting());
    EXPECT_GT(fires.load(), 0);
    EXPECT_GT(sdm.drift_events(), 0u);
}

// ---------------------------------------------------------------------------
// Drift clears after distribution returns to baseline
// ---------------------------------------------------------------------------

TEST(SignalDriftMonitor, DriftClears) {
    SignalDriftMonitor::Config cfg;
    cfg.baseline_size   = 32;
    cfg.recent_size     = 8;
    cfg.drift_threshold = 0.05;
    cfg.clear_hysteresis= 0.5;
    std::atomic<int> cleared{0};
    cfg.on_drift_cleared = [&](double) { ++cleared; };
    SignalDriftMonitor sdm(cfg);

    for (int i = 0; i < 40; ++i) sdm.record(0.001);
    // Cause drift.
    for (int i = 0; i < 10; ++i) sdm.record(1.0);
    // Return to baseline: need ≥32 records to push all 10 drift records out of
    // the 32-entry baseline window so W1 drops back below the clear threshold.
    for (int i = 0; i < 45; ++i) sdm.record(0.001);

    EXPECT_GT(cleared.load(), 0);
}

// ---------------------------------------------------------------------------
// Wasserstein distance is non-negative
// ---------------------------------------------------------------------------

TEST(SignalDriftMonitor, WassersteinNonNegative) {
    SignalDriftMonitor::Config cfg;
    cfg.baseline_size = 32;
    cfg.recent_size   = 8;
    SignalDriftMonitor sdm(cfg);
    for (int i = 0; i < 50; ++i)
        sdm.record(static_cast<double>(i % 10) * 0.1 - 0.5);
    EXPECT_GE(sdm.wasserstein_distance(), 0.0);
}

// ---------------------------------------------------------------------------
// baseline_mean and recent_mean are finite
// ---------------------------------------------------------------------------

TEST(SignalDriftMonitor, MeansFinite) {
    SignalDriftMonitor::Config cfg;
    cfg.baseline_size = 32;
    cfg.recent_size   = 8;
    SignalDriftMonitor sdm(cfg);
    for (int i = 0; i < 50; ++i)
        sdm.record(static_cast<double>(i % 5) * 0.2 - 0.4);
    EXPECT_TRUE(std::isfinite(sdm.baseline_mean()));
    EXPECT_TRUE(std::isfinite(sdm.recent_mean()));
}

// ---------------------------------------------------------------------------
// Reset clears all state
// ---------------------------------------------------------------------------

TEST(SignalDriftMonitor, ResetClearsState) {
    SignalDriftMonitor::Config cfg;
    cfg.baseline_size   = 32;
    cfg.recent_size     = 8;
    cfg.drift_threshold = 0.01;
    SignalDriftMonitor sdm(cfg);
    for (int i = 0; i < 50; ++i) sdm.record(static_cast<double>(i));
    sdm.reset();
    EXPECT_EQ(sdm.total_records(), 0u);
    EXPECT_EQ(sdm.drift_events(), 0u);
    EXPECT_FALSE(sdm.is_drifting());
    EXPECT_DOUBLE_EQ(sdm.last_w1(), 0.0);
}

// ---------------------------------------------------------------------------
// update_config resets state
// ---------------------------------------------------------------------------

TEST(SignalDriftMonitor, UpdateConfigResetsState) {
    SignalDriftMonitor sdm;
    for (int i = 0; i < 30; ++i) sdm.record(0.1);
    SignalDriftMonitor::Config cfg2;
    cfg2.baseline_size = 16;
    cfg2.recent_size   = 4;
    sdm.update_config(cfg2);
    EXPECT_EQ(sdm.total_records(), 0u);
    EXPECT_FALSE(sdm.is_drifting());
}

// ---------------------------------------------------------------------------
// to_stats_json contains expected fields
// ---------------------------------------------------------------------------

TEST(SignalDriftMonitor, ToStatsJsonContainsFields) {
    SignalDriftMonitor sdm;
    sdm.record(0.1);
    std::string j = sdm.to_stats_json();
    EXPECT_NE(j.find("wasserstein_1"),  std::string::npos);
    EXPECT_NE(j.find("is_drifting"),    std::string::npos);
    EXPECT_NE(j.find("drift_events"),   std::string::npos);
    EXPECT_NE(j.find("total_records"),  std::string::npos);
}

// ---------------------------------------------------------------------------
// Concurrent record() is thread-safe
// ---------------------------------------------------------------------------

TEST(SignalDriftMonitor, ConcurrentRecordSafe) {
    SignalDriftMonitor::Config cfg;
    cfg.baseline_size = 64;
    cfg.recent_size   = 16;
    SignalDriftMonitor sdm(cfg);

    std::atomic<bool> go{false};
    auto worker = [&] {
        while (!go.load()) {}
        for (int i = 0; i < 250; ++i)
            sdm.record(static_cast<double>(i % 7) * 0.1 - 0.3);
    };
    std::vector<std::thread> threads;
    for (int t = 0; t < 4; ++t) threads.emplace_back(worker);
    go.store(true);
    for (auto& t : threads) t.join();

    EXPECT_EQ(sdm.total_records(), 1000u);
}

}  // namespace

#else

TEST(SignalDriftMonitor, DisabledAtBuildTime) { SUCCEED(); }

#endif  // LLMQUANT_SIGNAL_DRIFT_ENABLED
