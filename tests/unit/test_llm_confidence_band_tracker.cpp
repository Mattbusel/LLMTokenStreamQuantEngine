#include <gtest/gtest.h>

#ifdef LLMQUANT_CONFIDENCE_BAND_ENABLED

#include "LLMConfidenceBandTracker.h"

#include <atomic>
#include <cmath>
#include <thread>
#include <vector>

using namespace llmquant;

namespace {

// ============================================================
// Default state
// ============================================================

TEST(LLMConfidenceBandTracker, DefaultStateIsZero) {
    LLMConfidenceBandTracker trk;
    EXPECT_DOUBLE_EQ(trk.center(), 0.0);
    EXPECT_EQ(trk.observation_count(), 0u);
    EXPECT_EQ(trk.band_change_count(), 0u);
}

// ============================================================
// Single update seeds the filter
// ============================================================

TEST(LLMConfidenceBandTracker, SingleUpdateSeedsCenter) {
    LLMConfidenceBandTracker trk;
    trk.update(0.5);
    // After one update the center should be close to the observation.
    EXPECT_NEAR(trk.center(), 0.5, 0.05);
    EXPECT_EQ(trk.observation_count(), 1u);
}

// ============================================================
// Band is symmetric around center
// ============================================================

TEST(LLMConfidenceBandTracker, BandSymmetricAroundCenter) {
    LLMConfidenceBandTracker trk;
    trk.update(1.0);
    double c  = trk.center();
    double hw = trk.half_width();
    EXPECT_NEAR(trk.lower(), c - hw, 1e-9);
    EXPECT_NEAR(trk.upper(), c + hw, 1e-9);
    EXPECT_GT(hw, 0.0);
}

// ============================================================
// Repeated identical observations — center converges, band narrows
// ============================================================

TEST(LLMConfidenceBandTracker, CenterConvergesOnConstantInput) {
    LLMConfidenceBandTracker trk;
    for (int i = 0; i < 100; ++i) trk.update(0.8);
    EXPECT_NEAR(trk.center(), 0.8, 0.01);
    // Band should have narrowed after many consistent observations.
    // Steady-state half_width = z * sqrt((-R + sqrt(R²+4QR))/2) ≈ 0.112 with defaults.
    EXPECT_LT(trk.half_width(), 0.15);
}

// ============================================================
// Variance decreases with consistent observations
// ============================================================

TEST(LLMConfidenceBandTracker, VarianceDecreasesWithConsistentObs) {
    LLMConfidenceBandTracker trk;
    trk.update(0.0);
    double v0 = trk.variance();
    for (int i = 0; i < 50; ++i) trk.update(0.0);
    EXPECT_LT(trk.variance(), v0);
}

// ============================================================
// Noisy observations — center tracks a very different path than stable
// (Kalman variance is observation-independent — it depends only on Q,R)
// ============================================================

TEST(LLMConfidenceBandTracker, NoisyObsProduceDifferentCenter) {
    LLMConfidenceBandTracker::Config cfg;
    cfg.process_noise     = 0.01;
    cfg.measurement_noise = 0.01;
    LLMConfidenceBandTracker trk(cfg);

    // Alternating extremes: center should hover near 0 after many oscillations.
    for (int i = 0; i < 50; ++i)
        trk.update(i % 2 == 0 ? 1.0 : -1.0);

    LLMConfidenceBandTracker trk_stable(cfg);
    for (int i = 0; i < 50; ++i) trk_stable.update(0.8);

    // Stable tracker's center should be clearly positive; noisy one near 0.
    EXPECT_GT(trk_stable.center(), 0.5);
    EXPECT_LT(std::abs(trk.center()), 0.5);
}

// ============================================================
// on_band_change callback fires
// ============================================================

TEST(LLMConfidenceBandTracker, CallbackFires) {
    LLMConfidenceBandTracker::Config cfg;
    cfg.band_change_threshold = 0.01;  // fire on small changes
    std::atomic<int> fires{0};
    cfg.on_band_change = [&](double, double, double) { ++fires; };
    LLMConfidenceBandTracker trk(cfg);

    for (int i = 0; i < 20; ++i) trk.update(static_cast<double>(i) * 0.1);
    EXPECT_GT(fires.load(), 0);
    EXPECT_GT(trk.band_change_count(), 0u);
}

// ============================================================
// reset() restores initial state
// ============================================================

TEST(LLMConfidenceBandTracker, ResetRestoresState) {
    LLMConfidenceBandTracker trk;
    for (int i = 0; i < 30; ++i) trk.update(1.0);
    EXPECT_GT(trk.observation_count(), 0u);

    trk.reset();
    EXPECT_EQ(trk.observation_count(), 0u);
    EXPECT_EQ(trk.band_change_count(), 0u);
    EXPECT_DOUBLE_EQ(trk.center(), 0.0);
}

// ============================================================
// update_config() resets state
// ============================================================

TEST(LLMConfidenceBandTracker, UpdateConfigResetsFilterState) {
    LLMConfidenceBandTracker trk;
    for (int i = 0; i < 20; ++i) trk.update(0.5);
    // Center should be close to 0.5 now.
    EXPECT_GT(trk.center(), 0.3);

    // update_config resets the filter; next update should re-seed from 0.
    LLMConfidenceBandTracker::Config cfg2;
    cfg2.process_noise = 0.01;
    trk.update_config(cfg2);
    // After update_config the Kalman state is reset to 0; the first new
    // observation seeds x to that value, so center should shift away from 0.5.
    trk.update(-0.9);
    EXPECT_LT(trk.center(), 0.0);
}

// ============================================================
// to_stats_json contains required keys
// ============================================================

TEST(LLMConfidenceBandTracker, ToStatsJsonContainsKeys) {
    LLMConfidenceBandTracker trk;
    trk.update(0.3);
    std::string json = trk.to_stats_json();
    EXPECT_NE(json.find("center"),           std::string::npos);
    EXPECT_NE(json.find("lower"),            std::string::npos);
    EXPECT_NE(json.find("upper"),            std::string::npos);
    EXPECT_NE(json.find("half_width"),       std::string::npos);
    EXPECT_NE(json.find("variance"),         std::string::npos);
    EXPECT_NE(json.find("observation_count"), std::string::npos);
    EXPECT_NE(json.find("band_change_count"), std::string::npos);
}

// ============================================================
// z_score scales band width proportionally
// ============================================================

TEST(LLMConfidenceBandTracker, ZScoreScalesBandWidth) {
    LLMConfidenceBandTracker::Config cfg1, cfg2;
    cfg1.z_score = 1.0;
    cfg2.z_score = 2.0;
    LLMConfidenceBandTracker t1(cfg1), t2(cfg2);
    t1.update(0.0);
    t2.update(0.0);
    EXPECT_NEAR(t2.half_width(), 2.0 * t1.half_width(), 1e-9);
}

// ============================================================
// Concurrent update() is thread-safe
// ============================================================

TEST(LLMConfidenceBandTracker, ConcurrentUpdateSafe) {
    LLMConfidenceBandTracker trk;
    std::vector<std::thread> threads;
    for (int t = 0; t < 4; ++t) {
        threads.emplace_back([&, t] {
            for (int i = 0; i < 100; ++i)
                trk.update(static_cast<double>(i + t) * 0.01);
        });
    }
    for (auto& th : threads) th.join();
    EXPECT_EQ(trk.observation_count(), 400u);
    // Variance and center should be finite positive values.
    EXPECT_GT(trk.variance(), 0.0);
    EXPECT_TRUE(std::isfinite(trk.center()));
}

} // namespace

#else

TEST(LLMConfidenceBandTracker, DisabledAtBuildTime) { SUCCEED(); }

#endif  // LLMQUANT_CONFIDENCE_BAND_ENABLED
