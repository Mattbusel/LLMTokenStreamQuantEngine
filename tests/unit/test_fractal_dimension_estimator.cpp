#include <gtest/gtest.h>

#ifdef LLMQUANT_FRACTAL_DIMENSION_ENABLED

#include "FractalDimensionEstimator.h"

#include <cmath>
#include <numeric>
#include <vector>

using namespace llmquant;

namespace {

// ============================================================
// Construction
// ============================================================

TEST(FractalDimensionEstimator, DefaultConstructsWithHalfHurst) {
    FractalDimensionEstimator est;
    EXPECT_NEAR(est.hurst(), 0.5, 1e-9);
    EXPECT_EQ(est.sample_count(), 0);
    EXPECT_EQ(est.total_records(), 0u);
}

// ============================================================
// Returns 0.5 before min_samples
// ============================================================

TEST(FractalDimensionEstimator, HalfBeforeMinSamples) {
    FractalDimensionEstimator::Config cfg;
    cfg.min_samples = 10;
    cfg.window_size = 64;
    FractalDimensionEstimator est(cfg);

    for (int i = 0; i < 5; ++i) est.record(static_cast<double>(i));
    EXPECT_NEAR(est.hurst(), 0.5, 1e-9);
}

// ============================================================
// Trending series → H > 0.5
// ============================================================

TEST(FractalDimensionEstimator, TrendingSeriesHighHurst) {
    FractalDimensionEstimator::Config cfg;
    cfg.window_size = 64;
    cfg.min_samples = 32;
    FractalDimensionEstimator est(cfg);

    // Monotonically increasing — very persistent.
    for (int i = 0; i < 64; ++i) est.record(static_cast<double>(i));
    EXPECT_GT(est.hurst(), 0.5);
    EXPECT_TRUE(est.is_trending());
    EXPECT_FALSE(est.is_mean_reverting());
}

// ============================================================
// Alternating series → H < 0.5
// ============================================================

TEST(FractalDimensionEstimator, AlternatingSeriesLowHurst) {
    FractalDimensionEstimator::Config cfg;
    cfg.window_size = 64;
    cfg.min_samples = 32;
    FractalDimensionEstimator est(cfg);

    // Perfectly alternating +1/-1 — maximally anti-persistent.
    for (int i = 0; i < 64; ++i)
        est.record(i % 2 == 0 ? 1.0 : -1.0);
    EXPECT_LT(est.hurst(), 0.5);
    EXPECT_TRUE(est.is_mean_reverting());
    EXPECT_FALSE(est.is_trending());
}

// ============================================================
// Hurst clamped to (0, 1)
// ============================================================

TEST(FractalDimensionEstimator, HurstClamped) {
    FractalDimensionEstimator::Config cfg;
    cfg.window_size = 64;
    cfg.min_samples = 32;
    FractalDimensionEstimator est(cfg);
    for (int i = 0; i < 64; ++i) est.record(static_cast<double>(i) * 1000.0);
    EXPECT_GE(est.hurst(), 0.0);
    EXPECT_LE(est.hurst(), 1.0);
}

// ============================================================
// Window cap enforced
// ============================================================

TEST(FractalDimensionEstimator, WindowCapEnforced) {
    FractalDimensionEstimator::Config cfg;
    cfg.window_size = 32;
    cfg.min_samples = 10;
    FractalDimensionEstimator est(cfg);
    for (int i = 0; i < 100; ++i) est.record(1.0);
    EXPECT_EQ(est.sample_count(), 32);
}

// ============================================================
// total_records counts all calls
// ============================================================

TEST(FractalDimensionEstimator, TotalRecordsCounted) {
    FractalDimensionEstimator est;
    for (int i = 0; i < 17; ++i) est.record(1.0);
    EXPECT_EQ(est.total_records(), 17u);
}

// ============================================================
// reset
// ============================================================

TEST(FractalDimensionEstimator, ResetClearsWindow) {
    FractalDimensionEstimator::Config cfg;
    cfg.window_size = 64;
    cfg.min_samples = 32;
    FractalDimensionEstimator est(cfg);
    for (int i = 0; i < 64; ++i) est.record(static_cast<double>(i));
    est.reset();
    EXPECT_EQ(est.sample_count(), 0);
    EXPECT_NEAR(est.hurst(), 0.5, 1e-9);
}

// ============================================================
// Regime-change callback fires
// ============================================================

TEST(FractalDimensionEstimator, RegimeChangeCallbackFires) {
    FractalDimensionEstimator::Config cfg;
    cfg.window_size = 32;
    cfg.min_samples = 16;

    int changes = 0;
    cfg.on_regime_change = [&](double, double) { ++changes; };
    FractalDimensionEstimator est(cfg);

    // Trending phase.
    for (int i = 0; i < 32; ++i) est.record(static_cast<double>(i));
    // Switch to anti-persistent.
    for (int i = 0; i < 32; ++i) est.record(i % 2 == 0 ? 1.0 : -1.0);

    EXPECT_GE(est.regime_changes(), 1u);
    EXPECT_GE(changes, 1);
}

// ============================================================
// update_config resets
// ============================================================

TEST(FractalDimensionEstimator, UpdateConfigResetsState) {
    FractalDimensionEstimator est;
    for (int i = 0; i < 64; ++i) est.record(static_cast<double>(i));
    FractalDimensionEstimator::Config cfg2;
    cfg2.window_size = 32;
    cfg2.min_samples = 8;
    est.update_config(cfg2);
    EXPECT_EQ(est.sample_count(), 0);
    EXPECT_NEAR(est.hurst(), 0.5, 1e-9);
}

// ============================================================
// rs_full is non-negative after data
// ============================================================

TEST(FractalDimensionEstimator, RsFullNonNegative) {
    FractalDimensionEstimator::Config cfg;
    cfg.window_size = 64;
    cfg.min_samples = 32;
    FractalDimensionEstimator est(cfg);
    for (int i = 0; i < 64; ++i) est.record(static_cast<double>(i));
    EXPECT_GE(est.rs_full(), 0.0);
}

// ============================================================
// Constant series (stddev == 0) doesn't crash
// ============================================================

TEST(FractalDimensionEstimator, ConstantSeriesNoCrash) {
    FractalDimensionEstimator::Config cfg;
    cfg.window_size = 32;
    cfg.min_samples = 16;
    FractalDimensionEstimator est(cfg);
    for (int i = 0; i < 32; ++i) est.record(1.0);
    // S = 0, so R/S = 0; hurst stays at 0.5.
    EXPECT_NEAR(est.hurst(), 0.5, 1e-9);
}

// ============================================================
// to_stats_json contains required keys
// ============================================================

TEST(FractalDimensionEstimator, ToStatsJsonContainsKeys) {
    FractalDimensionEstimator::Config cfg;
    cfg.window_size = 64;
    cfg.min_samples = 32;
    FractalDimensionEstimator est(cfg);
    for (int i = 0; i < 64; ++i) est.record(static_cast<double>(i));
    std::string json = est.to_stats_json();
    EXPECT_NE(json.find("hurst"),           std::string::npos);
    EXPECT_NE(json.find("rs_full"),         std::string::npos);
    EXPECT_NE(json.find("sample_count"),    std::string::npos);
    EXPECT_NE(json.find("total_records"),   std::string::npos);
    EXPECT_NE(json.find("regime_changes"),  std::string::npos);
    EXPECT_NE(json.find("is_trending"),     std::string::npos);
}

} // namespace

#else  // LLMQUANT_FRACTAL_DIM_ENABLED not defined

TEST(FractalDimensionEstimator, DisabledAtBuildTime) {
    SUCCEED();
}

#endif  // LLMQUANT_FRACTAL_DIMENSION_ENABLED
