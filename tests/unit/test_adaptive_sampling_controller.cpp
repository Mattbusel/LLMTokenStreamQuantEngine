#include <gtest/gtest.h>

#ifdef LLMQUANT_ADAPTIVE_SAMPLING_ENABLED

#include "AdaptiveSamplingController.h"

using namespace llmquant;

namespace {

// ============================================================
// Construction
// ============================================================

TEST(AdaptiveSamplingController, DefaultConstructsWithInitialInterval) {
    AdaptiveSamplingController::Config cfg;
    cfg.initial_interval_ms = 200;
    AdaptiveSamplingController ctrl(cfg);
    EXPECT_EQ(ctrl.recommended_interval_ms(), 200);
    EXPECT_EQ(ctrl.total_records(), 0u);
}

// ============================================================
// High activity shrinks interval
// ============================================================

TEST(AdaptiveSamplingController, HighActivityShrinksInterval) {
    AdaptiveSamplingController::Config cfg;
    cfg.initial_interval_ms     = 1000;
    cfg.min_interval_ms         = 10;
    cfg.high_activity_threshold = 0.01;
    cfg.decay_factor            = 0.5;
    AdaptiveSamplingController ctrl(cfg);

    ctrl.record_activity(0.5);  // above threshold
    EXPECT_LT(ctrl.recommended_interval_ms(), 1000);
    EXPECT_EQ(ctrl.accelerations(), 1u);
}

// ============================================================
// Low activity grows interval
// ============================================================

TEST(AdaptiveSamplingController, LowActivityGrowsInterval) {
    AdaptiveSamplingController::Config cfg;
    cfg.initial_interval_ms    = 100;
    cfg.max_interval_ms        = 5000;
    cfg.low_activity_threshold = 0.001;
    cfg.growth_factor          = 2.0;
    AdaptiveSamplingController ctrl(cfg);

    ctrl.record_activity(0.0);  // below threshold
    EXPECT_GT(ctrl.recommended_interval_ms(), 100);
    EXPECT_EQ(ctrl.decelerations(), 1u);
}

// ============================================================
// Interval clamped at min
// ============================================================

TEST(AdaptiveSamplingController, IntervalClampedAtMin) {
    AdaptiveSamplingController::Config cfg;
    cfg.initial_interval_ms     = 10;
    cfg.min_interval_ms         = 10;
    cfg.high_activity_threshold = 0.0001;
    cfg.decay_factor            = 0.1;
    AdaptiveSamplingController ctrl(cfg);

    for (int i = 0; i < 10; ++i) ctrl.record_activity(1.0);
    EXPECT_EQ(ctrl.recommended_interval_ms(), cfg.min_interval_ms);
    EXPECT_TRUE(ctrl.is_at_min());
}

// ============================================================
// Interval clamped at max
// ============================================================

TEST(AdaptiveSamplingController, IntervalClampedAtMax) {
    AdaptiveSamplingController::Config cfg;
    cfg.initial_interval_ms    = 5000;
    cfg.max_interval_ms        = 5000;
    cfg.low_activity_threshold = 1.0;
    cfg.growth_factor          = 10.0;
    AdaptiveSamplingController ctrl(cfg);

    ctrl.record_activity(0.0);
    EXPECT_EQ(ctrl.recommended_interval_ms(), cfg.max_interval_ms);
    EXPECT_TRUE(ctrl.is_at_max());
}

// ============================================================
// Middle activity — no change
// ============================================================

TEST(AdaptiveSamplingController, MiddleActivityNoChange) {
    AdaptiveSamplingController::Config cfg;
    cfg.initial_interval_ms     = 200;
    cfg.high_activity_threshold = 0.1;
    cfg.low_activity_threshold  = 0.01;
    AdaptiveSamplingController ctrl(cfg);

    ctrl.record_activity(0.05);  // in the neutral band
    EXPECT_EQ(ctrl.recommended_interval_ms(), 200);
    EXPECT_EQ(ctrl.accelerations(), 0u);
    EXPECT_EQ(ctrl.decelerations(), 0u);
}

// ============================================================
// on_interval_change callback fires
// ============================================================

TEST(AdaptiveSamplingController, CallbackFiresOnChange) {
    AdaptiveSamplingController::Config cfg;
    cfg.initial_interval_ms     = 1000;
    cfg.min_interval_ms         = 10;
    cfg.high_activity_threshold = 0.001;
    cfg.decay_factor            = 0.5;
    int fires = 0;
    cfg.on_interval_change = [&](int64_t) { ++fires; };
    AdaptiveSamplingController ctrl(cfg);

    ctrl.record_activity(1.0);
    EXPECT_GE(fires, 1);
}

// ============================================================
// total_records counts all calls
// ============================================================

TEST(AdaptiveSamplingController, TotalRecordsCounted) {
    AdaptiveSamplingController ctrl;
    for (int i = 0; i < 7; ++i) ctrl.record_activity(0.5);
    EXPECT_EQ(ctrl.total_records(), 7u);
}

// ============================================================
// reset restores initial interval
// ============================================================

TEST(AdaptiveSamplingController, ResetRestoresInitialInterval) {
    AdaptiveSamplingController::Config cfg;
    cfg.initial_interval_ms     = 500;
    cfg.min_interval_ms         = 10;
    cfg.high_activity_threshold = 0.001;
    cfg.decay_factor            = 0.1;
    AdaptiveSamplingController ctrl(cfg);

    for (int i = 0; i < 20; ++i) ctrl.record_activity(1.0);
    EXPECT_LT(ctrl.recommended_interval_ms(), 500);
    ctrl.reset();
    EXPECT_EQ(ctrl.recommended_interval_ms(), 500);
}

// ============================================================
// update_config resets to new initial
// ============================================================

TEST(AdaptiveSamplingController, UpdateConfigSetsNewInitial) {
    AdaptiveSamplingController ctrl;
    ctrl.record_activity(1.0);
    AdaptiveSamplingController::Config cfg2;
    cfg2.initial_interval_ms = 250;
    ctrl.update_config(cfg2);
    EXPECT_EQ(ctrl.recommended_interval_ms(), 250);
}

// ============================================================
// to_stats_json contains required keys
// ============================================================

TEST(AdaptiveSamplingController, ToStatsJsonContainsKeys) {
    AdaptiveSamplingController ctrl;
    ctrl.record_activity(0.5);
    std::string json = ctrl.to_stats_json();
    EXPECT_NE(json.find("interval_ms"),     std::string::npos);
    EXPECT_NE(json.find("min_interval_ms"), std::string::npos);
    EXPECT_NE(json.find("max_interval_ms"), std::string::npos);
    EXPECT_NE(json.find("total_records"),   std::string::npos);
    EXPECT_NE(json.find("accelerations"),   std::string::npos);
    EXPECT_NE(json.find("decelerations"),   std::string::npos);
}

} // namespace

#else  // LLMQUANT_ADAPTIVE_SAMPLING_ENABLED not defined

TEST(AdaptiveSamplingController, DisabledAtBuildTime) {
    SUCCEED();
}

#endif  // LLMQUANT_ADAPTIVE_SAMPLING_ENABLED
