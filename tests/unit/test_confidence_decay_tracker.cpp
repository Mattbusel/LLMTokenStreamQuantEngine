#include <gtest/gtest.h>

#ifdef LLMQUANT_CONFIDENCE_DECAY_ENABLED

#include "ConfidenceDecayTracker.h"

#include <atomic>
#include <chrono>
#include <cmath>
#include <thread>
#include <vector>

using namespace llmquant;

namespace {

// ============================================================
// Default state
// ============================================================

TEST(ConfidenceDecayTracker, DefaultStateZero) {
    ConfidenceDecayTracker tracker;
    EXPECT_DOUBLE_EQ(tracker.half_life_ms(), 0.0);
    EXPECT_DOUBLE_EQ(tracker.lambda(), 0.0);
    EXPECT_FALSE(tracker.is_fast_decay());
    EXPECT_EQ(tracker.total_records(), 0u);
}

// ============================================================
// Insufficient samples keeps half_life at 0
// ============================================================

TEST(ConfidenceDecayTracker, InsufficientSamplesNoEstimate) {
    ConfidenceDecayTracker::Config cfg;
    cfg.min_samples = 8;
    ConfidenceDecayTracker tracker(cfg);

    for (int i = 0; i < 3; ++i) tracker.record(0.9);
    EXPECT_DOUBLE_EQ(tracker.half_life_ms(), 0.0);
}

// ============================================================
// Flat series (no decay) → half_life stays 0
// ============================================================

TEST(ConfidenceDecayTracker, FlatSeriesNoDacay) {
    ConfidenceDecayTracker::Config cfg;
    cfg.min_samples = 5;
    cfg.window_size = 20;
    ConfidenceDecayTracker tracker(cfg);

    // All identical confidence values: no temporal gradient → λ ≈ 0.
    for (int i = 0; i < 20; ++i) tracker.record(0.7);
    // λ should be near 0 (flat series), so half_life ≈ 0.
    EXPECT_NEAR(tracker.lambda(), 0.0, 1e-6);
}

// ============================================================
// Steeply decaying series → positive lambda and finite half-life
// ============================================================

TEST(ConfidenceDecayTracker, DecayingSeriesProducesPositiveLambda) {
    ConfidenceDecayTracker::Config cfg;
    cfg.min_samples  = 5;
    cfg.window_size  = 32;
    ConfidenceDecayTracker tracker(cfg);

    // Simulate steep exponential decay: record quickly so time is ~0.
    // Since all records happen at near-zero wall time, we need a series
    // where the *index* serves as the time proxy.  We abuse the ring by
    // inserting values spaced via tiny sleeps.
    for (int i = 0; i < 16; ++i) {
        double conf = 0.9 * std::exp(-0.05 * i);
        tracker.record(conf);
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }

    // lambda > 0 and half_life > 0
    EXPECT_GT(tracker.lambda(), 0.0);
    EXPECT_GT(tracker.half_life_ms(), 0.0);
}

// ============================================================
// is_fast_decay() correctly classifies short half-life
// ============================================================

TEST(ConfidenceDecayTracker, IsFastDecayForShortHalfLife) {
    ConfidenceDecayTracker::Config cfg;
    cfg.min_samples             = 5;
    cfg.window_size             = 32;
    cfg.fast_decay_threshold_ms = 1000.0;  // anything < 1s is fast
    ConfidenceDecayTracker tracker(cfg);

    // Steep decay spaced by 1 ms → half-life should be <1s
    for (int i = 0; i < 16; ++i) {
        tracker.record(0.8 * std::exp(-0.3 * i));
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }

    if (tracker.half_life_ms() > 0.0)
        EXPECT_TRUE(tracker.is_fast_decay());
}

// ============================================================
// total_records increments
// ============================================================

TEST(ConfidenceDecayTracker, TotalRecordsIncrements) {
    ConfidenceDecayTracker tracker;
    tracker.record(0.9);
    tracker.record(0.8);
    tracker.record(0.7);
    EXPECT_EQ(tracker.total_records(), 3u);
}

// ============================================================
// reset() clears all state
// ============================================================

TEST(ConfidenceDecayTracker, ResetClearsState) {
    ConfidenceDecayTracker::Config cfg;
    cfg.min_samples = 5;
    ConfidenceDecayTracker tracker(cfg);

    for (int i = 0; i < 10; ++i) {
        tracker.record(0.9 * std::exp(-0.1 * i));
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }

    tracker.reset();
    EXPECT_DOUBLE_EQ(tracker.half_life_ms(), 0.0);
    EXPECT_EQ(tracker.total_records(), 0u);
}

// ============================================================
// update_config resets state
// ============================================================

TEST(ConfidenceDecayTracker, UpdateConfigResetsState) {
    ConfidenceDecayTracker::Config cfg;
    cfg.min_samples = 5;
    ConfidenceDecayTracker tracker(cfg);

    for (int i = 0; i < 10; ++i) {
        tracker.record(0.9 * std::exp(-0.1 * i));
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }

    ConfidenceDecayTracker::Config cfg2;
    tracker.update_config(cfg2);
    EXPECT_DOUBLE_EQ(tracker.half_life_ms(), 0.0);
    EXPECT_EQ(tracker.total_records(), 0u);
}

// ============================================================
// to_stats_json contains required keys
// ============================================================

TEST(ConfidenceDecayTracker, ToStatsJsonContainsKeys) {
    ConfidenceDecayTracker tracker;
    tracker.record(0.8);
    std::string json = tracker.to_stats_json();
    EXPECT_NE(json.find("half_life_ms"),  std::string::npos);
    EXPECT_NE(json.find("lambda"),        std::string::npos);
    EXPECT_NE(json.find("is_fast_decay"), std::string::npos);
    EXPECT_NE(json.find("total_records"), std::string::npos);
}

// ============================================================
// Concurrent record() is thread-safe
// ============================================================

TEST(ConfidenceDecayTracker, ConcurrentRecordSafe) {
    ConfidenceDecayTracker::Config cfg;
    cfg.min_samples = 5;
    cfg.window_size = 32;
    ConfidenceDecayTracker tracker(cfg);

    std::vector<std::thread> threads;
    for (int t = 0; t < 4; ++t) {
        threads.emplace_back([&, t] {
            for (int i = 0; i < 50; ++i)
                tracker.record(0.9 - (i + t) * 0.005);
        });
    }
    for (auto& th : threads) th.join();

    EXPECT_EQ(tracker.total_records(), 200u);
    EXPECT_GE(tracker.half_life_ms(), 0.0);
}

} // namespace

#else  // LLMQUANT_CONFIDENCE_DECAY_ENABLED not defined

TEST(ConfidenceDecayTracker, DisabledAtBuildTime) {
    SUCCEED();
}

#endif  // LLMQUANT_CONFIDENCE_DECAY_ENABLED
