#include <gtest/gtest.h>

#ifdef LLMQUANT_VELOCITY_TRACKER_ENABLED

#include "TokenVelocityTracker.h"

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

TEST(TokenVelocityTracker, DefaultStateZero) {
    TokenVelocityTracker tracker;
    EXPECT_DOUBLE_EQ(tracker.velocity(), 0.0);
    EXPECT_DOUBLE_EQ(tracker.acceleration(), 0.0);
    EXPECT_FALSE(tracker.is_fast_move());
    EXPECT_EQ(tracker.total_records(), 0u);
}

// ============================================================
// Insufficient samples keeps velocity at 0
// ============================================================

TEST(TokenVelocityTracker, InsufficientSamplesZeroVelocity) {
    TokenVelocityTracker::Config cfg;
    cfg.min_samples = 4;
    TokenVelocityTracker tracker(cfg);

    tracker.record(0.5);
    tracker.record(0.6);
    EXPECT_DOUBLE_EQ(tracker.velocity(), 0.0);
}

// ============================================================
// Rising series → positive velocity
// ============================================================

TEST(TokenVelocityTracker, RisingBiasPositiveVelocity) {
    TokenVelocityTracker::Config cfg;
    cfg.min_samples         = 4;
    cfg.window_size         = 8;
    cfg.fast_move_threshold = 100.0;  // disable for this test
    TokenVelocityTracker tracker(cfg);

    for (int i = 0; i < 8; ++i) {
        tracker.record(static_cast<double>(i) * 0.1);
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }

    EXPECT_GT(tracker.velocity(), 0.0);
}

// ============================================================
// Falling series → negative velocity
// ============================================================

TEST(TokenVelocityTracker, FallingBiasNegativeVelocity) {
    TokenVelocityTracker::Config cfg;
    cfg.min_samples         = 4;
    cfg.window_size         = 8;
    cfg.fast_move_threshold = 100.0;
    TokenVelocityTracker tracker(cfg);

    for (int i = 0; i < 8; ++i) {
        tracker.record(1.0 - static_cast<double>(i) * 0.1);
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }

    EXPECT_LT(tracker.velocity(), 0.0);
}

// ============================================================
// is_fast_move fires callback when velocity exceeds threshold
// ============================================================

TEST(TokenVelocityTracker, FastMoveCallbackFires) {
    TokenVelocityTracker::Config cfg;
    cfg.min_samples         = 4;
    cfg.window_size         = 8;
    cfg.fast_move_threshold = 0.001;  // very low threshold
    std::atomic<int> fires{0};
    cfg.on_fast_move = [&](double, double) { ++fires; };
    TokenVelocityTracker tracker(cfg);

    for (int i = 0; i < 8; ++i) {
        tracker.record(static_cast<double>(i) * 0.5);
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }

    EXPECT_GT(fires.load(), 0);
    EXPECT_TRUE(tracker.is_fast_move());
}

// ============================================================
// total_records increments
// ============================================================

TEST(TokenVelocityTracker, TotalRecordsIncrements) {
    TokenVelocityTracker tracker;
    for (int i = 0; i < 5; ++i) tracker.record(0.5);
    EXPECT_EQ(tracker.total_records(), 5u);
}

// ============================================================
// reset() clears state
// ============================================================

TEST(TokenVelocityTracker, ResetClearsState) {
    TokenVelocityTracker::Config cfg;
    cfg.min_samples = 4;
    cfg.window_size = 8;
    TokenVelocityTracker tracker(cfg);

    for (int i = 0; i < 8; ++i) {
        tracker.record(static_cast<double>(i) * 0.1);
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }

    tracker.reset();
    EXPECT_DOUBLE_EQ(tracker.velocity(), 0.0);
    EXPECT_EQ(tracker.total_records(), 0u);
}

// ============================================================
// update_config resets state
// ============================================================

TEST(TokenVelocityTracker, UpdateConfigResetsState) {
    TokenVelocityTracker::Config cfg;
    cfg.min_samples = 4;
    TokenVelocityTracker tracker(cfg);

    for (int i = 0; i < 8; ++i) {
        tracker.record(static_cast<double>(i));
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }

    TokenVelocityTracker::Config cfg2;
    tracker.update_config(cfg2);
    EXPECT_DOUBLE_EQ(tracker.velocity(), 0.0);
    EXPECT_EQ(tracker.total_records(), 0u);
}

// ============================================================
// to_stats_json contains required keys
// ============================================================

TEST(TokenVelocityTracker, ToStatsJsonContainsKeys) {
    TokenVelocityTracker tracker;
    tracker.record(0.5);
    std::string json = tracker.to_stats_json();
    EXPECT_NE(json.find("velocity"),       std::string::npos);
    EXPECT_NE(json.find("acceleration"),   std::string::npos);
    EXPECT_NE(json.find("is_fast_move"),   std::string::npos);
    EXPECT_NE(json.find("total_records"),  std::string::npos);
}

// ============================================================
// Concurrent record() is thread-safe
// ============================================================

TEST(TokenVelocityTracker, ConcurrentRecordSafe) {
    TokenVelocityTracker::Config cfg;
    cfg.min_samples         = 4;
    cfg.window_size         = 32;
    cfg.fast_move_threshold = 100.0;
    TokenVelocityTracker tracker(cfg);

    std::vector<std::thread> threads;
    for (int t = 0; t < 4; ++t) {
        threads.emplace_back([&, t] {
            for (int i = 0; i < 100; ++i)
                tracker.record(static_cast<double>((i + t) % 10) * 0.1);
        });
    }
    for (auto& th : threads) th.join();

    EXPECT_EQ(tracker.total_records(), 400u);
}

} // namespace

#else  // LLMQUANT_VELOCITY_TRACKER_ENABLED not defined

TEST(TokenVelocityTracker, DisabledAtBuildTime) {
    SUCCEED();
}

#endif  // LLMQUANT_VELOCITY_TRACKER_ENABLED
