#include <gtest/gtest.h>

#ifdef LLMQUANT_VELOCITY_BREAKER_ENABLED

#include "AdaptiveVelocityBreaker.h"

#include <atomic>
#include <chrono>
#include <thread>
#include <vector>

using namespace llmquant;

namespace {

// ============================================================
// Default state — breaker closed
// ============================================================

TEST(AdaptiveVelocityBreaker, DefaultStateClosed) {
    AdaptiveVelocityBreaker brk;
    EXPECT_FALSE(brk.is_open());
    EXPECT_EQ(brk.trip_count(), 0u);
    EXPECT_EQ(brk.recovery_count(), 0u);
    EXPECT_DOUBLE_EQ(brk.smoothed_velocity(), 0.0);
}

// ============================================================
// First record seeds state, no trip
// ============================================================

TEST(AdaptiveVelocityBreaker, FirstRecordDoesNotTrip) {
    AdaptiveVelocityBreaker brk;
    bool open = brk.record(1.0);
    EXPECT_FALSE(open);
    EXPECT_FALSE(brk.is_open());
}

// ============================================================
// Rapid large bias shifts trip the breaker
// ============================================================

TEST(AdaptiveVelocityBreaker, RapidBiasShiftTripsBreaker) {
    AdaptiveVelocityBreaker::Config cfg;
    cfg.trip_threshold = 0.5;  // very low threshold
    cfg.velocity_alpha = 1.0;  // instant response
    cfg.min_dt_s       = 0.0;  // no min dt
    std::atomic<int> trips{0};
    cfg.on_trip = [&](double) { ++trips; };
    AdaptiveVelocityBreaker brk(cfg);

    brk.record(0.0);
    std::this_thread::sleep_for(std::chrono::milliseconds(1));
    brk.record(10.0);  // huge jump → velocity > threshold

    EXPECT_TRUE(brk.is_open());
    EXPECT_GT(trips.load(), 0);
    EXPECT_GT(brk.trip_count(), 0u);
}

// ============================================================
// Breaker auto-recovers when velocity drops below recovery threshold
// ============================================================

TEST(AdaptiveVelocityBreaker, AutoRecoveryWhenVelocityDrops) {
    AdaptiveVelocityBreaker::Config cfg;
    cfg.trip_threshold = 0.5;
    cfg.recovery_factor = 0.5;   // recover when vel < 0.25
    cfg.velocity_alpha  = 0.8;
    cfg.min_dt_s        = 0.0;
    std::atomic<int> recoveries{0};
    cfg.on_recovery = [&](double) { ++recoveries; };
    AdaptiveVelocityBreaker brk(cfg);

    // Trip it.
    brk.record(0.0);
    std::this_thread::sleep_for(std::chrono::milliseconds(2));
    brk.record(100.0);
    EXPECT_TRUE(brk.is_open());

    // Now feed many stable values to let velocity_ema decay.
    for (int i = 0; i < 20; ++i) {
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
        brk.record(100.0 + i * 0.001);  // tiny changes
    }

    EXPECT_FALSE(brk.is_open());
    EXPECT_GT(recoveries.load(), 0);
    EXPECT_GT(brk.recovery_count(), 0u);
}

// ============================================================
// manual reset() closes breaker
// ============================================================

TEST(AdaptiveVelocityBreaker, ManualResetCloses) {
    AdaptiveVelocityBreaker::Config cfg;
    cfg.trip_threshold = 0.1;
    cfg.velocity_alpha = 1.0;
    cfg.min_dt_s       = 0.0;
    AdaptiveVelocityBreaker brk(cfg);

    brk.record(0.0);
    std::this_thread::sleep_for(std::chrono::milliseconds(1));
    brk.record(100.0);
    EXPECT_TRUE(brk.is_open());

    brk.reset();
    EXPECT_FALSE(brk.is_open());
    EXPECT_DOUBLE_EQ(brk.smoothed_velocity(), 0.0);
}

// ============================================================
// to_stats_json contains required keys
// ============================================================

TEST(AdaptiveVelocityBreaker, ToStatsJsonContainsKeys) {
    AdaptiveVelocityBreaker brk;
    brk.record(0.5);
    std::string json = brk.to_stats_json();
    EXPECT_NE(json.find("is_open"),      std::string::npos);
    EXPECT_NE(json.find("velocity"),     std::string::npos);
    EXPECT_NE(json.find("trip_count"),   std::string::npos);
}

// ============================================================
// Concurrent record() is thread-safe
// ============================================================

TEST(AdaptiveVelocityBreaker, ConcurrentRecordSafe) {
    AdaptiveVelocityBreaker::Config cfg;
    cfg.trip_threshold = 1000.0;  // very high to avoid trips during test
    cfg.velocity_alpha = 0.1;
    cfg.min_dt_s       = 0.0;
    AdaptiveVelocityBreaker brk(cfg);

    std::vector<std::thread> threads;
    for (int t = 0; t < 4; ++t) {
        threads.emplace_back([&, t] {
            for (int i = 0; i < 100; ++i)
                brk.record(static_cast<double>(i + t) * 0.01);
        });
    }
    for (auto& th : threads) th.join();
    // No crash; velocity is a valid non-negative double.
    EXPECT_GE(brk.smoothed_velocity(), 0.0);
}

} // namespace

#else  // LLMQUANT_VELOCITY_BREAKER_ENABLED not defined

TEST(AdaptiveVelocityBreaker, DisabledAtBuildTime) {
    SUCCEED();
}

#endif  // LLMQUANT_VELOCITY_BREAKER_ENABLED
