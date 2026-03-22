#include <gtest/gtest.h>

#ifdef LLMQUANT_LATENCY_JITTER_ENABLED

#include "LatencyJitterMonitor.h"

#include <atomic>
#include <cmath>
#include <thread>
#include <vector>

using namespace llmquant;

namespace {

// ---------------------------------------------------------------------------
// Default state
// ---------------------------------------------------------------------------

TEST(LatencyJitterMonitor, DefaultState) {
    LatencyJitterMonitor ljm;
    EXPECT_DOUBLE_EQ(ljm.mad_us(), 0.0);
    EXPECT_DOUBLE_EQ(ljm.median_us(), 0.0);
    EXPECT_FALSE(ljm.is_jittery());
    EXPECT_EQ(ljm.total_samples(), 0u);
    EXPECT_EQ(ljm.jitter_events(), 0u);
}

// ---------------------------------------------------------------------------
// total_samples increments on every record
// ---------------------------------------------------------------------------

TEST(LatencyJitterMonitor, TotalSamplesIncrements) {
    LatencyJitterMonitor ljm;
    ljm.record(10.0);
    ljm.record(20.0);
    ljm.record(30.0);
    EXPECT_EQ(ljm.total_samples(), 3u);
}

// ---------------------------------------------------------------------------
// Constant latency → MAD = 0
// ---------------------------------------------------------------------------

TEST(LatencyJitterMonitor, ConstantLatencyMadZero) {
    LatencyJitterMonitor::Config cfg;
    cfg.window_size       = 16;
    cfg.min_samples       = 4;
    cfg.jitter_threshold_us = 100.0;
    LatencyJitterMonitor ljm(cfg);
    for (int i = 0; i < 20; ++i) ljm.record(50.0);
    EXPECT_NEAR(ljm.mad_us(), 0.0, 1e-6);
    EXPECT_FALSE(ljm.is_jittery());
}

// ---------------------------------------------------------------------------
// High-variance input exceeds threshold → is_jittery + callback
// ---------------------------------------------------------------------------

TEST(LatencyJitterMonitor, HighVarianceTriggers) {
    LatencyJitterMonitor::Config cfg;
    cfg.window_size         = 32;
    cfg.min_samples         = 8;
    cfg.jitter_threshold_us = 100.0;
    std::atomic<int> fires{0};
    cfg.on_jitter_spike = [&](double) { ++fires; };
    LatencyJitterMonitor ljm(cfg);

    // Alternating 0 μs and 2000 μs — MAD will be large.
    for (int i = 0; i < 20; ++i) {
        ljm.record(i % 2 == 0 ? 0.0 : 2000.0);
    }
    EXPECT_TRUE(ljm.is_jittery());
    EXPECT_GT(ljm.mad_us(), 100.0);
    EXPECT_GT(fires.load(), 0);
    EXPECT_GT(ljm.jitter_events(), 0u);
}

// ---------------------------------------------------------------------------
// Jitter clears after returning to stable latency
// ---------------------------------------------------------------------------

TEST(LatencyJitterMonitor, JitterClears) {
    LatencyJitterMonitor::Config cfg;
    cfg.window_size         = 16;
    cfg.min_samples         = 4;
    cfg.jitter_threshold_us = 100.0;
    cfg.clear_hysteresis    = 0.75;
    std::atomic<int> clears{0};
    cfg.on_jitter_clear = [&](double) { ++clears; };
    LatencyJitterMonitor ljm(cfg);

    // Cause jitter.
    for (int i = 0; i < 10; ++i)
        ljm.record(i % 2 == 0 ? 0.0 : 2000.0);

    // Return to stable — fill entire window with constant values.
    for (int i = 0; i < 20; ++i) ljm.record(50.0);

    EXPECT_FALSE(ljm.is_jittery());
    EXPECT_GT(clears.load(), 0);
}

// ---------------------------------------------------------------------------
// MAD is non-negative
// ---------------------------------------------------------------------------

TEST(LatencyJitterMonitor, MadNonNegative) {
    LatencyJitterMonitor::Config cfg;
    cfg.window_size = 32;
    cfg.min_samples = 4;
    LatencyJitterMonitor ljm(cfg);
    for (int i = 0; i < 40; ++i)
        ljm.record(static_cast<double>(i % 10) * 50.0);
    EXPECT_GE(ljm.mad_us(), 0.0);
}

// ---------------------------------------------------------------------------
// Median is within range of recorded values
// ---------------------------------------------------------------------------

TEST(LatencyJitterMonitor, MedianInRange) {
    LatencyJitterMonitor::Config cfg;
    cfg.window_size = 16;
    cfg.min_samples = 4;
    LatencyJitterMonitor ljm(cfg);
    for (int i = 0; i < 16; ++i)
        ljm.record(static_cast<double>(i) * 10.0);  // 0..150 μs
    EXPECT_GE(ljm.median_us(), 0.0);
    EXPECT_LE(ljm.median_us(), 160.0);
}

// ---------------------------------------------------------------------------
// reset() clears all state
// ---------------------------------------------------------------------------

TEST(LatencyJitterMonitor, ResetClearsState) {
    LatencyJitterMonitor ljm;
    for (int i = 0; i < 30; ++i) ljm.record(static_cast<double>(i) * 10.0);
    ljm.reset();
    EXPECT_EQ(ljm.total_samples(), 0u);
    EXPECT_EQ(ljm.jitter_events(), 0u);
    EXPECT_FALSE(ljm.is_jittery());
    EXPECT_DOUBLE_EQ(ljm.mad_us(), 0.0);
    EXPECT_DOUBLE_EQ(ljm.median_us(), 0.0);
}

// ---------------------------------------------------------------------------
// update_config resets state
// ---------------------------------------------------------------------------

TEST(LatencyJitterMonitor, UpdateConfigResetsState) {
    LatencyJitterMonitor ljm;
    for (int i = 0; i < 20; ++i) ljm.record(1000.0);
    LatencyJitterMonitor::Config cfg2;
    cfg2.window_size = 8;
    ljm.update_config(cfg2);
    EXPECT_EQ(ljm.total_samples(), 0u);
    EXPECT_FALSE(ljm.is_jittery());
}

// ---------------------------------------------------------------------------
// to_stats_json contains expected keys
// ---------------------------------------------------------------------------

TEST(LatencyJitterMonitor, ToStatsJsonContainsKeys) {
    LatencyJitterMonitor ljm;
    ljm.record(50.0);
    std::string j = ljm.to_stats_json();
    EXPECT_NE(j.find("mad_us"),        std::string::npos);
    EXPECT_NE(j.find("median_us"),     std::string::npos);
    EXPECT_NE(j.find("is_jittery"),    std::string::npos);
    EXPECT_NE(j.find("jitter_events"), std::string::npos);
    EXPECT_NE(j.find("total_samples"), std::string::npos);
}

// ---------------------------------------------------------------------------
// Below min_samples → MAD stays 0
// ---------------------------------------------------------------------------

TEST(LatencyJitterMonitor, BelowMinSamplesMadZero) {
    LatencyJitterMonitor::Config cfg;
    cfg.window_size = 32;
    cfg.min_samples = 16;
    LatencyJitterMonitor ljm(cfg);
    // Record fewer than min_samples with high variance.
    for (int i = 0; i < 10; ++i)
        ljm.record(i % 2 == 0 ? 0.0 : 9999.0);
    EXPECT_DOUBLE_EQ(ljm.mad_us(), 0.0);
    EXPECT_FALSE(ljm.is_jittery());
}

// ---------------------------------------------------------------------------
// Concurrent record() is thread-safe
// ---------------------------------------------------------------------------

TEST(LatencyJitterMonitor, ConcurrentRecordSafe) {
    LatencyJitterMonitor::Config cfg;
    cfg.window_size = 64;
    cfg.min_samples = 8;
    LatencyJitterMonitor ljm(cfg);

    std::atomic<bool> go{false};
    auto worker = [&](int id) {
        while (!go.load()) {}
        for (int i = 0; i < 250; ++i)
            ljm.record(static_cast<double>((i + id * 7) % 200) * 5.0);
    };
    std::vector<std::thread> threads;
    for (int t = 0; t < 4; ++t) threads.emplace_back(worker, t);
    go.store(true);
    for (auto& t : threads) t.join();

    EXPECT_EQ(ljm.total_samples(), 1000u);
    EXPECT_GE(ljm.mad_us(), 0.0);
    EXPECT_TRUE(std::isfinite(ljm.median_us()));
}

} // namespace

#else

TEST(LatencyJitterMonitor, DisabledAtBuildTime) { SUCCEED(); }

#endif  // LLMQUANT_LATENCY_JITTER_ENABLED
