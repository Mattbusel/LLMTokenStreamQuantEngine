#include <gtest/gtest.h>

#ifdef LLMQUANT_STREAMING_STATS_ENABLED

#include "StreamingStatsCollector.h"
#include <thread>
#include <chrono>

using namespace llmquant;

namespace {

// ============================================================
// Construction and defaults
// ============================================================

TEST(StreamingStatsCollector, DefaultConstructsOk) {
    StreamingStatsCollector sc;
    EXPECT_EQ(sc.total_count(), 0u);
    EXPECT_EQ(sc.total_bytes(), 0u);
    EXPECT_EQ(sc.overflow_count(), 0u);
    EXPECT_NEAR(sc.mean_gap_us(), 0.0, 1e-9);
    EXPECT_EQ(sc.max_gap_us(), 0);
}

// ============================================================
// Latency recording and percentile queries
// ============================================================

TEST(StreamingStatsCollector, SingleLatencyRecorded) {
    StreamingStatsCollector sc;
    sc.record_latency(1000);
    EXPECT_EQ(sc.total_count(), 1u);
    auto pct = sc.latency_percentiles();
    EXPECT_EQ(pct.count, 1u);
    EXPECT_GE(pct.min_us, 0);
    EXPECT_GE(pct.max_us, 0);
}

TEST(StreamingStatsCollector, PercentilesMonotonic) {
    StreamingStatsCollector sc;
    // Record uniform distribution from 0 to 10000 µs.
    for (int i = 0; i < 1000; ++i) {
        sc.record_latency(static_cast<int64_t>(i * 10));
    }
    auto pct = sc.latency_percentiles();
    EXPECT_LE(pct.p50_us, pct.p75_us);
    EXPECT_LE(pct.p75_us, pct.p90_us);
    EXPECT_LE(pct.p90_us, pct.p95_us);
    EXPECT_LE(pct.p95_us, pct.p99_us);
}

TEST(StreamingStatsCollector, P50ApproximatelyMedian) {
    StreamingStatsCollector sc;
    // 1000 observations: half at 100 µs, half at 900 µs.
    for (int i = 0; i < 500; ++i) sc.record_latency(100);
    for (int i = 0; i < 500; ++i) sc.record_latency(900);

    auto pct = sc.latency_percentiles();
    // p50 should be somewhere between 100 and 900 µs.
    EXPECT_GE(pct.p50_us, 0);
    EXPECT_LE(pct.p50_us, 900);
}

TEST(StreamingStatsCollector, OverflowCountsRecorded) {
    StreamingStatsCollector::Config cfg;
    cfg.max_latency_us = 1000;
    StreamingStatsCollector sc(cfg);

    sc.record_latency(500);   // within range
    sc.record_latency(2000);  // overflow
    sc.record_latency(5000);  // overflow

    EXPECT_EQ(sc.total_count(), 3u);
    EXPECT_EQ(sc.overflow_count(), 2u);
}

TEST(StreamingStatsCollector, MinMaxTracked) {
    StreamingStatsCollector sc;
    sc.record_latency(200);
    sc.record_latency(50);
    sc.record_latency(800);

    auto pct = sc.latency_percentiles();
    EXPECT_LE(pct.min_us, 200);
    EXPECT_GE(pct.max_us, 800);
}

// ============================================================
// Byte throughput
// ============================================================

TEST(StreamingStatsCollector, BytesAccumulated) {
    StreamingStatsCollector sc;
    sc.record_latency(100, 32);
    sc.record_latency(200, 64);
    EXPECT_EQ(sc.total_bytes(), 96u);
}

// ============================================================
// Throughput (TPS)
// ============================================================

TEST(StreamingStatsCollector, ThroughputNonNegative) {
    StreamingStatsCollector sc;
    for (int i = 0; i < 50; ++i) sc.record_arrival();
    double tps = sc.throughput_tps();
    EXPECT_GE(tps, 0.0);
}

TEST(StreamingStatsCollector, ThroughputZeroBeforeArrivals) {
    StreamingStatsCollector sc;
    EXPECT_NEAR(sc.throughput_tps(), 0.0, 1e-9);
}

// ============================================================
// Inter-token gap tracking
// ============================================================

TEST(StreamingStatsCollector, GapTrackedAfterMultipleArrivals) {
    StreamingStatsCollector sc;
    sc.record_arrival();
    std::this_thread::sleep_for(std::chrono::milliseconds(5));
    sc.record_arrival();
    std::this_thread::sleep_for(std::chrono::milliseconds(5));
    sc.record_arrival();

    // Mean gap should be positive after multiple arrivals.
    EXPECT_GT(sc.mean_gap_us(), 0.0);
    EXPECT_GT(sc.max_gap_us(), 0);
}

// ============================================================
// P99 alert callback
// ============================================================

TEST(StreamingStatsCollector, P99AlertFires) {
    StreamingStatsCollector::Config cfg;
    cfg.p99_alert_us = 100;  // Alert if p99 > 100 µs.

    int alert_count     = 0;
    int64_t last_p99    = 0;
    cfg.on_p99_alert = [&](int64_t p99, double /*tps*/) {
        ++alert_count;
        last_p99 = p99;
    };

    StreamingStatsCollector sc(cfg);

    // Record 1000 observations above the threshold.
    for (int i = 0; i < 1000; ++i) {
        sc.record_latency(10'000);  // 10 ms each — well above 100 µs
    }

    EXPECT_GT(alert_count, 0);
    EXPECT_GT(last_p99, 0);
}

// ============================================================
// Gap alert callback
// ============================================================

TEST(StreamingStatsCollector, GapAlertFires) {
    StreamingStatsCollector::Config cfg;
    cfg.gap_alert_us = 1000;  // 1 ms alert threshold.

    int gap_fires  = 0;
    int64_t gap_v  = 0;
    cfg.on_high_gap = [&](int64_t gap_us) {
        ++gap_fires;
        gap_v = gap_us;
    };

    StreamingStatsCollector sc(cfg);
    sc.record_arrival();
    std::this_thread::sleep_for(std::chrono::milliseconds(50));  // 50 ms > 1 ms threshold
    sc.record_arrival();

    EXPECT_GT(gap_fires, 0);
    EXPECT_GT(gap_v, 1000);
}

// ============================================================
// Reset
// ============================================================

TEST(StreamingStatsCollector, ResetClearsState) {
    StreamingStatsCollector sc;
    for (int i = 0; i < 100; ++i) sc.record_latency(static_cast<int64_t>(i * 100), 10);
    sc.record_arrival();

    sc.reset();

    EXPECT_EQ(sc.total_count(), 0u);
    EXPECT_EQ(sc.total_bytes(), 0u);
    EXPECT_EQ(sc.overflow_count(), 0u);
    EXPECT_NEAR(sc.mean_gap_us(), 0.0, 1e-9);

    auto pct = sc.latency_percentiles();
    EXPECT_EQ(pct.count, 0u);
}

// ============================================================
// Combined record()
// ============================================================

TEST(StreamingStatsCollector, RecordIncrementsCountAndTracksArrival) {
    StreamingStatsCollector sc;
    sc.record(500, 16);
    sc.record(300, 8);
    EXPECT_EQ(sc.total_count(), 2u);
    EXPECT_EQ(sc.total_bytes(), 24u);
}

// ============================================================
// Stats JSON
// ============================================================

TEST(StreamingStatsCollector, StatsJsonContainsExpectedKeys) {
    StreamingStatsCollector sc;
    sc.record(1000, 32);
    std::string json = sc.to_stats_json();
    EXPECT_NE(json.find("total_count"),    std::string::npos);
    EXPECT_NE(json.find("p50_us"),         std::string::npos);
    EXPECT_NE(json.find("p99_us"),         std::string::npos);
    EXPECT_NE(json.find("throughput_tps"), std::string::npos);
    EXPECT_NE(json.find("mean_gap_us"),    std::string::npos);
    EXPECT_NE(json.find("overflow_count"), std::string::npos);
}

// ============================================================
// Thread safety: concurrent record_latency
// ============================================================

TEST(StreamingStatsCollector, ThreadSafeConcurrentRecords) {
    StreamingStatsCollector sc;
    constexpr int N_THREADS = 4;
    constexpr int N_PER     = 250;

    std::vector<std::thread> threads;
    threads.reserve(N_THREADS);
    for (int t = 0; t < N_THREADS; ++t) {
        threads.emplace_back([&sc, t] {
            for (int i = 0; i < N_PER; ++i) {
                sc.record_latency(static_cast<int64_t>((t + 1) * 100 + i));
            }
        });
    }
    for (auto& th : threads) th.join();

    EXPECT_EQ(sc.total_count(), static_cast<uint64_t>(N_THREADS * N_PER));
}

// ============================================================
// Update config preserves structure
// ============================================================

TEST(StreamingStatsCollector, UpdateConfigApplied) {
    StreamingStatsCollector::Config cfg;
    cfg.throughput_window_ms = 2000;
    StreamingStatsCollector sc(cfg);

    // Verify it doesn't crash and returns reasonable values.
    sc.record_latency(500);
    auto pct = sc.latency_percentiles();
    EXPECT_EQ(pct.count, 1u);

    // Reconfigure.
    cfg.max_latency_us = 5000;
    sc.update_config(cfg);
    sc.record_latency(1000);
    EXPECT_EQ(sc.total_count(), 2u);
}

} // namespace

#else

TEST(StreamingStatsCollector, FeatureDisabled) {
    GTEST_SKIP() << "LLMQUANT_STREAMING_STATS_ENABLED not defined";
}

#endif // LLMQUANT_STREAMING_STATS_ENABLED
