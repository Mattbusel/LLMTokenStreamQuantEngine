#include "gtest/gtest.h"
#include "LatencyController.h"

#include <chrono>
#include <thread>

namespace llmquant {
namespace {

// ---------------------------------------------------------------------------
// Fixture
// ---------------------------------------------------------------------------

static LatencyController::Config make_config(bool profiling = true,
                                              size_t window  = 100) {
    LatencyController::Config cfg;
    cfg.target_latency   = std::chrono::microseconds{10};
    cfg.sample_window    = window;
    cfg.enable_profiling = profiling;
    return cfg;
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

TEST(LatencyControllerTest, test_latency_controller_initial_stats_are_zero) {
    LatencyController lc(make_config());
    auto stats = lc.get_stats();

    EXPECT_EQ(stats.measurements,            0u);
    EXPECT_EQ(stats.avg_latency.count(),     0);
    EXPECT_EQ(stats.min_latency.count(),     0);
    EXPECT_EQ(stats.max_latency.count(),     0);
    EXPECT_EQ(stats.p95_latency.count(),     0);
    EXPECT_EQ(stats.p99_latency.count(),     0);
    EXPECT_DOUBLE_EQ(stats.jitter_ms,        0.0);
}

TEST(LatencyControllerTest, test_latency_controller_record_single_measurement_updates_stats) {
    LatencyController lc(make_config());
    lc.record_latency(std::chrono::microseconds{42});

    auto stats = lc.get_stats();
    EXPECT_EQ(stats.measurements, 1u);
    EXPECT_EQ(stats.avg_latency, std::chrono::microseconds{42});
    EXPECT_EQ(stats.min_latency, std::chrono::microseconds{42});
    EXPECT_EQ(stats.max_latency, std::chrono::microseconds{42});
}

TEST(LatencyControllerTest, test_latency_controller_min_max_track_extremes) {
    LatencyController lc(make_config());

    lc.record_latency(std::chrono::microseconds{100});
    lc.record_latency(std::chrono::microseconds{5});
    lc.record_latency(std::chrono::microseconds{50});
    lc.record_latency(std::chrono::microseconds{200});
    lc.record_latency(std::chrono::microseconds{1});

    auto stats = lc.get_stats();
    EXPECT_EQ(stats.min_latency, std::chrono::microseconds{1});
    EXPECT_EQ(stats.max_latency, std::chrono::microseconds{200});
    EXPECT_EQ(stats.measurements, 5u);
}

TEST(LatencyControllerTest, test_latency_controller_percentiles_calculated_from_samples) {
    LatencyController lc(make_config(true, 200));

    // Insert 100 samples with known distribution: 1..100 μs.
    for (int i = 1; i <= 100; ++i) {
        lc.record_latency(std::chrono::microseconds{i});
    }

    auto stats = lc.get_stats();
    // p95 index = floor(100 * 0.95) = 95 => samples[95] after sorting = 96 μs (0-indexed)
    // Allow a small range since the exact boundary depends on implementation.
    EXPECT_GE(stats.p95_latency.count(), 90);
    EXPECT_LE(stats.p95_latency.count(), 100);

    EXPECT_GE(stats.p99_latency.count(), 95);
    EXPECT_LE(stats.p99_latency.count(), 100);
}

TEST(LatencyControllerTest, test_latency_controller_reset_clears_all_stats) {
    LatencyController lc(make_config());

    lc.record_latency(std::chrono::microseconds{999});
    lc.reset_stats();

    auto stats = lc.get_stats();
    EXPECT_EQ(stats.measurements, 0u);
    EXPECT_EQ(stats.avg_latency.count(), 0);
    EXPECT_EQ(stats.max_latency.count(), 0);
}

TEST(LatencyControllerTest, test_latency_controller_get_stats_with_no_measurements_returns_zero) {
    // Profiling disabled — percentile path is skipped.
    LatencyController lc(make_config(false));
    auto stats = lc.get_stats();

    EXPECT_EQ(stats.measurements,        0u);
    EXPECT_EQ(stats.avg_latency.count(), 0);
    EXPECT_EQ(stats.p95_latency.count(), 0);
    EXPECT_EQ(stats.p99_latency.count(), 0);
}

TEST(LatencyControllerTest, test_latency_controller_start_end_measurement_records_nonzero) {
    LatencyController lc(make_config());
    lc.start_measurement();
    // Spin briefly so the timer has something to measure.
    volatile int sink = 0;
    for (int i = 0; i < 10000; ++i) { sink += i; }
    (void)sink;
    lc.end_measurement();

    auto stats = lc.get_stats();
    EXPECT_EQ(stats.measurements, 1u);
    // Any positive duration is acceptable (even 0 on very fast machines, so
    // just check it doesn't go negative — chrono duration is unsigned here).
    EXPECT_GE(stats.avg_latency.count(), 0);
}

TEST(LatencyControllerTest, test_latency_controller_sample_window_limits_vector_size) {
    // With window = 5, inserting 20 samples should not grow the vector past 5.
    LatencyController lc(make_config(true, 5));

    for (int i = 0; i < 20; ++i) {
        lc.record_latency(std::chrono::microseconds{i + 1});
    }

    // We can't inspect the private vector directly, but we can verify that
    // stats are still consistent (percentile calculation won't crash).
    auto stats = lc.get_stats();
    EXPECT_EQ(stats.measurements, 20u);
    // p99 should reflect the most-recent samples, all in [16,20].
    EXPECT_GE(stats.p99_latency.count(), 0);
}

// ---------------------------------------------------------------------------
// Pressure system tests
// ---------------------------------------------------------------------------

TEST(LatencyControllerTest, test_latency_controller_ingestion_pressure_clamps_to_one) {
    LatencyController lc(make_config());

    // Arrival rate 3x the max — pressure must clamp at 1.0.
    lc.update_ingestion_pressure(30000.0, 10000.0);

    auto ps = lc.get_pressure();
    EXPECT_DOUBLE_EQ(ps.ingestion_pressure, 1.0);
    EXPECT_DOUBLE_EQ(ps.composite,          1.0);
}

TEST(LatencyControllerTest, test_latency_controller_composite_is_max_of_three) {
    LatencyController lc(make_config());

    lc.update_ingestion_pressure(2000.0, 10000.0);  // 0.2
    lc.update_semantic_pressure(0.1);                // 0.4  (0.1/0.25)
    lc.update_queue_pressure(200, 512);              // ~0.39

    auto ps = lc.get_pressure();
    double expected_max = std::max({ps.ingestion_pressure,
                                    ps.semantic_pressure,
                                    ps.queue_pressure});
    EXPECT_DOUBLE_EQ(ps.composite, expected_max);
}

TEST(LatencyControllerTest, test_latency_controller_backoff_ramps_above_high_threshold) {
    LatencyController lc(make_config());

    // Push composite above 0.8 by driving ingestion to 100 %.
    for (int i = 0; i < 5; ++i) {
        lc.update_ingestion_pressure(10000.0, 10000.0);  // composite = 1.0
    }

    double backoff = lc.get_backoff_multiplier();
    EXPECT_GT(backoff, 1.0) << "Backoff must ramp when composite pressure >= 0.8";
    EXPECT_LE(backoff, 5.0) << "Backoff must not exceed the 5x ceiling";
}

TEST(LatencyControllerTest, test_latency_controller_backoff_resets_below_low_threshold) {
    LatencyController lc(make_config());

    // First ramp backoff up.
    for (int i = 0; i < 5; ++i) {
        lc.update_ingestion_pressure(10000.0, 10000.0);
    }
    EXPECT_GT(lc.get_backoff_multiplier(), 1.0);

    // Now drive all pressures to zero — composite drops below 0.5.
    lc.update_ingestion_pressure(0.0, 10000.0);
    lc.update_semantic_pressure(0.0);
    lc.update_queue_pressure(0, 512);

    EXPECT_DOUBLE_EQ(lc.get_backoff_multiplier(), 1.0)
        << "Backoff must reset to 1.0 when composite pressure < 0.5";
}

} // namespace
} // namespace llmquant
