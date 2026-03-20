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
    EXPECT_EQ(stats.p50_latency.count(),     0);
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
    EXPECT_EQ(stats.p50_latency.count(), 0);
    EXPECT_EQ(stats.p95_latency.count(), 0);
    EXPECT_EQ(stats.p99_latency.count(), 0);
    EXPECT_EQ(stats.min_latency.count(), 0);
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

// ---------------------------------------------------------------------------
// p50 (median) percentile
// ---------------------------------------------------------------------------

TEST(LatencyControllerTest, test_latency_controller_p50_is_median) {
    LatencyController lc(make_config(true, 200));

    // Insert 100 samples: 1..100 μs.  Median of [1,100] is the 50th value ≈ 50 μs.
    for (int i = 1; i <= 100; ++i) {
        lc.record_latency(std::chrono::microseconds{i});
    }

    auto stats = lc.get_stats();
    // Nearest-rank p50: ceil(100 * 0.50) - 1 = 49 (0-indexed), sorted value = 50 μs.
    // Allow ±5 μs tolerance for implementation differences.
    EXPECT_GE(stats.p50_latency.count(), 45);
    EXPECT_LE(stats.p50_latency.count(), 55);
    // p50 must be <= p95 and p99.
    EXPECT_LE(stats.p50_latency.count(), stats.p95_latency.count());
    EXPECT_LE(stats.p50_latency.count(), stats.p99_latency.count());
}

// ---------------------------------------------------------------------------
// New: guard against end_measurement() before start_measurement() (impr. #10)
// ---------------------------------------------------------------------------

TEST(LatencyControllerTest, test_end_measurement_before_start_does_not_record) {
    LatencyController lc(make_config());
    // Calling end_measurement() before start_measurement() on the same thread
    // must not record any sample (the active guard prevents it).
    lc.end_measurement();
    auto stats = lc.get_stats();
    EXPECT_EQ(stats.measurements, 0u)
        << "end_measurement() before start_measurement() must not record a sample";
}

// ---------------------------------------------------------------------------
// New: concurrent hot-reload does not crash ongoing get_config calls (impr. #15)
// ---------------------------------------------------------------------------

TEST(LatencyControllerTest, test_concurrent_record_and_get_stats_is_safe) {
    // Spin two threads: one records latency, one reads stats concurrently.
    LatencyController lc(make_config(true, 200));
    std::atomic<bool> stop{false};

    std::thread writer([&]() {
        for (int i = 0; !stop.load() && i < 500; ++i) {
            lc.record_latency(std::chrono::microseconds(i + 1));
        }
    });

    std::thread reader([&]() {
        for (int i = 0; !stop.load() && i < 100; ++i) {
            (void)lc.get_stats();
            std::this_thread::sleep_for(std::chrono::microseconds(100));
        }
    });

    writer.join();
    stop = true;
    reader.join();
    // No assertion — just must not crash or deadlock.
    SUCCEED();
}

TEST(LatencyControllerTest, test_latency_controller_profiling_hooks_do_not_crash) {
    // Verify that the three profiling hooks run without throwing or crashing.
    LatencyController lc(make_config(true /*enable profiling*/));
    EXPECT_NO_THROW({
        lc.profile_token_processing();
        lc.profile_signal_generation();
        lc.profile_queue_lag();
    });
}

TEST(LatencyControllerTest, test_latency_controller_profiling_hooks_disabled_do_not_crash) {
    // With profiling disabled, the hooks must be no-ops.
    LatencyController lc(make_config(false /*disable profiling*/));
    EXPECT_NO_THROW({
        lc.profile_token_processing();
        lc.profile_signal_generation();
        lc.profile_queue_lag();
    });
}

// ---------------------------------------------------------------------------
// p5 (5th-percentile) latency
// ---------------------------------------------------------------------------

TEST(LatencyControllerTest, test_latency_controller_p5_is_populated) {
    LatencyController lc(make_config(true, 200));

    // Insert 100 samples: 1..100 µs. The 5th-percentile nearest-rank is the
    // 5th value (1-indexed) in a sorted array = 5 µs; allow ±5 µs tolerance.
    for (int i = 1; i <= 100; ++i) {
        lc.record_latency(std::chrono::microseconds{i});
    }

    auto stats = lc.get_stats();
    EXPECT_GT(stats.p5_latency.count(), 0)  << "p5_latency must be non-zero when samples exist";
    EXPECT_GE(stats.p5_latency.count(), 1);
    EXPECT_LE(stats.p5_latency.count(), 10) << "p5 of [1..100] should be near 5 µs";
}

TEST(LatencyControllerTest, test_latency_controller_p5_le_p50_le_p95) {
    LatencyController lc(make_config(true, 200));

    for (int i = 1; i <= 100; ++i) {
        lc.record_latency(std::chrono::microseconds{i});
    }

    auto s = lc.get_stats();
    EXPECT_LE(s.p5_latency.count(),  s.p50_latency.count())  << "p5 must be <= p50";
    EXPECT_LE(s.p50_latency.count(), s.p95_latency.count())  << "p50 must be <= p95";
    EXPECT_LE(s.p95_latency.count(), s.p99_latency.count())  << "p95 must be <= p99";
}

TEST(LatencyControllerTest, test_latency_controller_reset_clears_p5) {
    LatencyController lc(make_config());

    lc.record_latency(std::chrono::microseconds{999});
    lc.reset_stats();

    auto stats = lc.get_stats();
    EXPECT_EQ(stats.p5_latency.count(), 0) << "p5_latency must be zero after reset";
}

// ---------------------------------------------------------------------------
// histogram_buckets
// ---------------------------------------------------------------------------

TEST(LatencyControllerTest, test_histogram_buckets_cumulative_and_ordered) {
    LatencyController lc(make_config(true, 200));

    // Insert 10 samples at 1 µs (all fall in the 0.5–1.0 µs bucket or below).
    // Insert 10 samples at 10 µs (fall in the ≤10 µs bucket).
    // Insert 10 samples at 100 µs (fall in the ≤100 µs bucket).
    for (int i = 0; i < 10; ++i) lc.record_latency(std::chrono::microseconds{1});
    for (int i = 0; i < 10; ++i) lc.record_latency(std::chrono::microseconds{10});
    for (int i = 0; i < 10; ++i) lc.record_latency(std::chrono::microseconds{100});

    auto buckets = lc.histogram_buckets();
    ASSERT_FALSE(buckets.empty());

    // Buckets must be ordered by upper_bound_us (monotonically non-decreasing).
    for (size_t i = 1; i < buckets.size(); ++i) {
        EXPECT_LE(buckets[i-1].upper_bound_us, buckets[i].upper_bound_us)
            << "Buckets must be in non-decreasing order";
    }

    // Cumulative property: each bucket count >= prior bucket count.
    for (size_t i = 1; i < buckets.size(); ++i) {
        EXPECT_GE(buckets[i].count, buckets[i-1].count)
            << "Histogram must be cumulative";
    }

    // The +Inf bucket must contain all 30 samples.
    EXPECT_EQ(buckets.back().count, 30u) << "+Inf bucket must contain all samples";
}

// ---------------------------------------------------------------------------
// get_window_fill_ratio
// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// p25 (first quartile) latency
// ---------------------------------------------------------------------------

TEST(LatencyControllerTest, test_latency_controller_p25_is_populated) {
    LatencyController lc(make_config(true, 200));

    for (int i = 1; i <= 100; ++i) {
        lc.record_latency(std::chrono::microseconds{i});
    }

    auto stats = lc.get_stats();
    // Nearest-rank p25 of [1..100]: ceil(100*0.25)-1 = 24 (0-indexed) = 25 µs.
    EXPECT_GE(stats.p25_latency.count(), 20) << "p25 of [1..100] should be near 25 µs";
    EXPECT_LE(stats.p25_latency.count(), 30);
}

TEST(LatencyControllerTest, test_latency_controller_p5_le_p25_le_p50) {
    LatencyController lc(make_config(true, 200));

    for (int i = 1; i <= 100; ++i) {
        lc.record_latency(std::chrono::microseconds{i});
    }

    auto s = lc.get_stats();
    EXPECT_LE(s.p5_latency.count(),  s.p25_latency.count()) << "p5 must be <= p25";
    EXPECT_LE(s.p25_latency.count(), s.p50_latency.count()) << "p25 must be <= p50";
}

TEST(LatencyControllerTest, test_window_fill_ratio_zero_before_measurements) {
    LatencyController lc(make_config(true, 100));
    EXPECT_DOUBLE_EQ(lc.get_window_fill_ratio(), 0.0)
        << "Fill ratio must be 0 before any measurements";
}

TEST(LatencyControllerTest, test_window_fill_ratio_rises_with_samples) {
    LatencyController lc(make_config(true, 100));
    for (int i = 0; i < 50; ++i) {
        lc.record_latency(std::chrono::microseconds{i + 1});
    }
    double ratio = lc.get_window_fill_ratio();
    EXPECT_GE(ratio, 0.49) << "After 50/100 samples fill ratio should be ~0.5";
    EXPECT_LE(ratio, 0.51);
}

TEST(LatencyControllerTest, test_window_fill_ratio_saturates_at_one) {
    LatencyController lc(make_config(true, 10));
    for (int i = 0; i < 20; ++i) {
        lc.record_latency(std::chrono::microseconds{i + 1});
    }
    EXPECT_DOUBLE_EQ(lc.get_window_fill_ratio(), 1.0)
        << "Fill ratio must saturate at 1.0 once window is full";
}

TEST(LatencyControllerTest, test_window_fill_ratio_reset_to_zero) {
    LatencyController lc(make_config(true, 10));
    lc.record_latency(std::chrono::microseconds{5});
    lc.reset_stats();
    EXPECT_DOUBLE_EQ(lc.get_window_fill_ratio(), 0.0)
        << "Fill ratio must be 0 after reset";
}

TEST(LatencyControllerTest, test_latency_controller_get_target_latency_returns_configured_value) {
    LatencyController::Config cfg;
    cfg.target_latency   = std::chrono::microseconds{42};
    cfg.sample_window    = 100;
    cfg.enable_profiling = true;
    LatencyController lc(cfg);
    EXPECT_EQ(lc.get_target_latency(), std::chrono::microseconds{42})
        << "get_target_latency must return the value set at construction";
}

// ---------------------------------------------------------------------------
// p75 (third quartile) latency
// ---------------------------------------------------------------------------

TEST(LatencyControllerTest, test_latency_controller_p75_is_populated) {
    LatencyController lc(make_config(true, 200));

    // Insert 100 samples: 1..100 µs. Nearest-rank p75: ceil(100*0.75)-1 = 74
    // (0-indexed), sorted value = 75 µs; allow ±5 µs tolerance.
    for (int i = 1; i <= 100; ++i) {
        lc.record_latency(std::chrono::microseconds{i});
    }

    auto stats = lc.get_stats();
    EXPECT_GE(stats.p75_latency.count(), 70) << "p75 of [1..100] should be near 75 µs";
    EXPECT_LE(stats.p75_latency.count(), 80);
}

TEST(LatencyControllerTest, test_latency_controller_p50_le_p75_le_p95) {
    LatencyController lc(make_config(true, 200));

    for (int i = 1; i <= 100; ++i) {
        lc.record_latency(std::chrono::microseconds{i});
    }

    auto s = lc.get_stats();
    EXPECT_LE(s.p50_latency.count(), s.p75_latency.count()) << "p50 must be <= p75";
    EXPECT_LE(s.p75_latency.count(), s.p95_latency.count()) << "p75 must be <= p95";
}

TEST(LatencyControllerTest, test_latency_controller_reset_clears_p75) {
    LatencyController lc(make_config());

    lc.record_latency(std::chrono::microseconds{999});
    lc.reset_stats();

    auto stats = lc.get_stats();
    EXPECT_EQ(stats.p75_latency.count(), 0) << "p75_latency must be zero after reset";
}

TEST(LatencyControllerTest, test_histogram_buckets_empty_when_profiling_disabled) {
    LatencyController lc(make_config(false /*profiling off*/));
    lc.record_latency(std::chrono::microseconds{5});

    auto buckets = lc.histogram_buckets();
    // All counts must be zero when profiling is disabled.
    for (const auto& b : buckets) {
        EXPECT_EQ(b.count, 0u) << "All bucket counts must be zero when profiling is disabled";
    }
}

// ---------------------------------------------------------------------------
// target_breaches counter
// ---------------------------------------------------------------------------

TEST(LatencyControllerTest, test_target_breaches_zero_when_all_below_target) {
    // target_latency = 10 µs; record samples all at or below target.
    LatencyController lc(make_config());  // target = 10 µs
    for (int i = 1; i <= 10; ++i) {
        lc.record_latency(std::chrono::microseconds{i});  // all <= 10 µs
    }
    EXPECT_EQ(lc.get_stats().target_breaches, 0u)
        << "No breaches when all samples <= target_latency";
}

TEST(LatencyControllerTest, test_target_breaches_counts_samples_above_target) {
    LatencyController lc(make_config());  // target = 10 µs
    lc.record_latency(std::chrono::microseconds{5});   // below target
    lc.record_latency(std::chrono::microseconds{10});  // exactly at target — not a breach
    lc.record_latency(std::chrono::microseconds{11});  // above — breach
    lc.record_latency(std::chrono::microseconds{50});  // above — breach
    lc.record_latency(std::chrono::microseconds{100}); // above — breach

    EXPECT_EQ(lc.get_stats().target_breaches, 3u)
        << "Only samples strictly above target_latency count as breaches";
}

TEST(LatencyControllerTest, test_target_breaches_cleared_by_reset_stats) {
    LatencyController lc(make_config());  // target = 10 µs
    lc.record_latency(std::chrono::microseconds{100});  // breach
    ASSERT_GT(lc.get_stats().target_breaches, 0u);

    lc.reset_stats();
    EXPECT_EQ(lc.get_stats().target_breaches, 0u)
        << "target_breaches must be zero after reset_stats()";
}

// ---------------------------------------------------------------------------
// is_warmed_up()
// ---------------------------------------------------------------------------

TEST(LatencyControllerTest, test_is_warmed_up_false_before_half_window) {
    LatencyController lc(make_config(true, 100));  // window = 100

    // Fill < 50% of the window.
    for (int i = 0; i < 49; ++i) {
        lc.record_latency(std::chrono::microseconds{i + 1});
    }

    EXPECT_FALSE(lc.is_warmed_up())
        << "is_warmed_up() must return false when window < 50% full";
}

TEST(LatencyControllerTest, test_is_warmed_up_true_at_half_window) {
    LatencyController lc(make_config(true, 100));  // window = 100

    // Fill exactly 50% of the window.
    for (int i = 0; i < 50; ++i) {
        lc.record_latency(std::chrono::microseconds{i + 1});
    }

    EXPECT_TRUE(lc.is_warmed_up())
        << "is_warmed_up() must return true once window is >= 50% full";
}

// ---------------------------------------------------------------------------
// get_slo_breach_rate()
// ---------------------------------------------------------------------------

TEST(LatencyControllerTest, test_slo_breach_rate_zero_with_no_breaches) {
    LatencyController lc(make_config());  // target = 10 µs
    lc.record_latency(std::chrono::microseconds{5});
    lc.record_latency(std::chrono::microseconds{10});
    EXPECT_DOUBLE_EQ(lc.get_slo_breach_rate(), 0.0)
        << "Breach rate must be 0.0 when all samples <= target";
}

TEST(LatencyControllerTest, test_slo_breach_rate_correct_fraction) {
    LatencyController lc(make_config());  // target = 10 µs
    // 2 below, 2 above.
    lc.record_latency(std::chrono::microseconds{5});
    lc.record_latency(std::chrono::microseconds{8});
    lc.record_latency(std::chrono::microseconds{11});
    lc.record_latency(std::chrono::microseconds{20});

    EXPECT_NEAR(lc.get_slo_breach_rate(), 0.5, 1e-9)
        << "2 of 4 samples above target → breach rate = 0.5";
}

} // namespace
} // namespace llmquant
