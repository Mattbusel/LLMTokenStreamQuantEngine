#include <gtest/gtest.h>

#ifdef LLMQUANT_SENTIMENT_CYCLE_ENABLED

#include "SentimentCycleDetector.h"

#include <atomic>
#include <cmath>
#include <thread>
#include <vector>

using namespace llmquant;

namespace {

// ---------------------------------------------------------------------------
// Construction / default state
// ---------------------------------------------------------------------------

TEST(SentimentCycleDetector, DefaultState) {
    SentimentCycleDetector det;
    EXPECT_EQ(det.dominant_period(), 0);
    EXPECT_DOUBLE_EQ(det.cycle_strength(), 0.0);
    EXPECT_FALSE(det.is_cyclic());
    EXPECT_EQ(det.total_records(), 0u);
}

// ---------------------------------------------------------------------------
// Insufficient samples → period stays 0
// ---------------------------------------------------------------------------

TEST(SentimentCycleDetector, InsufficientSamplesNoPeriod) {
    SentimentCycleDetector::Config cfg;
    cfg.min_samples = 32;
    SentimentCycleDetector det(cfg);

    for (int i = 0; i < 10; ++i) det.record(static_cast<double>(i % 2));

    EXPECT_EQ(det.dominant_period(), 0);
}

// ---------------------------------------------------------------------------
// Constant series → no cycle detected (variance = 0)
// ---------------------------------------------------------------------------

TEST(SentimentCycleDetector, ConstantSeriesNoCycle) {
    SentimentCycleDetector det;
    for (int i = 0; i < 128; ++i) det.record(1.0);
    EXPECT_EQ(det.dominant_period(), 0);
    EXPECT_FALSE(det.is_cyclic());
}

// ---------------------------------------------------------------------------
// Perfect period-4 sine wave → dominant_period == 4
// ---------------------------------------------------------------------------

TEST(SentimentCycleDetector, Period4Detected) {
    SentimentCycleDetector::Config cfg;
    cfg.window_size  = 128;
    cfg.max_lag      = 32;
    cfg.min_samples  = 32;
    cfg.cyclic_threshold = 0.5;
    SentimentCycleDetector det(cfg);

    // Feed 200 samples of period-4 square wave.
    for (int i = 0; i < 200; ++i) det.record((i % 4 < 2) ? 1.0 : -1.0);

    EXPECT_EQ(det.dominant_period(), 4);
    EXPECT_GT(det.cycle_strength(), 0.5);
    EXPECT_TRUE(det.is_cyclic());
}

// ---------------------------------------------------------------------------
// Period-8 sine wave → dominant_period == 8
// ---------------------------------------------------------------------------

TEST(SentimentCycleDetector, Period8Detected) {
    SentimentCycleDetector::Config cfg;
    cfg.window_size  = 128;
    cfg.max_lag      = 32;
    cfg.min_samples  = 32;
    cfg.cyclic_threshold = 0.4;
    SentimentCycleDetector det(cfg);

    for (int i = 0; i < 256; ++i)
        det.record(std::sin(2.0 * M_PI * i / 8.0));

    EXPECT_EQ(det.dominant_period(), 8);
    EXPECT_GT(det.cycle_strength(), 0.4);
    EXPECT_TRUE(det.is_cyclic());
}

// ---------------------------------------------------------------------------
// Noisy random series → is_cyclic() probably false (weak ACF)
// ---------------------------------------------------------------------------

TEST(SentimentCycleDetector, NoisySeriesNotCyclic) {
    SentimentCycleDetector::Config cfg;
    cfg.cyclic_threshold = 0.5;  // high bar
    SentimentCycleDetector det(cfg);

    // Pseudo-random input without strong periodicity.
    double v = 1.0;
    for (int i = 0; i < 256; ++i) {
        v = v * 1.6180339887 - std::floor(v * 1.6180339887);
        det.record(v * 2.0 - 1.0);
    }

    EXPECT_FALSE(det.is_cyclic());
}

// ---------------------------------------------------------------------------
// on_period_change callback fires when period shifts
// ---------------------------------------------------------------------------

TEST(SentimentCycleDetector, PeriodChangeCallbackFires) {
    SentimentCycleDetector::Config cfg;
    cfg.window_size  = 64;
    cfg.max_lag      = 16;
    cfg.min_samples  = 16;
    std::atomic<int> changes{0};
    cfg.on_period_change = [&](int, int, double) { ++changes; };
    SentimentCycleDetector det(cfg);

    // Period 4 for a while.
    for (int i = 0; i < 64; ++i) det.record((i % 4 < 2) ? 1.0 : -1.0);
    // Switch to period 8.
    for (int i = 0; i < 64; ++i) det.record((i % 8 < 4) ? 1.0 : -1.0);

    EXPECT_GT(changes.load(), 0);
    EXPECT_GT(det.period_changes(), 0u);
}

// ---------------------------------------------------------------------------
// acf_snapshot() length matches max_lag
// ---------------------------------------------------------------------------

TEST(SentimentCycleDetector, AcfSnapshotLength) {
    SentimentCycleDetector::Config cfg;
    cfg.max_lag = 12;
    SentimentCycleDetector det(cfg);
    for (int i = 0; i < 64; ++i) det.record(1.0 * (i % 3));
    auto acf = det.acf_snapshot();
    EXPECT_EQ(static_cast<int>(acf.size()), cfg.max_lag);
}

// ---------------------------------------------------------------------------
// total_records increments correctly
// ---------------------------------------------------------------------------

TEST(SentimentCycleDetector, TotalRecordsIncrements) {
    SentimentCycleDetector det;
    det.record(0.1);
    det.record(0.2);
    det.record(0.3);
    EXPECT_EQ(det.total_records(), 3u);
}

// ---------------------------------------------------------------------------
// reset() clears all state
// ---------------------------------------------------------------------------

TEST(SentimentCycleDetector, ResetClearsState) {
    SentimentCycleDetector::Config cfg;
    cfg.window_size = 64;
    cfg.max_lag     = 16;
    cfg.min_samples = 8;
    SentimentCycleDetector det(cfg);

    for (int i = 0; i < 128; ++i) det.record((i % 4 < 2) ? 1.0 : -1.0);
    det.reset();

    EXPECT_EQ(det.dominant_period(), 0);
    EXPECT_DOUBLE_EQ(det.cycle_strength(), 0.0);
    EXPECT_EQ(det.total_records(), 0u);
}

// ---------------------------------------------------------------------------
// update_config resets state
// ---------------------------------------------------------------------------

TEST(SentimentCycleDetector, UpdateConfigResetsState) {
    SentimentCycleDetector det;
    for (int i = 0; i < 64; ++i) det.record((i % 4 < 2) ? 1.0 : -1.0);

    SentimentCycleDetector::Config cfg2;
    cfg2.window_size = 32;
    det.update_config(cfg2);

    EXPECT_EQ(det.dominant_period(), 0);
    EXPECT_EQ(det.total_records(), 0u);
}

// ---------------------------------------------------------------------------
// to_stats_json() contains expected fields
// ---------------------------------------------------------------------------

TEST(SentimentCycleDetector, ToStatsJsonContainsFields) {
    SentimentCycleDetector det;
    det.record(0.5);
    std::string j = det.to_stats_json();
    EXPECT_NE(j.find("dominant_period"), std::string::npos);
    EXPECT_NE(j.find("cycle_strength"),  std::string::npos);
    EXPECT_NE(j.find("is_cyclic"),       std::string::npos);
    EXPECT_NE(j.find("total_records"),   std::string::npos);
}

// ---------------------------------------------------------------------------
// Concurrent record() is thread-safe
// ---------------------------------------------------------------------------

TEST(SentimentCycleDetector, ConcurrentRecordSafe) {
    SentimentCycleDetector::Config cfg;
    cfg.window_size = 64;
    cfg.min_samples = 16;
    SentimentCycleDetector det(cfg);

    std::atomic<bool> go{false};
    auto worker = [&] {
        while (!go.load()) {}
        for (int i = 0; i < 200; ++i)
            det.record(static_cast<double>(i % 4) * 0.5);
    };
    std::vector<std::thread> threads;
    for (int t = 0; t < 4; ++t) threads.emplace_back(worker);
    go.store(true);
    for (auto& t : threads) t.join();

    EXPECT_EQ(det.total_records(), 800u);
    // Should not crash; strength is a valid double.
    double cs = det.cycle_strength();
    EXPECT_GE(cs, -1.0);
    EXPECT_LE(cs, 1.0);
}

} // namespace

#else

TEST(SentimentCycleDetector, DisabledAtBuildTime) { SUCCEED(); }

#endif  // LLMQUANT_SENTIMENT_CYCLE_ENABLED
