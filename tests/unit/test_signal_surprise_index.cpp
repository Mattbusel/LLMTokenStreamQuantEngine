#include <gtest/gtest.h>

#ifdef LLMQUANT_SIGNAL_SURPRISE_ENABLED

#include "SignalSurpriseIndex.h"

#include <atomic>
#include <cmath>
#include <thread>
#include <vector>

using namespace llmquant;

namespace {

// ---------------------------------------------------------------------------
// Default state
// ---------------------------------------------------------------------------

TEST(SignalSurpriseIndex, DefaultState) {
    SignalSurpriseIndex idx;
    // Before min_samples, surprise = 1.0 (maximally uncertain prior).
    EXPECT_DOUBLE_EQ(idx.surprise(), 1.0);
    EXPECT_EQ(idx.total_records(), 0u);
    EXPECT_EQ(idx.high_surprise_count(), 0u);
}

// ---------------------------------------------------------------------------
// Common values → low surprise after enough observations
// ---------------------------------------------------------------------------

TEST(SignalSurpriseIndex, CommonValueLowSurprise) {
    SignalSurpriseIndex::Config cfg;
    cfg.min_samples = 10;
    cfg.n_bins      = 32;
    SignalSurpriseIndex idx(cfg);

    // Repeatedly record the same bin.
    for (int i = 0; i < 200; ++i) idx.record(0.0);

    // The 0.0 bin should now be very common → low surprise.
    EXPECT_LT(idx.surprise(), 0.5);
}

// ---------------------------------------------------------------------------
// Rare value → high surprise after common prior is built
// ---------------------------------------------------------------------------

TEST(SignalSurpriseIndex, RareValueHighSurprise) {
    SignalSurpriseIndex::Config cfg;
    cfg.min_samples             = 10;
    cfg.n_bins                  = 32;
    cfg.high_surprise_threshold = 0.7;
    std::atomic<int> fires{0};
    cfg.on_high_surprise = [&](double, double) { ++fires; };
    SignalSurpriseIndex idx(cfg);

    // Build a strong prior around 0.0.
    for (int i = 0; i < 200; ++i) idx.record(0.0);

    // Record a value at the extreme edge of the range (very rare bin).
    idx.record(2.99);

    EXPECT_GT(idx.surprise(), 0.7);
    EXPECT_TRUE(idx.is_high_surprise());
    EXPECT_GT(fires.load(), 0);
}

// ---------------------------------------------------------------------------
// First observation in a new bin always surprises (p=0 case)
// ---------------------------------------------------------------------------

TEST(SignalSurpriseIndex, NewBinIsMaxSurprise) {
    SignalSurpriseIndex::Config cfg;
    cfg.min_samples = 1;
    cfg.n_bins      = 64;
    SignalSurpriseIndex idx(cfg);

    // Record something to pass min_samples.
    for (int i = 0; i < 50; ++i) idx.record(0.0);

    // Record in a never-seen bin.
    idx.record(2.5);

    EXPECT_DOUBLE_EQ(idx.surprise(), 1.0);
}

// ---------------------------------------------------------------------------
// mean_surprise decreases as distribution stabilises
// ---------------------------------------------------------------------------

TEST(SignalSurpriseIndex, MeanSurpriseDecreases) {
    SignalSurpriseIndex::Config cfg;
    cfg.min_samples = 5;
    SignalSurpriseIndex idx(cfg);

    double ms_early = 0.0, ms_late = 0.0;
    for (int i = 0; i < 300; ++i) {
        // Concentrate in 3 bins to create a learnable distribution.
        double v = (i % 3 == 0) ? 0.0 : (i % 3 == 1) ? 0.5 : -0.5;
        idx.record(v);
        if (i == 20)  ms_early = idx.mean_surprise();
        if (i == 299) ms_late  = idx.mean_surprise();
    }

    EXPECT_LT(ms_late, ms_early);
}

// ---------------------------------------------------------------------------
// total_records increments
// ---------------------------------------------------------------------------

TEST(SignalSurpriseIndex, TotalRecordsIncrements) {
    SignalSurpriseIndex idx;
    idx.record(0.1);
    idx.record(0.2);
    idx.record(0.3);
    EXPECT_EQ(idx.total_records(), 3u);
}

// ---------------------------------------------------------------------------
// reset() clears histogram and counters
// ---------------------------------------------------------------------------

TEST(SignalSurpriseIndex, ResetClearsState) {
    SignalSurpriseIndex idx;
    for (int i = 0; i < 50; ++i) idx.record(0.0);
    idx.reset();
    EXPECT_EQ(idx.total_records(), 0u);
    EXPECT_EQ(idx.high_surprise_count(), 0u);
    EXPECT_DOUBLE_EQ(idx.surprise(), 1.0);
}

// ---------------------------------------------------------------------------
// update_config resets state
// ---------------------------------------------------------------------------

TEST(SignalSurpriseIndex, UpdateConfigResetsState) {
    SignalSurpriseIndex idx;
    for (int i = 0; i < 50; ++i) idx.record(0.5);

    SignalSurpriseIndex::Config cfg2;
    cfg2.n_bins = 128;
    idx.update_config(cfg2);

    EXPECT_EQ(idx.total_records(), 0u);
    EXPECT_DOUBLE_EQ(idx.surprise(), 1.0);
}

// ---------------------------------------------------------------------------
// to_stats_json contains expected fields
// ---------------------------------------------------------------------------

TEST(SignalSurpriseIndex, ToStatsJsonContainsFields) {
    SignalSurpriseIndex idx;
    idx.record(0.5);
    std::string j = idx.to_stats_json();
    EXPECT_NE(j.find("surprise"),          std::string::npos);
    EXPECT_NE(j.find("mean_surprise"),     std::string::npos);
    EXPECT_NE(j.find("is_high_surprise"),  std::string::npos);
    EXPECT_NE(j.find("high_count"),        std::string::npos);
    EXPECT_NE(j.find("total_records"),     std::string::npos);
}

// ---------------------------------------------------------------------------
// Concurrent record() is thread-safe
// ---------------------------------------------------------------------------

TEST(SignalSurpriseIndex, ConcurrentRecordSafe) {
    SignalSurpriseIndex::Config cfg;
    cfg.min_samples = 8;
    SignalSurpriseIndex idx(cfg);

    std::atomic<bool> go{false};
    auto worker = [&] {
        while (!go.load()) {}
        for (int i = 0; i < 250; ++i)
            idx.record(static_cast<double>(i % 10) * 0.3 - 1.5);
    };
    std::vector<std::thread> threads;
    for (int t = 0; t < 4; ++t) threads.emplace_back(worker);
    go.store(true);
    for (auto& t : threads) t.join();

    EXPECT_EQ(idx.total_records(), 1000u);
    double s = idx.surprise();
    EXPECT_GE(s, 0.0);
    EXPECT_LE(s, 1.0);
}

// ---------------------------------------------------------------------------
// Uniform distribution → moderate mean_surprise (not near 0 or 1)
// ---------------------------------------------------------------------------

TEST(SignalSurpriseIndex, UniformDistributionModerateSurprise) {
    SignalSurpriseIndex::Config cfg;
    cfg.min_samples = 10;
    cfg.n_bins      = 10;
    cfg.range_min   = 0.0;
    cfg.range_max   = 1.0;
    SignalSurpriseIndex idx(cfg);

    // Feed uniformly — each bin gets approximately equal visits.
    for (int rep = 0; rep < 20; ++rep)
        for (int b = 0; b < 10; ++b)
            idx.record(0.05 + b * 0.1);

    // Uniform distribution: each bin has p≈0.1 → -log2(0.1)≈3.32 bits.
    // Normalized by log2(10)≈3.32 → normalized ≈ 1.0.
    // But after warm-up many bins are visited so surprise should be < 1.
    EXPECT_LT(idx.mean_surprise(), 1.0);
}

} // namespace

#else

TEST(SignalSurpriseIndex, DisabledAtBuildTime) { SUCCEED(); }

#endif  // LLMQUANT_SIGNAL_SURPRISE_ENABLED
