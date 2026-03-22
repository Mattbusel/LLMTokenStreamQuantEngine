#include <gtest/gtest.h>

#ifdef LLMQUANT_NARRATIVE_CHANGE_ENABLED

#include "NarrativeChangeDetector.h"

#include <atomic>
#include <functional>
#include <string>
#include <thread>
#include <vector>

using namespace llmquant;

namespace {

static uint64_t H(const std::string& s) {
    return std::hash<std::string>{}(s);
}

// ---------------------------------------------------------------------------
// Construction
// ---------------------------------------------------------------------------

TEST(NarrativeChangeDetector, DefaultConstructs) {
    NarrativeChangeDetector ncd;
    EXPECT_NEAR(ncd.get_similarity(), 1.0, 1e-9);
    EXPECT_EQ(ncd.total_recorded(), 0u);
    EXPECT_EQ(ncd.total_breaks(), 0u);
}

TEST(NarrativeChangeDetector, ConfigConstructs) {
    NarrativeChangeDetector::Config cfg;
    cfg.window_size     = 32;
    cfg.break_threshold = 0.3;
    NarrativeChangeDetector ncd(cfg);
    auto c = ncd.get_config();
    EXPECT_EQ(c.window_size, 32);
    EXPECT_NEAR(c.break_threshold, 0.3, 1e-9);
}

// ---------------------------------------------------------------------------
// Identical streams → similarity ≈ 1.0
// ---------------------------------------------------------------------------

TEST(NarrativeChangeDetector, IdenticalWindowsMaxSimilarity) {
    NarrativeChangeDetector::Config cfg;
    cfg.window_size = 16;
    cfg.min_fill    = 16;
    NarrativeChangeDetector ncd(cfg);

    // Fill two windows with the same tokens.
    for (int i = 0; i < 32; ++i) ncd.record(H("bullish"));

    EXPECT_NEAR(ncd.get_similarity(), 1.0, 0.01);
}

// ---------------------------------------------------------------------------
// Completely disjoint streams → similarity ≈ 0.0
// ---------------------------------------------------------------------------

TEST(NarrativeChangeDetector, DisjointWindowsLowSimilarity) {
    NarrativeChangeDetector::Config cfg;
    cfg.window_size = 16;
    cfg.min_fill    = 16;
    NarrativeChangeDetector ncd(cfg);

    // First window: all "bullish".
    for (int i = 0; i < 16; ++i) ncd.record(H("bullish"));
    // Second window: almost fully replaced with disjoint "crash" tokens.
    // Use 15 (not 16) to avoid the re-snapshot boundary where window_a_
    // would be overwritten with window_b_ at exactly the 16th replacement,
    // which would reset similarity to 1.0.
    for (int i = 0; i < 15; ++i) ncd.record(H("crash"));

    EXPECT_LT(ncd.get_similarity(), 0.1);
}

// ---------------------------------------------------------------------------
// Break callback fires when similarity drops
// ---------------------------------------------------------------------------

TEST(NarrativeChangeDetector, BreakCallbackFires) {
    NarrativeChangeDetector::Config cfg;
    cfg.window_size     = 8;
    cfg.min_fill        = 8;
    cfg.break_threshold = 0.5;
    NarrativeChangeDetector ncd(cfg);

    std::atomic<int> fire_count{0};
    ncd.set_break_callback([&](double) { ++fire_count; });

    // Window A: all "bullish".
    for (int i = 0; i < 8; ++i) ncd.record(H("bullish"));
    // Window B: different tokens.
    for (int i = 0; i < 8; ++i) ncd.record(H("crash_" + std::to_string(i)));

    EXPECT_GT(fire_count.load(), 0);
    EXPECT_GT(ncd.total_breaks(), 0u);
}

// ---------------------------------------------------------------------------
// No break callback for similar windows
// ---------------------------------------------------------------------------

TEST(NarrativeChangeDetector, NoBreakForSimilarWindows) {
    NarrativeChangeDetector::Config cfg;
    cfg.window_size     = 16;
    cfg.min_fill        = 16;
    cfg.break_threshold = 0.3;
    NarrativeChangeDetector ncd(cfg);

    std::atomic<int> fires{0};
    ncd.set_break_callback([&](double) { ++fires; });

    for (int i = 0; i < 32; ++i) ncd.record(H("bullish"));

    EXPECT_EQ(fires.load(), 0);
}

// ---------------------------------------------------------------------------
// total_recorded increments
// ---------------------------------------------------------------------------

TEST(NarrativeChangeDetector, TotalRecordedIncrements) {
    NarrativeChangeDetector ncd;
    ncd.record(1);
    ncd.record(2);
    ncd.record(3);
    EXPECT_EQ(ncd.total_recorded(), 3u);
}

// ---------------------------------------------------------------------------
// reset() clears all state
// ---------------------------------------------------------------------------

TEST(NarrativeChangeDetector, ResetClearsState) {
    NarrativeChangeDetector ncd;
    for (int i = 0; i < 10; ++i) ncd.record(static_cast<uint64_t>(i));
    ncd.reset();
    EXPECT_EQ(ncd.total_recorded(), 0u);
    EXPECT_EQ(ncd.total_breaks(), 0u);
    EXPECT_NEAR(ncd.get_similarity(), 1.0, 1e-9);
}

// ---------------------------------------------------------------------------
// is_narrative_break
// ---------------------------------------------------------------------------

TEST(NarrativeChangeDetector, IsNarrativeBreakFalseInitially) {
    NarrativeChangeDetector ncd;
    EXPECT_FALSE(ncd.is_narrative_break());
}

// ---------------------------------------------------------------------------
// update_config clears data
// ---------------------------------------------------------------------------

TEST(NarrativeChangeDetector, UpdateConfigClearsData) {
    NarrativeChangeDetector ncd;
    for (int i = 0; i < 5; ++i) ncd.record(static_cast<uint64_t>(i));

    NarrativeChangeDetector::Config cfg2;
    cfg2.window_size = 32;
    ncd.update_config(cfg2);

    EXPECT_EQ(ncd.total_breaks(), 0u);
    EXPECT_NEAR(ncd.get_similarity(), 1.0, 1e-9);
}

// ---------------------------------------------------------------------------
// to_stats_json contains expected fields
// ---------------------------------------------------------------------------

TEST(NarrativeChangeDetector, ToStatsJsonContainsFields) {
    NarrativeChangeDetector ncd;
    ncd.record(1);
    std::string j = ncd.to_stats_json();
    EXPECT_NE(j.find("similarity"), std::string::npos);
    EXPECT_NE(j.find("is_break"), std::string::npos);
    EXPECT_NE(j.find("total_breaks"), std::string::npos);
    EXPECT_NE(j.find("total_recorded"), std::string::npos);
}

// ---------------------------------------------------------------------------
// Concurrent record() is safe
// ---------------------------------------------------------------------------

TEST(NarrativeChangeDetector, ConcurrentRecordSafe) {
    NarrativeChangeDetector::Config cfg;
    cfg.window_size = 32;
    NarrativeChangeDetector ncd(cfg);

    std::atomic<bool> go{false};
    auto worker = [&] {
        while (!go.load()) {}
        for (int i = 0; i < 300; ++i)
            ncd.record(static_cast<uint64_t>(i % 8));
    };

    std::vector<std::thread> threads;
    for (int t = 0; t < 4; ++t) threads.emplace_back(worker);
    go.store(true);
    for (auto& t : threads) t.join();

    EXPECT_EQ(ncd.total_recorded(), 1200u);
    double sim = ncd.get_similarity();
    EXPECT_GE(sim, 0.0);
    EXPECT_LE(sim, 1.0);
}

// ---------------------------------------------------------------------------
// Gradual topic shift → intermediate similarity
// ---------------------------------------------------------------------------

TEST(NarrativeChangeDetector, GradualShiftIntermediateSimilarity) {
    NarrativeChangeDetector::Config cfg;
    cfg.window_size = 16;
    cfg.min_fill    = 16;
    NarrativeChangeDetector ncd(cfg);

    // Window A: 8 "bullish" + 8 "crash".
    for (int i = 0; i < 8;  ++i) ncd.record(H("bullish"));
    for (int i = 0; i < 8;  ++i) ncd.record(H("crash"));
    // Window B: 4 "bullish" + 4 "crash" + 8 new tokens.
    for (int i = 0; i < 4;  ++i) ncd.record(H("bullish"));
    for (int i = 0; i < 4;  ++i) ncd.record(H("crash"));
    for (int i = 0; i < 8;  ++i) ncd.record(H("rate_hike_" + std::to_string(i)));

    double sim = ncd.get_similarity();
    // Some overlap → not 0, not 1.
    EXPECT_GT(sim, 0.0);
    EXPECT_LT(sim, 0.99);
}

} // namespace

#else

TEST(NarrativeChangeDetector, DisabledAtBuildTime) { SUCCEED(); }

#endif  // LLMQUANT_NARRATIVE_CHANGE_ENABLED
