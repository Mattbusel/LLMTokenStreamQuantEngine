#include <gtest/gtest.h>

#ifdef LLMQUANT_SENTIMENT_PHASE_PORTRAIT_ENABLED

#include "SentimentPhasePortrait.h"

#include <atomic>
#include <thread>
#include <vector>

using namespace llmquant;

namespace {

// ---------------------------------------------------------------------------
// Default state
// ---------------------------------------------------------------------------

TEST(SentimentPhasePortrait, DefaultState) {
    SentimentPhasePortrait spp;
    EXPECT_EQ(spp.total_records(), 0u);
    EXPECT_EQ(spp.cell_transitions(), 0u);
    EXPECT_EQ(spp.attractor_row(), -1);  // no data
    EXPECT_EQ(spp.attractor_col(), -1);
    EXPECT_FALSE(spp.cycle_detected());
}

// ---------------------------------------------------------------------------
// Record increments total
// ---------------------------------------------------------------------------

TEST(SentimentPhasePortrait, TotalRecordsIncrements) {
    SentimentPhasePortrait spp;
    spp.record(0.0);
    spp.record(0.1);
    spp.record(-0.2);
    EXPECT_EQ(spp.total_records(), 3u);
}

// ---------------------------------------------------------------------------
// Boundary values map to valid cells
// ---------------------------------------------------------------------------

TEST(SentimentPhasePortrait, BoundaryCellMapping) {
    SentimentPhasePortrait::Config cfg;
    cfg.grid_size = 4;
    cfg.bias_min  = -1.0;
    cfg.bias_max  =  1.0;
    SentimentPhasePortrait spp(cfg);
    spp.record(-1.0);
    EXPECT_GE(spp.current_row(), 0);
    EXPECT_LT(spp.current_row(), cfg.grid_size);
    spp.record(1.0);
    EXPECT_GE(spp.current_row(), 0);
    EXPECT_LT(spp.current_row(), cfg.grid_size);
}

// ---------------------------------------------------------------------------
// Dwell fraction sums to 1 after observations
// ---------------------------------------------------------------------------

TEST(SentimentPhasePortrait, DwellFractionSumsToOne) {
    SentimentPhasePortrait::Config cfg;
    cfg.grid_size = 4;
    SentimentPhasePortrait spp(cfg);
    for (int i = 0; i < 80; ++i)
        spp.record(static_cast<double>(i % 4) * 0.5 - 0.75);

    double sum = 0.0;
    for (int r = 0; r < cfg.grid_size; ++r)
        for (int c = 0; c < cfg.grid_size; ++c)
            sum += spp.dwell_fraction(r, c);
    EXPECT_NEAR(sum, 1.0, 0.001);
}

// ---------------------------------------------------------------------------
// Attractor detected after sustained concentration
// ---------------------------------------------------------------------------

TEST(SentimentPhasePortrait, AttractorDetected) {
    SentimentPhasePortrait::Config cfg;
    cfg.grid_size           = 4;
    cfg.min_visits          = 10;
    cfg.attractor_threshold = 0.50;
    SentimentPhasePortrait spp(cfg);
    // Feed almost all observations into one state.
    for (int i = 0; i < 90; ++i) spp.record(0.0);  // one bias cluster
    for (int i = 0; i < 10; ++i) spp.record(static_cast<double>(i % 4) * 0.5 - 0.75);

    int ar = spp.attractor_row();
    int ac = spp.attractor_col();
    EXPECT_GE(ar, 0);
    EXPECT_GE(ac, 0);
}

// ---------------------------------------------------------------------------
// Cycle detected when alternating between two cells
// ---------------------------------------------------------------------------

TEST(SentimentPhasePortrait, CycleDetected) {
    SentimentPhasePortrait::Config cfg;
    cfg.grid_size    = 4;
    cfg.cycle_window = 10;
    cfg.bias_min     = -1.0;
    cfg.bias_max     =  1.0;
    SentimentPhasePortrait spp(cfg);
    // Alternate between two extreme biases to force two different cells.
    for (int i = 0; i < 20; ++i)
        spp.record((i % 2 == 0) ? -0.9 : 0.9);
    EXPECT_TRUE(spp.cycle_detected());
}

// ---------------------------------------------------------------------------
// Attractor callback fires when dominant cell changes
// ---------------------------------------------------------------------------

TEST(SentimentPhasePortrait, AttractorCallbackFires) {
    SentimentPhasePortrait::Config cfg;
    cfg.grid_size           = 4;
    cfg.min_visits          = 5;
    cfg.attractor_threshold = 0.30;
    std::atomic<int> fires{0};
    cfg.on_attractor_change = [&](int, int) { ++fires; };
    SentimentPhasePortrait spp(cfg);

    // Push into one attractor then shift to another.
    for (int i = 0; i < 30; ++i) spp.record(-0.8);
    for (int i = 0; i < 30; ++i) spp.record(0.8);

    EXPECT_GT(fires.load(), 0);
}

// ---------------------------------------------------------------------------
// Cycle callback fires
// ---------------------------------------------------------------------------

TEST(SentimentPhasePortrait, CycleCallbackFires) {
    SentimentPhasePortrait::Config cfg;
    cfg.grid_size    = 4;
    cfg.cycle_window = 10;
    cfg.bias_min     = -1.0;
    cfg.bias_max     =  1.0;
    std::atomic<int> fires{0};
    cfg.on_cycle_detected = [&](int, int, int, int) { ++fires; };
    SentimentPhasePortrait spp(cfg);
    for (int i = 0; i < 20; ++i)
        spp.record((i % 2 == 0) ? -0.9 : 0.9);
    EXPECT_GT(fires.load(), 0);
}

// ---------------------------------------------------------------------------
// Reset clears all state
// ---------------------------------------------------------------------------

TEST(SentimentPhasePortrait, ResetClearsState) {
    SentimentPhasePortrait spp;
    for (int i = 0; i < 50; ++i) spp.record(static_cast<double>(i % 5) * 0.4 - 0.8);
    spp.reset();
    EXPECT_EQ(spp.total_records(), 0u);
    EXPECT_EQ(spp.cell_transitions(), 0u);
    EXPECT_EQ(spp.attractor_row(), -1);
    EXPECT_FALSE(spp.cycle_detected());
    EXPECT_DOUBLE_EQ(spp.dwell_fraction(0, 0), 0.0);
}

// ---------------------------------------------------------------------------
// update_config resets state and changes grid
// ---------------------------------------------------------------------------

TEST(SentimentPhasePortrait, UpdateConfigResetsState) {
    SentimentPhasePortrait spp;
    for (int i = 0; i < 30; ++i) spp.record(0.1);

    SentimentPhasePortrait::Config cfg2;
    cfg2.grid_size = 6;
    spp.update_config(cfg2);

    EXPECT_EQ(spp.total_records(), 0u);
    EXPECT_EQ(spp.attractor_row(), -1);
}

// ---------------------------------------------------------------------------
// to_stats_json contains expected keys
// ---------------------------------------------------------------------------

TEST(SentimentPhasePortrait, ToStatsJsonContainsFields) {
    SentimentPhasePortrait spp;
    spp.record(0.3);
    std::string j = spp.to_stats_json();
    EXPECT_NE(j.find("row"),              std::string::npos);
    EXPECT_NE(j.find("col"),              std::string::npos);
    EXPECT_NE(j.find("attractor_row"),    std::string::npos);
    EXPECT_NE(j.find("cycle_detected"),   std::string::npos);
    EXPECT_NE(j.find("divergence_index"), std::string::npos);
    EXPECT_NE(j.find("total_records"),    std::string::npos);
}

// ---------------------------------------------------------------------------
// Divergence index in [0, 1]
// ---------------------------------------------------------------------------

TEST(SentimentPhasePortrait, DivergenceIndexBounded) {
    SentimentPhasePortrait::Config cfg;
    cfg.grid_size           = 4;
    cfg.min_visits          = 5;
    cfg.attractor_threshold = 0.10;
    SentimentPhasePortrait spp(cfg);
    for (int i = 0; i < 60; ++i)
        spp.record(static_cast<double>(i % 4) * 0.5 - 0.75);

    double di = spp.divergence_index();
    EXPECT_GE(di, 0.0);
    EXPECT_LE(di, 1.0);
}

// ---------------------------------------------------------------------------
// Concurrent record() is thread-safe
// ---------------------------------------------------------------------------

TEST(SentimentPhasePortrait, ConcurrentRecordSafe) {
    SentimentPhasePortrait::Config cfg;
    cfg.grid_size    = 4;
    cfg.min_visits   = 8;
    cfg.cycle_window = 20;
    SentimentPhasePortrait spp(cfg);

    std::atomic<bool> go{false};
    auto worker = [&] {
        while (!go.load()) {}
        for (int i = 0; i < 250; ++i)
            spp.record(static_cast<double>(i % 8) * 0.25 - 1.0);
    };
    std::vector<std::thread> threads;
    for (int t = 0; t < 4; ++t) threads.emplace_back(worker);
    go.store(true);
    for (auto& t : threads) t.join();

    EXPECT_EQ(spp.total_records(), 1000u);
    EXPECT_GE(spp.current_row(), 0);
    EXPECT_LT(spp.current_row(), cfg.grid_size);
}

}  // namespace

#else

TEST(SentimentPhasePortrait, DisabledAtBuildTime) { SUCCEED(); }

#endif  // LLMQUANT_SENTIMENT_PHASE_PORTRAIT_ENABLED
