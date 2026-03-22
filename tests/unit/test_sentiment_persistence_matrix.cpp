#include <gtest/gtest.h>

#ifdef LLMQUANT_SENTIMENT_PERSISTENCE_ENABLED

#include "SentimentPersistenceMatrix.h"

#include <atomic>
#include <thread>
#include <vector>

using namespace llmquant;

namespace {

// ---------------------------------------------------------------------------
// Default state
// ---------------------------------------------------------------------------

TEST(SentimentPersistenceMatrix, DefaultState) {
    SentimentPersistenceMatrix spm;
    EXPECT_EQ(spm.total_records(), 0u);
    EXPECT_EQ(spm.state_changes(), 0u);
    EXPECT_EQ(spm.predicted_state(), -1);  // no data yet
}

// ---------------------------------------------------------------------------
// First record seeds state without counting as a transition
// ---------------------------------------------------------------------------

TEST(SentimentPersistenceMatrix, FirstRecordSeeds) {
    SentimentPersistenceMatrix spm;
    spm.record(0.5);
    EXPECT_EQ(spm.total_records(), 1u);
    EXPECT_EQ(spm.state_changes(), 0u);  // seeding is not a transition
    EXPECT_GE(spm.current_state(), 0);
    EXPECT_LT(spm.current_state(), SentimentPersistenceMatrix::kDefaultStates);
}

// ---------------------------------------------------------------------------
// Bias-to-state mapping: extreme values land in boundary states
// ---------------------------------------------------------------------------

TEST(SentimentPersistenceMatrix, BoundaryStateMapping) {
    SentimentPersistenceMatrix::Config cfg;
    cfg.range_min = -1.0;
    cfg.range_max =  1.0;
    cfg.n_states  = 5;
    SentimentPersistenceMatrix spm(cfg);

    spm.record(-1.0);  // should map to state 0
    EXPECT_EQ(spm.current_state(), 0);

    spm.record(1.0);   // should map to state 4
    EXPECT_EQ(spm.current_state(), 4);
}

// ---------------------------------------------------------------------------
// Repeated same-state recording increases stickiness
// ---------------------------------------------------------------------------

TEST(SentimentPersistenceMatrix, RepeatedStateIncreasesStickiness) {
    SentimentPersistenceMatrix::Config cfg;
    cfg.n_states = 5;
    cfg.min_row_count = 4;
    SentimentPersistenceMatrix spm(cfg);

    // Repeatedly push a value that maps to the same state.
    for (int i = 0; i < 30; ++i) spm.record(0.0);  // state 2 (middle)

    // With 30 self-transitions, stickiness should be very high.
    EXPECT_GT(spm.stickiness(), 0.8);
}

// ---------------------------------------------------------------------------
// Transition callback fires on state change
// ---------------------------------------------------------------------------

TEST(SentimentPersistenceMatrix, StateChangeCallbackFires) {
    SentimentPersistenceMatrix::Config cfg;
    cfg.n_states      = 5;
    cfg.min_row_count = 1;
    std::atomic<int> fires{0};
    cfg.on_state_change = [&](int, int, double) { ++fires; };
    SentimentPersistenceMatrix spm(cfg);

    // Alternate between extreme positive and extreme negative.
    for (int i = 0; i < 20; ++i) spm.record((i % 2 == 0) ? -0.9 : 0.9);

    EXPECT_GT(fires.load(), 0);
    EXPECT_GT(spm.state_changes(), 0u);
}

// ---------------------------------------------------------------------------
// transition_probability returns plausible values after observations
// ---------------------------------------------------------------------------

TEST(SentimentPersistenceMatrix, TransitionProbabilitySum) {
    SentimentPersistenceMatrix::Config cfg;
    cfg.n_states = 3;
    cfg.range_min = -1.0;
    cfg.range_max =  1.0;
    cfg.min_row_count = 2;
    SentimentPersistenceMatrix spm(cfg);

    // Force state 1 (middle) many times.
    for (int i = 0; i < 20; ++i) spm.record(0.0);
    // Now occasionally transition.
    spm.record(-0.9);  // state 0
    spm.record(0.0);   // back to state 1
    spm.record(0.9);   // state 2

    // Sum of transition probabilities from state 1 should be ≈ 1.
    double sum = 0.0;
    for (int j = 0; j < cfg.n_states; ++j)
        sum += spm.transition_probability(1, j);
    EXPECT_NEAR(sum, 1.0, 0.01);
}

// ---------------------------------------------------------------------------
// predicted_state is within valid range after enough observations
// ---------------------------------------------------------------------------

TEST(SentimentPersistenceMatrix, PredictedStateValid) {
    SentimentPersistenceMatrix::Config cfg;
    cfg.n_states      = 5;
    cfg.min_row_count = 3;
    SentimentPersistenceMatrix spm(cfg);

    for (int i = 0; i < 50; ++i)
        spm.record((i % 2 == 0) ? 0.5 : 0.6);  // stays in states 3-4

    int pred = spm.predicted_state();
    EXPECT_GE(pred, 0);
    EXPECT_LT(pred, cfg.n_states);
}

// ---------------------------------------------------------------------------
// stationary_estimate sums to 1 after observations
// ---------------------------------------------------------------------------

TEST(SentimentPersistenceMatrix, StationaryEstimateSumsToOne) {
    SentimentPersistenceMatrix::Config cfg;
    cfg.n_states = 5;
    SentimentPersistenceMatrix spm(cfg);

    for (int i = 0; i < 100; ++i)
        spm.record(static_cast<double>(i % 5) * 0.4 - 0.8);

    double sum = 0.0;
    for (int s = 0; s < cfg.n_states; ++s)
        sum += spm.stationary_estimate(s);
    EXPECT_NEAR(sum, 1.0, 0.01);
}

// ---------------------------------------------------------------------------
// total_records increments correctly
// ---------------------------------------------------------------------------

TEST(SentimentPersistenceMatrix, TotalRecordsIncrements) {
    SentimentPersistenceMatrix spm;
    spm.record(0.0);
    spm.record(0.1);
    spm.record(0.2);
    EXPECT_EQ(spm.total_records(), 3u);
}

// ---------------------------------------------------------------------------
// reset() clears all state
// ---------------------------------------------------------------------------

TEST(SentimentPersistenceMatrix, ResetClearsState) {
    SentimentPersistenceMatrix::Config cfg;
    cfg.n_states = 5;
    SentimentPersistenceMatrix spm(cfg);

    for (int i = 0; i < 50; ++i) spm.record((i % 2 == 0) ? -0.9 : 0.9);
    spm.reset();

    EXPECT_EQ(spm.total_records(), 0u);
    EXPECT_EQ(spm.state_changes(), 0u);
    EXPECT_EQ(spm.predicted_state(), -1);
    EXPECT_DOUBLE_EQ(spm.stickiness(), 0.0);
}

// ---------------------------------------------------------------------------
// update_config resets state and resizes matrix
// ---------------------------------------------------------------------------

TEST(SentimentPersistenceMatrix, UpdateConfigResetsState) {
    SentimentPersistenceMatrix spm;
    for (int i = 0; i < 30; ++i) spm.record(static_cast<double>(i % 3) * 0.5 - 0.5);

    SentimentPersistenceMatrix::Config cfg2;
    cfg2.n_states = 3;
    spm.update_config(cfg2);

    EXPECT_EQ(spm.total_records(), 0u);
    EXPECT_EQ(spm.predicted_state(), -1);
}

// ---------------------------------------------------------------------------
// to_stats_json contains expected fields
// ---------------------------------------------------------------------------

TEST(SentimentPersistenceMatrix, ToStatsJsonContainsFields) {
    SentimentPersistenceMatrix spm;
    spm.record(0.5);
    std::string j = spm.to_stats_json();
    EXPECT_NE(j.find("current_state"),   std::string::npos);
    EXPECT_NE(j.find("predicted_state"), std::string::npos);
    EXPECT_NE(j.find("stickiness"),      std::string::npos);
    EXPECT_NE(j.find("state_changes"),   std::string::npos);
    EXPECT_NE(j.find("total_records"),   std::string::npos);
}

// ---------------------------------------------------------------------------
// Concurrent record() is thread-safe
// ---------------------------------------------------------------------------

TEST(SentimentPersistenceMatrix, ConcurrentRecordSafe) {
    SentimentPersistenceMatrix::Config cfg;
    cfg.n_states      = 5;
    cfg.min_row_count = 4;
    SentimentPersistenceMatrix spm(cfg);

    std::atomic<bool> go{false};
    auto worker = [&] {
        while (!go.load()) {}
        for (int i = 0; i < 250; ++i)
            spm.record(static_cast<double>(i % 5) * 0.4 - 0.8);
    };
    std::vector<std::thread> threads;
    for (int t = 0; t < 4; ++t) threads.emplace_back(worker);
    go.store(true);
    for (auto& t : threads) t.join();

    EXPECT_EQ(spm.total_records(), 1000u);
    int cur = spm.current_state();
    EXPECT_GE(cur, 0);
    EXPECT_LT(cur, cfg.n_states);
}

} // namespace

#else

TEST(SentimentPersistenceMatrix, DisabledAtBuildTime) { SUCCEED(); }

#endif  // LLMQUANT_SENTIMENT_PERSISTENCE_ENABLED
