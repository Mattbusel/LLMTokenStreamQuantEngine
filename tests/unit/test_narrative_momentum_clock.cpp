#include <gtest/gtest.h>

#ifdef LLMQUANT_NARRATIVE_CLOCK_ENABLED

#include "NarrativeMomentumClock.h"

#include <atomic>
#include <thread>
#include <vector>

using namespace llmquant;
using Q = NarrativeMomentumClock::Quadrant;

namespace {

// ---------------------------------------------------------------------------
// Default state
// ---------------------------------------------------------------------------

TEST(NarrativeMomentumClock, DefaultState) {
    NarrativeMomentumClock clk;
    EXPECT_EQ(clk.total_records(), 0u);
    EXPECT_EQ(clk.quadrant_transitions(), 0u);
    EXPECT_DOUBLE_EQ(clk.bias_ema(), 0.0);
    EXPECT_DOUBLE_EQ(clk.velocity_ema(), 0.0);
}

// ---------------------------------------------------------------------------
// First record seeds the EMA (no transition counted)
// ---------------------------------------------------------------------------

TEST(NarrativeMomentumClock, FirstRecordSeeds) {
    NarrativeMomentumClock clk;
    clk.record(0.5);
    EXPECT_EQ(clk.total_records(), 1u);
    EXPECT_EQ(clk.quadrant_transitions(), 0u);  // seeding is not a transition
    EXPECT_GT(clk.bias_ema(), 0.0);
}

// ---------------------------------------------------------------------------
// Sustained positive feed → Q1 Rising
// ---------------------------------------------------------------------------

TEST(NarrativeMomentumClock, SustainedPositiveFeedRising) {
    NarrativeMomentumClock::Config cfg;
    cfg.bias_alpha     = 0.30;
    cfg.velocity_alpha = 0.30;
    NarrativeMomentumClock clk(cfg);

    // Feed a ramping positive series so both bias_ema and velocity_ema go positive.
    for (int i = 1; i <= 50; ++i) clk.record(static_cast<double>(i) * 0.01);

    EXPECT_EQ(clk.quadrant(), Q::Rising);
    EXPECT_TRUE(clk.is_rising());
    EXPECT_GT(clk.bias_ema(), 0.0);
    EXPECT_GT(clk.velocity_ema(), 0.0);
}

// ---------------------------------------------------------------------------
// Sustained negative feed → Q3 Falling
// ---------------------------------------------------------------------------

TEST(NarrativeMomentumClock, SustainedNegativeFeedFalling) {
    NarrativeMomentumClock::Config cfg;
    cfg.bias_alpha     = 0.30;
    cfg.velocity_alpha = 0.30;
    NarrativeMomentumClock clk(cfg);

    for (int i = 1; i <= 50; ++i) clk.record(static_cast<double>(-i) * 0.01);

    EXPECT_EQ(clk.quadrant(), Q::Falling);
    EXPECT_TRUE(clk.is_falling());
    EXPECT_LT(clk.bias_ema(), 0.0);
    EXPECT_LT(clk.velocity_ema(), 0.0);
}

// ---------------------------------------------------------------------------
// Q1 → Q3 transition fires callback
// ---------------------------------------------------------------------------

TEST(NarrativeMomentumClock, TransitionCallbackFires) {
    NarrativeMomentumClock::Config cfg;
    cfg.bias_alpha     = 0.40;
    cfg.velocity_alpha = 0.40;
    std::atomic<int> fires{0};
    cfg.on_quadrant_change = [&](Q, Q, double, double) { ++fires; };
    NarrativeMomentumClock clk(cfg);

    // Drive into Q1 (Rising).
    for (int i = 0; i < 40; ++i) clk.record(1.0);
    // Then sharply reverse to drive into Q3 (Falling).
    for (int i = 0; i < 80; ++i) clk.record(-1.0);

    EXPECT_GT(fires.load(), 0);
    EXPECT_GT(clk.quadrant_transitions(), 0u);
}

// ---------------------------------------------------------------------------
// total_records increments
// ---------------------------------------------------------------------------

TEST(NarrativeMomentumClock, TotalRecordsIncrements) {
    NarrativeMomentumClock clk;
    clk.record(0.1);
    clk.record(0.2);
    clk.record(0.3);
    EXPECT_EQ(clk.total_records(), 3u);
}

// ---------------------------------------------------------------------------
// reset() clears all state
// ---------------------------------------------------------------------------

TEST(NarrativeMomentumClock, ResetClearsState) {
    NarrativeMomentumClock::Config cfg;
    cfg.bias_alpha = 0.30;
    NarrativeMomentumClock clk(cfg);

    for (int i = 0; i < 60; ++i) clk.record(1.0);
    clk.reset();

    EXPECT_EQ(clk.total_records(), 0u);
    EXPECT_EQ(clk.quadrant_transitions(), 0u);
    EXPECT_DOUBLE_EQ(clk.bias_ema(), 0.0);
    EXPECT_DOUBLE_EQ(clk.velocity_ema(), 0.0);
}

// ---------------------------------------------------------------------------
// update_config resets state
// ---------------------------------------------------------------------------

TEST(NarrativeMomentumClock, UpdateConfigResetsState) {
    NarrativeMomentumClock clk;
    for (int i = 0; i < 40; ++i) clk.record(0.5);

    NarrativeMomentumClock::Config cfg2;
    cfg2.bias_alpha = 0.05;
    clk.update_config(cfg2);

    EXPECT_EQ(clk.total_records(), 0u);
    EXPECT_DOUBLE_EQ(clk.bias_ema(), 0.0);
}

// ---------------------------------------------------------------------------
// to_stats_json contains expected fields
// ---------------------------------------------------------------------------

TEST(NarrativeMomentumClock, ToStatsJsonContainsFields) {
    NarrativeMomentumClock clk;
    clk.record(0.3);
    std::string j = clk.to_stats_json();
    EXPECT_NE(j.find("quadrant"),       std::string::npos);
    EXPECT_NE(j.find("bias_ema"),       std::string::npos);
    EXPECT_NE(j.find("velocity_ema"),   std::string::npos);
    EXPECT_NE(j.find("transitions"),    std::string::npos);
    EXPECT_NE(j.find("total_records"),  std::string::npos);
}

// ---------------------------------------------------------------------------
// Quadrant names appear in JSON
// ---------------------------------------------------------------------------

TEST(NarrativeMomentumClock, QuadrantNamesInJson) {
    NarrativeMomentumClock::Config cfg;
    cfg.bias_alpha     = 0.50;
    cfg.velocity_alpha = 0.50;
    NarrativeMomentumClock clk(cfg);

    // Drive Rising.
    for (int i = 0; i < 30; ++i) clk.record(1.0);
    EXPECT_NE(clk.to_stats_json().find("Rising"), std::string::npos);

    // Drive Falling.
    for (int i = 0; i < 60; ++i) clk.record(-1.0);
    EXPECT_NE(clk.to_stats_json().find("Falling"), std::string::npos);
}

// ---------------------------------------------------------------------------
// is_rising / is_falling flags match quadrant()
// ---------------------------------------------------------------------------

TEST(NarrativeMomentumClock, DirectionFlagsConsistent) {
    NarrativeMomentumClock::Config cfg;
    cfg.bias_alpha     = 0.40;
    cfg.velocity_alpha = 0.40;
    NarrativeMomentumClock clk(cfg);

    for (int i = 0; i < 40; ++i) clk.record(1.0);
    EXPECT_EQ(clk.is_rising(),  clk.quadrant() == Q::Rising);
    EXPECT_EQ(clk.is_falling(), clk.quadrant() == Q::Falling);

    for (int i = 0; i < 80; ++i) clk.record(-1.0);
    EXPECT_EQ(clk.is_rising(),  clk.quadrant() == Q::Rising);
    EXPECT_EQ(clk.is_falling(), clk.quadrant() == Q::Falling);
}

// ---------------------------------------------------------------------------
// Callback receives correct from/to quadrant pair
// ---------------------------------------------------------------------------

TEST(NarrativeMomentumClock, CallbackFromToCorrect) {
    NarrativeMomentumClock::Config cfg;
    cfg.bias_alpha     = 0.50;
    cfg.velocity_alpha = 0.50;

    Q last_from{}, last_to{};
    std::atomic<int> fires{0};
    cfg.on_quadrant_change = [&](Q from, Q to, double, double) {
        last_from = from;
        last_to   = to;
        ++fires;
    };
    NarrativeMomentumClock clk(cfg);

    // Establish Q1 (Rising).
    for (int i = 0; i < 30; ++i) clk.record(1.0);
    int fires_after_rising = fires.load();

    // Force a reversal through heavy negative feed.
    for (int i = 0; i < 60; ++i) clk.record(-1.0);

    EXPECT_GT(fires.load(), fires_after_rising);
    EXPECT_NE(last_from, last_to);
}

// ---------------------------------------------------------------------------
// Concurrent record() is thread-safe
// ---------------------------------------------------------------------------

TEST(NarrativeMomentumClock, ConcurrentRecordSafe) {
    NarrativeMomentumClock::Config cfg;
    cfg.bias_alpha     = 0.10;
    cfg.velocity_alpha = 0.20;
    NarrativeMomentumClock clk(cfg);

    std::atomic<bool> go{false};
    auto worker = [&] {
        while (!go.load()) {}
        for (int i = 0; i < 250; ++i)
            clk.record(static_cast<double>(i % 10) * 0.1 - 0.45);
    };
    std::vector<std::thread> threads;
    for (int t = 0; t < 4; ++t) threads.emplace_back(worker);
    go.store(true);
    for (auto& t : threads) t.join();

    EXPECT_EQ(clk.total_records(), 1000u);
    // Must not crash; bias_ema must be a finite double.
    double b = clk.bias_ema();
    EXPECT_GE(b, -10.0);
    EXPECT_LE(b,  10.0);
}

} // namespace

#else

TEST(NarrativeMomentumClock, DisabledAtBuildTime) { SUCCEED(); }

#endif  // LLMQUANT_NARRATIVE_CLOCK_ENABLED
