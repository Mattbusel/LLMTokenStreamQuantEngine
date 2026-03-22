#include <gtest/gtest.h>

#ifdef LLMQUANT_NARRATIVE_ENTROPY_CLOCK_ENABLED

#include "NarrativeEntropyClock.h"

#include <atomic>
#include <cmath>
#include <thread>
#include <vector>

using namespace llmquant;

namespace {

// -----------------------------------------------------------------------
// Default state
// -----------------------------------------------------------------------

TEST(NarrativeEntropyClock, DefaultState) {
    NarrativeEntropyClock c;
    EXPECT_DOUBLE_EQ(c.accumulated_kl(), 0.0);
    EXPECT_EQ(c.exhaustion_events(), 0u);
    EXPECT_EQ(c.total_records(), 0u);
}

// -----------------------------------------------------------------------
// Total records increments
// -----------------------------------------------------------------------

TEST(NarrativeEntropyClock, TotalRecordsIncrements) {
    NarrativeEntropyClock c;
    c.record(0.1);
    c.record(0.2);
    c.record(0.3);
    EXPECT_EQ(c.total_records(), 3u);
}

// -----------------------------------------------------------------------
// KL accumulates (surprisal is positive)
// -----------------------------------------------------------------------

TEST(NarrativeEntropyClock, KLAccumulates) {
    NarrativeEntropyClock c;
    c.record(0.5);
    EXPECT_GT(c.accumulated_kl(), 0.0);
    EXPECT_GT(c.last_delta_kl(), 0.0);
}

// -----------------------------------------------------------------------
// Exhaustion callback fires when budget reached
// -----------------------------------------------------------------------

TEST(NarrativeEntropyClock, ExhaustionCallbackFires) {
    NarrativeEntropyClock::Config cfg;
    cfg.budget_nats = 2.0;  // small budget
    cfg.min_samples = 2;
    std::atomic<int> fires{0};
    cfg.on_budget_exhausted = [&](double) { ++fires; };
    NarrativeEntropyClock c(cfg);
    // Feed many diverse observations to rapidly accumulate KL
    for (int i = 0; i < 50; ++i)
        c.record(i % 2 == 0 ? 0.9 : -0.9);  // alternates bins → high surprisal each step
    EXPECT_GE(fires.load(), 1);
    EXPECT_GT(c.exhaustion_events(), 0u);
}

// -----------------------------------------------------------------------
// After exhaustion, accumulated KL resets to near 0
// -----------------------------------------------------------------------

TEST(NarrativeEntropyClock, AccumKLResetsAfterExhaustion) {
    NarrativeEntropyClock::Config cfg;
    cfg.budget_nats = 2.0;
    cfg.min_samples = 2;
    NarrativeEntropyClock c(cfg);
    // Drive to exhaustion
    for (int i = 0; i < 100; ++i)
        c.record(i % 2 == 0 ? 0.8 : -0.8);
    // After exhaustion, accumulated KL was reset; confirm it's < budget
    EXPECT_LT(c.accumulated_kl(), cfg.budget_nats);
}

// -----------------------------------------------------------------------
// Uniform signal (same bin) → low surprisal per step
// -----------------------------------------------------------------------

TEST(NarrativeEntropyClock, UniformSignalLowKL) {
    NarrativeEntropyClock::Config cfg;
    cfg.budget_nats = 10.0;
    cfg.min_samples = 1;
    NarrativeEntropyClock c(cfg);
    // Feed 20 identical values — reference distribution adapts, surprisal drops
    for (int i = 0; i < 20; ++i) c.record(0.5);
    // After 20 identical steps, delta_kl should be very small
    EXPECT_LT(c.last_delta_kl(), 1.0);
}

// -----------------------------------------------------------------------
// KL update callback fires on each record
// -----------------------------------------------------------------------

TEST(NarrativeEntropyClock, KLUpdateCallbackFires) {
    NarrativeEntropyClock::Config cfg;
    std::atomic<int> updates{0};
    cfg.on_kl_update = [&](double, double) { ++updates; };
    NarrativeEntropyClock c(cfg);
    c.record(0.1);
    c.record(0.2);
    c.record(0.3);
    EXPECT_EQ(updates.load(), 3);
}

// -----------------------------------------------------------------------
// reset() clears state
// -----------------------------------------------------------------------

TEST(NarrativeEntropyClock, ResetClearsState) {
    NarrativeEntropyClock c;
    for (int i = 0; i < 10; ++i) c.record(static_cast<double>(i) * 0.1);
    c.reset();
    EXPECT_EQ(c.total_records(), 0u);
    EXPECT_DOUBLE_EQ(c.accumulated_kl(), 0.0);
    EXPECT_EQ(c.exhaustion_events(), 0u);
}

// -----------------------------------------------------------------------
// update_config resets state
// -----------------------------------------------------------------------

TEST(NarrativeEntropyClock, UpdateConfigResetsState) {
    NarrativeEntropyClock c;
    for (int i = 0; i < 5; ++i) c.record(0.5);
    NarrativeEntropyClock::Config cfg2;
    cfg2.budget_nats = 5.0;
    c.update_config(cfg2);
    EXPECT_EQ(c.total_records(), 0u);
    EXPECT_DOUBLE_EQ(c.accumulated_kl(), 0.0);
}

// -----------------------------------------------------------------------
// to_stats_json contains expected keys
// -----------------------------------------------------------------------

TEST(NarrativeEntropyClock, ToStatsJsonContainsKeys) {
    NarrativeEntropyClock c;
    c.record(0.3);
    std::string j = c.to_stats_json();
    EXPECT_NE(j.find("accumulated_kl"),   std::string::npos);
    EXPECT_NE(j.find("last_delta_kl"),    std::string::npos);
    EXPECT_NE(j.find("exhaustion_events"), std::string::npos);
    EXPECT_NE(j.find("total_records"),    std::string::npos);
}

// -----------------------------------------------------------------------
// Concurrent record() is thread-safe
// -----------------------------------------------------------------------

TEST(NarrativeEntropyClock, ConcurrentRecordSafe) {
    NarrativeEntropyClock c;
    std::vector<std::thread> threads;
    for (int t = 0; t < 4; ++t) {
        threads.emplace_back([&, t] {
            for (int i = 0; i < 250; ++i)
                c.record(static_cast<double>(t % 2 == 0 ? i : -i) * 0.01);
        });
    }
    for (auto& th : threads) th.join();
    EXPECT_EQ(c.total_records(), 1000u);
    EXPECT_TRUE(std::isfinite(c.accumulated_kl()) || c.accumulated_kl() >= 0.0);
}

} // namespace

#else

TEST(NarrativeEntropyClock, DisabledAtBuildTime) { SUCCEED(); }

#endif  // LLMQUANT_NARRATIVE_ENTROPY_CLOCK_ENABLED
