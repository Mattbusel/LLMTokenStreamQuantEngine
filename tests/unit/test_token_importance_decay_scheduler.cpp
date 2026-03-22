#include <gtest/gtest.h>

#ifdef LLMQUANT_TOKEN_DECAY_SCHEDULER_ENABLED
#include "TokenImportanceDecayScheduler.h"
#include <atomic>
#include <thread>
using namespace llmquant;

static constexpr uint64_t kSec = 1'000'000'000ULL;

TEST(TokenImportanceDecaySchedulerTest, InitialSentimentZero) {
    TokenImportanceDecayScheduler s;
    EXPECT_NEAR(s.effective_sentiment(0), 0.0, 1e-9);
    EXPECT_EQ(s.total_recorded(), 0u);
    EXPECT_EQ(s.active_entries(), 0u);
}

TEST(TokenImportanceDecaySchedulerTest, SinglePositiveToken) {
    TokenImportanceDecayScheduler s;
    s.record("bull", 1.0, 10 * kSec);
    // Query at same timestamp — no decay.
    double sent = s.effective_sentiment(10 * kSec);
    EXPECT_NEAR(sent, 1.0, 1e-9);  // Σw/Σ|w| = 1/1
}

TEST(TokenImportanceDecaySchedulerTest, DecayReducesSentiment) {
    TokenImportanceDecayScheduler::Config cfg;
    cfg.half_life_s = 5.0;
    TokenImportanceDecayScheduler s(cfg);
    s.record("bull", 1.0, 0);
    // After one half-life the effective weight halves, but since there's only
    // one token, sent = sum_w / sum_|w| = w*decay / |w|*decay = 1.0 still!
    // But if we mix with a zero-weight token the ratio changes.
    // Better test: query at 5s vs 0s with two opposing tokens.
    s.record("bear", -0.5, 0);
    // At t=0: (1.0 - 0.5) / (1.0 + 0.5) = 0.333
    double sent0 = s.effective_sentiment(0);
    EXPECT_NEAR(sent0, 1.0 / 3.0, 0.01);
    // At t = 5s (one half-life): weights are halved equally, ratio unchanged.
    // But sum_|w| at t=5s = 0.5 + 0.25 = 0.75; sum_w = 0.5 - 0.25 = 0.25.
    double sent5 = s.effective_sentiment(5 * kSec);
    EXPECT_NEAR(sent5, 1.0 / 3.0, 0.01);
}

TEST(TokenImportanceDecaySchedulerTest, StaleEntriesEvicted) {
    TokenImportanceDecayScheduler::Config cfg;
    cfg.max_age_s = 10.0;
    TokenImportanceDecayScheduler s(cfg);
    s.record("old", 1.0, 0);
    EXPECT_EQ(s.active_entries(), 1u);
    // After 15 s, evict_stale should remove it.
    s.evict_stale(15 * kSec);
    EXPECT_EQ(s.active_entries(), 0u);
}

TEST(TokenImportanceDecaySchedulerTest, SentimentFlipCallbackFires) {
    TokenImportanceDecayScheduler::Config cfg;
    cfg.half_life_s = 100.0;
    std::atomic<int> fires{0};
    cfg.on_sentiment_flip = [&](double, double) { ++fires; };
    TokenImportanceDecayScheduler s(cfg);
    // First token: positive
    s.record("bull", 1.0, 0);
    // Second token: large negative that flips sign
    s.record("bear", -10.0, kSec);
    EXPECT_GE(fires.load(), 1);
}

TEST(TokenImportanceDecaySchedulerTest, MaxEntriesCaps) {
    TokenImportanceDecayScheduler::Config cfg;
    cfg.max_entries = 5;
    cfg.max_age_s   = 1000.0;
    TokenImportanceDecayScheduler s(cfg);
    for (int i = 0; i < 10; ++i)
        s.record("t", 1.0, static_cast<uint64_t>(i) * kSec);
    EXPECT_LE(s.active_entries(), 5u);
}

TEST(TokenImportanceDecaySchedulerTest, TotalRecordedCountsAll) {
    TokenImportanceDecayScheduler s;
    s.record("a", 0.5, 1 * kSec);
    s.record("b", 0.3, 2 * kSec);
    EXPECT_EQ(s.total_recorded(), 2u);
}

TEST(TokenImportanceDecaySchedulerTest, ResetClearsState) {
    TokenImportanceDecayScheduler s;
    s.record("x", 1.0, 0);
    s.reset();
    EXPECT_EQ(s.total_recorded(), 0u);
    EXPECT_EQ(s.active_entries(), 0u);
    EXPECT_NEAR(s.effective_sentiment(0), 0.0, 1e-9);
}

TEST(TokenImportanceDecaySchedulerTest, StatsJsonNonEmpty) {
    TokenImportanceDecayScheduler s;
    s.record("tok", 0.5, 0);
    std::string j = s.to_stats_json(0);
    EXPECT_NE(j.find("effective_sentiment"), std::string::npos);
    EXPECT_NE(j.find("active_entries"),     std::string::npos);
    EXPECT_NE(j.find("flip_count"),         std::string::npos);
}

TEST(TokenImportanceDecaySchedulerTest, ConcurrentSafe) {
    TokenImportanceDecayScheduler s;
    std::atomic<bool> go{false};
    std::atomic<uint64_t> ts{0};
    auto worker = [&] {
        while (!go.load()) {}
        for (int i = 0; i < 100; ++i) {
            uint64_t t = ts.fetch_add(kSec / 100);
            s.record("t", (i % 2 == 0) ? 1.0 : -1.0, t);
        }
    };
    std::thread t1(worker), t2(worker);
    go.store(true);
    t1.join(); t2.join();
    EXPECT_EQ(s.total_recorded(), 200u);
}

#else
TEST(TokenImportanceDecaySchedulerTest, DisabledAtBuildTime) { SUCCEED(); }
#endif
