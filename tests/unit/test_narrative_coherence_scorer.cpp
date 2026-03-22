#include <gtest/gtest.h>

#ifdef LLMQUANT_COHERENCE_SCORER_ENABLED
#include "NarrativeCoherenceScorer.h"
#include <atomic>
#include <cmath>
#include <thread>
using namespace llmquant;

TEST(NarrativeCoherenceScorerTest, InitialState) {
    NarrativeCoherenceScorer sc;
    EXPECT_NEAR(sc.coherence(), 0.0, 1e-9);
    EXPECT_FALSE(sc.is_incoherent());
    EXPECT_EQ(sc.total_records(), 0u);
}

TEST(NarrativeCoherenceScorerTest, TotalRecordsIncrements) {
    NarrativeCoherenceScorer sc;
    sc.record(0.5);
    sc.record(0.5);
    EXPECT_EQ(sc.total_records(), 2u);
}

TEST(NarrativeCoherenceScorerTest, ConsistentSignalHighCoherence) {
    // All positive bias → |μ|/σ >> 1
    NarrativeCoherenceScorer::Config cfg;
    cfg.window      = 20;
    cfg.min_samples = 5;
    cfg.low_threshold = 0.5;
    NarrativeCoherenceScorer sc(cfg);
    for (int i = 0; i < 20; ++i) sc.record(0.9 + 0.01 * i);
    EXPECT_GT(sc.coherence(), 1.0);
    EXPECT_FALSE(sc.is_incoherent());
}

TEST(NarrativeCoherenceScorerTest, MixedSignalLowCoherence) {
    // Alternating ±1 → μ≈0, σ≈1 → coherence ≈ 0
    NarrativeCoherenceScorer::Config cfg;
    cfg.window        = 20;
    cfg.min_samples   = 5;
    cfg.low_threshold = 0.5;
    NarrativeCoherenceScorer sc(cfg);
    for (int i = 0; i < 20; ++i) sc.record(i % 2 == 0 ? 1.0 : -1.0);
    EXPECT_LT(sc.coherence(), 0.5);
    EXPECT_TRUE(sc.is_incoherent());
}

TEST(NarrativeCoherenceScorerTest, DropCallbackFires) {
    NarrativeCoherenceScorer::Config cfg;
    cfg.window        = 10;
    cfg.min_samples   = 5;
    cfg.low_threshold = 2.0;  // very high → will fire with mixed signal
    std::atomic<int> fires{0};
    cfg.on_coherence_drop = [&](double) { ++fires; };
    NarrativeCoherenceScorer sc(cfg);
    for (int i = 0; i < 10; ++i) sc.record(i % 2 == 0 ? 1.0 : -1.0);
    EXPECT_GE(fires.load(), 1);
}

TEST(NarrativeCoherenceScorerTest, RecoverCallbackFires) {
    NarrativeCoherenceScorer::Config cfg;
    cfg.window             = 10;
    cfg.min_samples        = 5;
    cfg.low_threshold      = 2.0;   // fire drop on mixed signal
    cfg.recover_threshold  = 0.1;   // very low recover — fires easily
    std::atomic<int> drops{0}, recovers{0};
    cfg.on_coherence_drop    = [&](double) { ++drops; };
    cfg.on_coherence_recover = [&](double) { ++recovers; };
    NarrativeCoherenceScorer sc(cfg);
    // Mixed → drop
    for (int i = 0; i < 10; ++i) sc.record(i % 2 == 0 ? 1.0 : -1.0);
    // Consistent → recover
    for (int i = 0; i < 10; ++i) sc.record(0.9);
    EXPECT_GE(drops.load(), 1);
    EXPECT_GE(recovers.load(), 1);
}

TEST(NarrativeCoherenceScorerTest, RollingMeanApprox) {
    NarrativeCoherenceScorer::Config cfg;
    cfg.window      = 20;
    cfg.min_samples = 5;
    NarrativeCoherenceScorer sc(cfg);
    for (int i = 0; i < 20; ++i) sc.record(0.5);
    EXPECT_NEAR(sc.rolling_mean(), 0.5, 1e-6);
    EXPECT_NEAR(sc.rolling_stddev(), 0.0, 1e-6);
}

TEST(NarrativeCoherenceScorerTest, ResetClearsState) {
    NarrativeCoherenceScorer sc;
    for (int i = 0; i < 20; ++i) sc.record(0.5);
    sc.reset();
    EXPECT_EQ(sc.total_records(), 0u);
    EXPECT_NEAR(sc.coherence(), 0.0, 1e-9);
    EXPECT_FALSE(sc.is_incoherent());
}

TEST(NarrativeCoherenceScorerTest, StatsJsonHasFields) {
    NarrativeCoherenceScorer sc;
    for (int i = 0; i < 10; ++i) sc.record(0.5);
    std::string j = sc.to_stats_json();
    EXPECT_NE(j.find("coherence"),       std::string::npos);
    EXPECT_NE(j.find("rolling_mean"),    std::string::npos);
    EXPECT_NE(j.find("rolling_stddev"),  std::string::npos);
    EXPECT_NE(j.find("is_incoherent"),   std::string::npos);
    EXPECT_NE(j.find("total_records"),   std::string::npos);
}

TEST(NarrativeCoherenceScorerTest, ConcurrentSafe) {
    NarrativeCoherenceScorer::Config cfg;
    cfg.window      = 30;
    cfg.min_samples = 5;
    NarrativeCoherenceScorer sc(cfg);
    std::atomic<bool> go{false};
    auto worker = [&](int id) {
        while (!go.load()) {}
        for (int i = 0; i < 200; ++i)
            sc.record(id % 2 == 0 ? 0.5 : -0.5);
    };
    std::thread t1(worker, 0), t2(worker, 1);
    go.store(true);
    t1.join(); t2.join();
    EXPECT_EQ(sc.total_records(), 400u);
    EXPECT_GE(sc.coherence(), 0.0);
}

#else
TEST(NarrativeCoherenceScorerTest, DisabledAtBuildTime) { SUCCEED(); }
#endif
