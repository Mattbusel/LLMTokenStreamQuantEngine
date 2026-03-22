#include <gtest/gtest.h>

#ifdef LLMQUANT_TOKEN_IB_ENABLED

#include "TokenInformationBottleneck.h"

#include <atomic>
#include <cmath>
#include <thread>
#include <vector>

using namespace llmquant;

namespace {

// ---------------------------------------------------------------------------
// Default state
// ---------------------------------------------------------------------------

TEST(TokenInformationBottleneck, DefaultState) {
    TokenInformationBottleneck tib;
    EXPECT_EQ(tib.total_records(), 0u);
    EXPECT_EQ(tib.distinct_tokens(), 0);
    EXPECT_EQ(tib.flagged_count(), 0);
}

// ---------------------------------------------------------------------------
// Records increment total and distinct count
// ---------------------------------------------------------------------------

TEST(TokenInformationBottleneck, RecordsIncrementCounters) {
    TokenInformationBottleneck tib;
    tib.record("bull", 1.0, 0.05);
    tib.record("bear", -1.0, -0.05);
    tib.record("bull", 1.0, 0.05);
    EXPECT_EQ(tib.total_records(), 3u);
    EXPECT_EQ(tib.distinct_tokens(), 2);
}

// ---------------------------------------------------------------------------
// Observation count per token
// ---------------------------------------------------------------------------

TEST(TokenInformationBottleneck, ObservationCountPerToken) {
    TokenInformationBottleneck tib;
    for (int i = 0; i < 5; ++i) tib.record("tok", 0.5, 0.01);
    EXPECT_EQ(tib.observation_count("tok"), 5);
    EXPECT_EQ(tib.observation_count("missing"), 0);
}

// ---------------------------------------------------------------------------
// IB score is 0 below min_observations
// ---------------------------------------------------------------------------

TEST(TokenInformationBottleneck, NoScoreBeforeMinObs) {
    TokenInformationBottleneck::Config cfg;
    cfg.min_observations = 10;
    TokenInformationBottleneck tib(cfg);
    for (int i = 0; i < 5; ++i) tib.record("t", 1.0, 0.1);
    EXPECT_DOUBLE_EQ(tib.ib_score("t"), 0.0);
}

// ---------------------------------------------------------------------------
// High-relevance token has positive score after sufficient observations
// ---------------------------------------------------------------------------

TEST(TokenInformationBottleneck, HighRelevanceTokenScoresPositive) {
    TokenInformationBottleneck::Config cfg;
    cfg.min_observations = 8;
    cfg.window_size      = 32;
    TokenInformationBottleneck tib(cfg);
    // Perfectly correlated: weight and bias shift always move together.
    for (int i = 0; i < 20; ++i) {
        double w = (i % 2 == 0) ? 1.0 : -1.0;
        tib.record("sig", w, w * 0.1);
    }
    EXPECT_GT(tib.ib_score("sig"), 0.0);
    EXPECT_GT(tib.relevance("sig"), 0.5);
}

// ---------------------------------------------------------------------------
// Complexity is bounded [0, 1]
// ---------------------------------------------------------------------------

TEST(TokenInformationBottleneck, ComplexityBounded) {
    TokenInformationBottleneck tib;
    for (int i = 0; i < 30; ++i)
        tib.record("rand", static_cast<double>(i % 7) * 0.2 - 0.6, 0.01);
    double c = tib.complexity("rand");
    EXPECT_GE(c, 0.0);
    EXPECT_LE(c, 1.0 + 1e-9);
}

// ---------------------------------------------------------------------------
// Uniform weight → high complexity; constant weight → near-zero complexity
// ---------------------------------------------------------------------------

TEST(TokenInformationBottleneck, ComplexityContrast) {
    TokenInformationBottleneck::Config cfg;
    cfg.min_observations = 8;
    cfg.window_size      = 32;
    cfg.n_weight_bins    = 8;
    TokenInformationBottleneck tib(cfg);

    for (int i = 0; i < 20; ++i)
        tib.record("uniform", static_cast<double>(i % 8) * 0.25 - 1.0, 0.0);
    for (int i = 0; i < 20; ++i)
        tib.record("constant", 0.5, 0.0);

    EXPECT_GT(tib.complexity("uniform"), tib.complexity("constant"));
}

// ---------------------------------------------------------------------------
// Low-score callback fires when score drops below threshold
// ---------------------------------------------------------------------------

TEST(TokenInformationBottleneck, LowScoreCallbackFires) {
    TokenInformationBottleneck::Config cfg;
    cfg.min_observations = 4;
    cfg.prune_threshold  = 1.0;  // very high — everything will be flagged
    cfg.window_size      = 16;
    std::atomic<int> fires{0};
    cfg.on_low_score = [&](const std::string&, double) { ++fires; };
    TokenInformationBottleneck tib(cfg);

    // Uncorrelated weight and bias → low relevance → low IB score.
    for (int i = 0; i < 10; ++i)
        tib.record("noisy", static_cast<double>(i % 3) * 0.5, 0.0);
    EXPECT_GT(fires.load(), 0);
    EXPECT_GT(tib.flagged_count(), 0);
}

// ---------------------------------------------------------------------------
// top_k_tokens returns sorted descending
// ---------------------------------------------------------------------------

TEST(TokenInformationBottleneck, TopKTokensSorted) {
    TokenInformationBottleneck::Config cfg;
    cfg.min_observations = 4;
    cfg.window_size      = 20;
    cfg.top_k            = 3;
    TokenInformationBottleneck tib(cfg);

    // Feed multiple tokens with different correlation patterns.
    for (int i = 0; i < 15; ++i) {
        double w = (i % 2 == 0) ? 1.0 : -1.0;
        tib.record("strong", w, w * 0.1);           // high relevance
        tib.record("weak",   w, static_cast<double>(i % 5) * 0.1 - 0.2);  // low relevance
        tib.record("noisy",  static_cast<double>(i % 7) * 0.3 - 1.0, 0.0);
    }

    auto top = tib.top_k_tokens();
    EXPECT_FALSE(top.empty());
    for (size_t i = 1; i < top.size(); ++i)
        EXPECT_GE(top[i-1].second, top[i].second);
}

// ---------------------------------------------------------------------------
// bottom_k_tokens returns sorted ascending
// ---------------------------------------------------------------------------

TEST(TokenInformationBottleneck, BottomKTokensSorted) {
    TokenInformationBottleneck::Config cfg;
    cfg.min_observations = 4;
    cfg.window_size      = 20;
    cfg.top_k            = 3;
    TokenInformationBottleneck tib(cfg);
    for (int i = 0; i < 15; ++i) {
        tib.record("a", static_cast<double>(i % 3), 0.0);
        tib.record("b", static_cast<double>(i % 5) * 0.1, static_cast<double>(i % 5) * 0.1);
        tib.record("c", 0.1, 0.01);
    }
    auto bot = tib.bottom_k_tokens();
    EXPECT_FALSE(bot.empty());
    for (size_t i = 1; i < bot.size(); ++i)
        EXPECT_LE(bot[i-1].second, bot[i].second);
}

// ---------------------------------------------------------------------------
// Reset clears all state
// ---------------------------------------------------------------------------

TEST(TokenInformationBottleneck, ResetClearsState) {
    TokenInformationBottleneck tib;
    for (int i = 0; i < 20; ++i) tib.record("t", 0.5, 0.1);
    tib.reset();
    EXPECT_EQ(tib.total_records(), 0u);
    EXPECT_EQ(tib.distinct_tokens(), 0);
    EXPECT_EQ(tib.flagged_count(), 0);
    EXPECT_DOUBLE_EQ(tib.ib_score("t"), 0.0);
}

// ---------------------------------------------------------------------------
// update_config resets state
// ---------------------------------------------------------------------------

TEST(TokenInformationBottleneck, UpdateConfigResetsState) {
    TokenInformationBottleneck tib;
    for (int i = 0; i < 20; ++i) tib.record("t", 0.5, 0.1);
    TokenInformationBottleneck::Config cfg2;
    cfg2.window_size = 16;
    tib.update_config(cfg2);
    EXPECT_EQ(tib.total_records(), 0u);
    EXPECT_EQ(tib.distinct_tokens(), 0);
}

// ---------------------------------------------------------------------------
// to_stats_json contains expected keys
// ---------------------------------------------------------------------------

TEST(TokenInformationBottleneck, ToStatsJsonContainsFields) {
    TokenInformationBottleneck tib;
    tib.record("x", 0.5, 0.1);
    std::string j = tib.to_stats_json();
    EXPECT_NE(j.find("total_records"),   std::string::npos);
    EXPECT_NE(j.find("distinct_tokens"), std::string::npos);
    EXPECT_NE(j.find("flagged_count"),   std::string::npos);
}

// ---------------------------------------------------------------------------
// Concurrent record() is thread-safe
// ---------------------------------------------------------------------------

TEST(TokenInformationBottleneck, ConcurrentRecordSafe) {
    TokenInformationBottleneck::Config cfg;
    cfg.window_size      = 32;
    cfg.min_observations = 4;
    TokenInformationBottleneck tib(cfg);

    std::atomic<bool> go{false};
    auto worker = [&](int id) {
        std::string tok = "t" + std::to_string(id % 3);
        while (!go.load()) {}
        for (int i = 0; i < 250; ++i)
            tib.record(tok, static_cast<double>(i % 5) * 0.2 - 0.5, static_cast<double>(i % 5) * 0.05);
    };
    std::vector<std::thread> threads;
    for (int t = 0; t < 4; ++t) threads.emplace_back(worker, t);
    go.store(true);
    for (auto& t : threads) t.join();

    EXPECT_EQ(tib.total_records(), 1000u);
    EXPECT_GE(tib.distinct_tokens(), 1);
}

}  // namespace

#else

TEST(TokenInformationBottleneck, DisabledAtBuildTime) { SUCCEED(); }

#endif  // LLMQUANT_TOKEN_IB_ENABLED
