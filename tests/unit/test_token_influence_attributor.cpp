#include <gtest/gtest.h>

#ifdef LLMQUANT_TOKEN_INFLUENCE_ENABLED

#include "TokenInfluenceAttributor.h"

#include <atomic>
#include <cmath>
#include <thread>
#include <vector>

using namespace llmquant;

namespace {

// ============================================================
// Default construction
// ============================================================

TEST(TokenInfluenceAttributor, DefaultConstructs) {
    TokenInfluenceAttributor attr;
    EXPECT_EQ(attr.window_size(), 0u);
    EXPECT_EQ(attr.total_recorded(), 0u);
    auto results = attr.attribute();
    EXPECT_TRUE(results.empty());
}

// ============================================================
// record() grows window up to capacity
// ============================================================

TEST(TokenInfluenceAttributor, WindowGrowsUpToCapacity) {
    TokenInfluenceAttributor::Config cfg;
    cfg.window_size = 4;
    TokenInfluenceAttributor attr(cfg);
    for (int i = 0; i < 10; ++i) attr.record("tok", 0.1);
    EXPECT_EQ(attr.window_size(), 4u);
    EXPECT_EQ(attr.total_recorded(), 10u);
}

// ============================================================
// High-weight token has highest absolute influence
// ============================================================

TEST(TokenInfluenceAttributor, HighWeightTokenTopResult) {
    TokenInfluenceAttributor::Config cfg;
    cfg.window_size = 8;
    cfg.top_k       = 3;
    TokenInfluenceAttributor attr(cfg);

    attr.record("small", 0.1);
    attr.record("small", 0.1);
    attr.record("small", 0.1);
    attr.record("large", 5.0);  // outlier — highest influence

    auto results = attr.attribute();
    ASSERT_FALSE(results.empty());
    EXPECT_EQ(results[0].token, "large");
    EXPECT_GT(results[0].abs_influence, 0.0);
}

// ============================================================
// Results are sorted by abs_influence descending
// ============================================================

TEST(TokenInfluenceAttributor, ResultsSortedDescending) {
    TokenInfluenceAttributor::Config cfg;
    cfg.window_size = 10;
    TokenInfluenceAttributor attr(cfg);
    attr.record("a", 0.1);
    attr.record("b", 1.0);
    attr.record("c", 0.5);

    auto results = attr.attribute();
    for (std::size_t i = 1; i < results.size(); ++i)
        EXPECT_GE(results[i-1].abs_influence, results[i].abs_influence);
}

// ============================================================
// cumulative_fraction reaches 1.0 for the last element
// ============================================================

TEST(TokenInfluenceAttributor, CumulativeFractionReachesOne) {
    TokenInfluenceAttributor::Config cfg;
    cfg.window_size = 10;
    cfg.top_k       = 10;  // get all results
    TokenInfluenceAttributor attr(cfg);
    for (int i = 0; i < 5; ++i)
        attr.record("t" + std::to_string(i), static_cast<double>(i + 1) * 0.2);

    auto results = attr.attribute();
    ASSERT_FALSE(results.empty());
    EXPECT_NEAR(results.back().cumulative_fraction, 1.0, 1e-9);
}

// ============================================================
// Visitor overload visits same results as vector overload
// ============================================================

TEST(TokenInfluenceAttributor, VisitorMatchesVector) {
    TokenInfluenceAttributor::Config cfg;
    cfg.window_size = 8;
    TokenInfluenceAttributor attr(cfg);
    attr.record("a", 0.3);
    attr.record("b", -0.5);
    attr.record("c", 0.7);

    auto vec_results = attr.attribute();
    std::vector<AttributionResult> visited;
    attr.attribute([&](const AttributionResult& r) { visited.push_back(r); });

    ASSERT_EQ(vec_results.size(), visited.size());
    for (std::size_t i = 0; i < vec_results.size(); ++i)
        EXPECT_NEAR(vec_results[i].abs_influence, visited[i].abs_influence, 1e-12);
}

// ============================================================
// dominant_token_fraction callback fires
// ============================================================

TEST(TokenInfluenceAttributor, DominantTokenCallbackFires) {
    TokenInfluenceAttributor::Config cfg;
    cfg.window_size            = 10;
    cfg.dominant_token_fraction = 0.5;  // fire if any token > 50% of total influence
    std::atomic<int> fires{0};
    std::string last_dominant;
    cfg.on_dominant_token = [&](const std::string& tok, double /*frac*/) {
        ++fires;
        last_dominant = tok;
    };
    TokenInfluenceAttributor attr(cfg);

    // Add many low-weight tokens first.
    for (int i = 0; i < 5; ++i) attr.record("noise", 0.01);
    // Add one very dominant token.
    attr.record("whale", 100.0);

    EXPECT_GT(fires.load(), 0);
    EXPECT_EQ(last_dominant, "whale");
}

// ============================================================
// reset() clears window
// ============================================================

TEST(TokenInfluenceAttributor, ResetClearsWindow) {
    TokenInfluenceAttributor attr;
    attr.record("x", 0.5);
    attr.reset();
    EXPECT_EQ(attr.window_size(), 0u);
    EXPECT_TRUE(attr.attribute().empty());
}

// ============================================================
// to_stats_json contains expected keys
// ============================================================

TEST(TokenInfluenceAttributor, ToStatsJsonContainsKeys) {
    TokenInfluenceAttributor attr;
    attr.record("bull", 0.8);
    std::string json = attr.to_stats_json();
    EXPECT_NE(json.find("total_recorded"),   std::string::npos);
    EXPECT_NE(json.find("dominant_events"),  std::string::npos);
    EXPECT_NE(json.find("top_tokens"),       std::string::npos);
    EXPECT_NE(json.find("influence"),        std::string::npos);
}

// ============================================================
// Concurrent record() is thread-safe
// ============================================================

TEST(TokenInfluenceAttributor, ConcurrentRecordSafe) {
    TokenInfluenceAttributor::Config cfg;
    cfg.window_size             = 32;
    cfg.dominant_token_fraction = 1.0;  // disable callback to avoid races
    TokenInfluenceAttributor attr(cfg);

    std::vector<std::thread> threads;
    for (int t = 0; t < 4; ++t) {
        threads.emplace_back([&, t] {
            for (int i = 0; i < 50; ++i)
                attr.record("t" + std::to_string(t),
                            static_cast<double>(i) * 0.01);
        });
    }
    for (auto& th : threads) th.join();

    EXPECT_EQ(attr.total_recorded(), 200u);
    // attribute() must not crash.
    auto results = attr.attribute();
    EXPECT_LE(results.size(), cfg.window_size);
}

} // namespace

#else

TEST(TokenInfluenceAttributor, DisabledAtBuildTime) { SUCCEED(); }

#endif  // LLMQUANT_TOKEN_INFLUENCE_ENABLED
