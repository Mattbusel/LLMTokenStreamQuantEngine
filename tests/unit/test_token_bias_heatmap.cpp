#include <gtest/gtest.h>

#ifdef LLMQUANT_TOKEN_BIAS_HEATMAP_ENABLED

#include "TokenBiasHeatmap.h"

using namespace llmquant;

namespace {

TEST(TokenBiasHeatmap, DefaultConstructsEmpty) {
    TokenBiasHeatmap h;
    EXPECT_EQ(h.total_records(), 0u);
    EXPECT_EQ(h.distinct_tokens(), 0u);
}

TEST(TokenBiasHeatmap, RecordIncrementsTotalRecords) {
    TokenBiasHeatmap h;
    h.record("bullish", 0.5);
    h.record("bearish", -0.3);
    EXPECT_EQ(h.total_records(), 2u);
    EXPECT_EQ(h.distinct_tokens(), 2u);
}

TEST(TokenBiasHeatmap, SameTokenAccumulates) {
    TokenBiasHeatmap h;
    h.record("bull", 0.5);
    h.record("bull", 0.3);
    h.record("bull", -0.2);
    auto top = h.top_by_abs_contribution(1);
    ASSERT_EQ(top.size(), 1u);
    EXPECT_EQ(top[0].token, "bull");
    EXPECT_NEAR(top[0].total_bias, 0.6, 1e-9);
    EXPECT_NEAR(top[0].abs_contribution, 1.0, 1e-9);
    EXPECT_EQ(top[0].count, 3u);
    EXPECT_NEAR(top[0].mean_bias, 0.2, 1e-9);
}

TEST(TokenBiasHeatmap, TopByAbsContributionSortsDescending) {
    TokenBiasHeatmap h;
    h.record("small",  0.1);
    h.record("medium", 0.5);
    h.record("large",  0.9);
    auto top = h.top_by_abs_contribution(3);
    ASSERT_EQ(top.size(), 3u);
    EXPECT_EQ(top[0].token, "large");
    EXPECT_EQ(top[1].token, "medium");
    EXPECT_EQ(top[2].token, "small");
}

TEST(TokenBiasHeatmap, TopBullishSortsDescending) {
    TokenBiasHeatmap h;
    h.record("bear", -2.0);
    h.record("bull",  3.0);
    h.record("neut",  0.1);
    auto top = h.top_bullish(3);
    ASSERT_EQ(top.size(), 3u);
    EXPECT_EQ(top[0].token, "bull");
}

TEST(TokenBiasHeatmap, TopBearishSortsAscending) {
    TokenBiasHeatmap h;
    h.record("bear", -2.0);
    h.record("bull",  3.0);
    h.record("neut",  0.1);
    auto top = h.top_bearish(3);
    ASSERT_EQ(top.size(), 3u);
    EXPECT_EQ(top[0].token, "bear");
}

TEST(TokenBiasHeatmap, TopNLimitsResults) {
    TokenBiasHeatmap h;
    for (int i = 0; i < 20; ++i)
        h.record("tok" + std::to_string(i), static_cast<double>(i));
    auto top = h.top_by_abs_contribution(5);
    EXPECT_EQ(top.size(), 5u);
}

TEST(TokenBiasHeatmap, MaxTokensEvictsOldest) {
    TokenBiasHeatmap::Config cfg;
    cfg.max_tokens = 3;
    TokenBiasHeatmap h(cfg);
    h.record("a", 1.0);
    h.record("b", 1.0);
    h.record("c", 1.0);
    EXPECT_EQ(h.distinct_tokens(), 3u);
    h.record("d", 1.0);  // should evict one entry
    EXPECT_EQ(h.distinct_tokens(), 3u);
}

TEST(TokenBiasHeatmap, ResetClearsAll) {
    TokenBiasHeatmap h;
    h.record("bull", 1.0);
    h.reset();
    EXPECT_EQ(h.total_records(), 0u);
    EXPECT_EQ(h.distinct_tokens(), 0u);
}

TEST(TokenBiasHeatmap, ToStatsJsonContainsKeys) {
    TokenBiasHeatmap h;
    h.record("bull", 0.5);
    std::string json = h.to_stats_json();
    EXPECT_NE(json.find("total_records"),    std::string::npos);
    EXPECT_NE(json.find("distinct_tokens"),  std::string::npos);
    EXPECT_NE(json.find("top_tokens"),       std::string::npos);
    EXPECT_NE(json.find("abs_contribution"), std::string::npos);
}

} // namespace

#else

TEST(TokenBiasHeatmap, DisabledAtBuildTime) { SUCCEED(); }

#endif  // LLMQUANT_TOKEN_BIAS_HEATMAP_ENABLED
