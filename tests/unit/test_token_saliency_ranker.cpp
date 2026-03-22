#include <gtest/gtest.h>
#include "TokenSaliencyRanker.h"
#include <cmath>

namespace llmquant::tests {

TEST(TokenSaliencyRanker, DefaultConstruction) {
    TokenSaliencyRanker sr;
    EXPECT_EQ(sr.total_records(), 0u);
}

TEST(TokenSaliencyRanker, RecordIncreasesCount) {
    TokenSaliencyRanker sr;
    sr.record(0, 0.1);
    sr.record(1, -0.1);
    sr.record(0, 0.2);
    EXPECT_EQ(sr.total_records(), 3u);
}

TEST(TokenSaliencyRanker, SaliencyZeroForUnseenToken) {
    TokenSaliencyRanker sr;
    EXPECT_DOUBLE_EQ(sr.saliency(42), 0.0);
}

TEST(TokenSaliencyRanker, BullishTokenHasPositiveSaliency) {
    TokenSaliencyRanker::Config cfg;
    cfg.min_token_count = 3;
    TokenSaliencyRanker sr(cfg);
    for (int i = 0; i < 10; ++i) sr.record(5, 0.8);  // consistently bullish
    EXPECT_GT(sr.saliency(5), 0.0);
}

TEST(TokenSaliencyRanker, BearishTokenHasNegativeSaliency) {
    TokenSaliencyRanker::Config cfg;
    cfg.min_token_count = 3;
    TokenSaliencyRanker sr(cfg);
    for (int i = 0; i < 10; ++i) sr.record(7, -0.8); // consistently bearish
    EXPECT_LT(sr.saliency(7), 0.0);
}

TEST(TokenSaliencyRanker, TopBullishNotEmpty) {
    TokenSaliencyRanker::Config cfg;
    cfg.min_token_count = 2;
    TokenSaliencyRanker sr(cfg);
    for (int i = 0; i < 5; ++i) sr.record(3, 0.5);
    auto top = sr.top_bullish();
    EXPECT_EQ(top[0], 3);
}

TEST(TokenSaliencyRanker, TopBearishNotEmpty) {
    TokenSaliencyRanker::Config cfg;
    cfg.min_token_count = 2;
    TokenSaliencyRanker sr(cfg);
    for (int i = 0; i < 5; ++i) sr.record(9, -0.5);
    auto top = sr.top_bearish();
    EXPECT_EQ(top[0], 9);
}

TEST(TokenSaliencyRanker, TopTokenChangeCallback) {
    TokenSaliencyRanker::Config cfg;
    cfg.min_token_count = 2;
    int changes = 0;
    cfg.on_top_token_change = [&](int, double) { ++changes; };
    TokenSaliencyRanker sr(cfg);
    for (int i = 0; i < 5; ++i) sr.record(1, 0.5);
    for (int i = 0; i < 5; ++i) sr.record(2, 0.9);
    EXPECT_GE(changes, 1);
}

TEST(TokenSaliencyRanker, Reset) {
    TokenSaliencyRanker sr;
    for (int i = 0; i < 10; ++i) sr.record(0, 0.1);
    sr.reset();
    EXPECT_EQ(sr.total_records(), 0u);
    EXPECT_DOUBLE_EQ(sr.saliency(0), 0.0);
}

TEST(TokenSaliencyRanker, ToStatsJson) {
    TokenSaliencyRanker sr;
    for (int i = 0; i < 10; ++i) sr.record(i % 8, 0.1);
    auto json = sr.to_stats_json();
    EXPECT_NE(json.find("top_bullish"), std::string::npos);
    EXPECT_NE(json.find("total_records"), std::string::npos);
}

TEST(TokenSaliencyRanker, OutOfBoundsTokenIdIgnored) {
    TokenSaliencyRanker sr;
    sr.record(-1, 0.5);
    sr.record(512, 0.5);
    EXPECT_EQ(sr.total_records(), 0u);
}

} // namespace llmquant::tests
