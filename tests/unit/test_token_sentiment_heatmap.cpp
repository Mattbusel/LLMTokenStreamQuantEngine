#include <gtest/gtest.h>

#ifdef LLMQUANT_SENTIMENT_HEATMAP_ENABLED
#include "TokenSentimentHeatmap.h"
#include <cmath>

using namespace llmquant;

TEST(TokenSentimentHeatmapTest, ConstructsOk) {
    TokenSentimentHeatmap h;
    EXPECT_EQ(h.token_count(), 0);
    EXPECT_EQ(h.total_records(), 0u);
}

TEST(TokenSentimentHeatmapTest, RecordNewToken) {
    TokenSentimentHeatmap h;
    h.record("bull", 0.5);
    EXPECT_EQ(h.token_count(), 1);
    EXPECT_EQ(h.total_records(), 1u);
}

TEST(TokenSentimentHeatmapTest, BucketCountsCorrect) {
    TokenSentimentHeatmap::Config cfg;
    cfg.num_buckets    = 10;
    cfg.min_sentiment  = -1.0;
    cfg.max_sentiment  = 1.0;
    TokenSentimentHeatmap h(cfg);

    // Sentiment 0.5 → bucket 7 (0.5 maps to frac=0.75, bucket=7.5 → 7)
    h.record("tok", 0.5);
    auto counts = h.bucket_counts("tok");
    ASSERT_EQ(static_cast<int>(counts.size()), 10);
    double total = 0.0;
    for (double c : counts) total += c;
    EXPECT_NEAR(total, 1.0, 1e-9);
}

TEST(TokenSentimentHeatmapTest, SentimentExtremesClamped) {
    TokenSentimentHeatmap h;
    h.record("tok", -999.0); // clamps to bucket 0
    h.record("tok",  999.0); // clamps to bucket num_buckets-1
    auto counts = h.bucket_counts("tok");
    EXPECT_GT(counts.front(), 0.0);
    EXPECT_GT(counts.back(), 0.0);
}

TEST(TokenSentimentHeatmapTest, PeakBucketCorrect) {
    TokenSentimentHeatmap::Config cfg;
    cfg.num_buckets   = 10;
    cfg.min_sentiment = -1.0;
    cfg.max_sentiment =  1.0;
    TokenSentimentHeatmap h(cfg);

    for (int i = 0; i < 10; ++i) h.record("bull", 0.9);  // high sentiment → high bucket
    for (int i = 0; i < 2;  ++i) h.record("bull", -0.9); // low sentiment → low bucket

    int peak = h.peak_bucket("bull");
    EXPECT_GE(peak, 5); // majority in upper half
}

TEST(TokenSentimentHeatmapTest, DominantSentimentPositive) {
    TokenSentimentHeatmap h;
    for (int i = 0; i < 10; ++i) h.record("bull", 0.8);
    EXPECT_DOUBLE_EQ(h.dominant_sentiment("bull"), 1.0);
}

TEST(TokenSentimentHeatmapTest, DominantSentimentNegative) {
    TokenSentimentHeatmap h;
    for (int i = 0; i < 10; ++i) h.record("bear", -0.8);
    EXPECT_DOUBLE_EQ(h.dominant_sentiment("bear"), -1.0);
}

TEST(TokenSentimentHeatmapTest, UnknownTokenReturnsEmpty) {
    TokenSentimentHeatmap h;
    EXPECT_TRUE(h.bucket_counts("unknown").empty());
    EXPECT_EQ(h.peak_bucket("unknown"), -1);
    EXPECT_DOUBLE_EQ(h.dominant_sentiment("unknown"), 0.0);
}

TEST(TokenSentimentHeatmapTest, MultipleTokensTrackedIndependently) {
    TokenSentimentHeatmap h;
    h.record("alpha", 0.9);
    h.record("beta", -0.9);
    EXPECT_EQ(h.token_count(), 2);
    EXPECT_DOUBLE_EQ(h.dominant_sentiment("alpha"), 1.0);
    EXPECT_DOUBLE_EQ(h.dominant_sentiment("beta"), -1.0);
}

TEST(TokenSentimentHeatmapTest, ForgettingDecaysCounts) {
    TokenSentimentHeatmap::Config cfg;
    cfg.forgetting = 0.5; // aggressive forgetting
    TokenSentimentHeatmap h(cfg);
    for (int i = 0; i < 100; ++i) h.record("tok", 0.5);
    auto counts = h.bucket_counts("tok");
    double total = 0.0;
    for (double c : counts) total += c;
    // With 50% forgetting, counts should saturate far below 100.
    EXPECT_LT(total, 10.0);
}

TEST(TokenSentimentHeatmapTest, BucketCentreIsWithinRange) {
    TokenSentimentHeatmap::Config cfg;
    cfg.num_buckets   = 10;
    cfg.min_sentiment = -1.0;
    cfg.max_sentiment =  1.0;
    TokenSentimentHeatmap h(cfg);
    for (int i = 0; i < 10; ++i) {
        double c = h.bucket_centre(i);
        EXPECT_GE(c, -1.0);
        EXPECT_LE(c,  1.0);
    }
}

TEST(TokenSentimentHeatmapTest, ResetClearsAllTokens) {
    TokenSentimentHeatmap h;
    h.record("alpha", 0.5);
    h.record("beta", -0.5);
    h.reset();
    EXPECT_EQ(h.token_count(), 0);
}

TEST(TokenSentimentHeatmapTest, StatsJsonContainsRequiredKeys) {
    TokenSentimentHeatmap h;
    h.record("foo", 0.1);
    h.record("bar", -0.3);
    std::string json = h.to_stats_json();
    EXPECT_NE(json.find("token_count"),   std::string::npos);
    EXPECT_NE(json.find("total_records"), std::string::npos);
    EXPECT_NE(json.find("top_tokens"),    std::string::npos);
}

#else
TEST(TokenSentimentHeatmapTest, FeatureFlagDisabled) { SUCCEED(); }
#endif
