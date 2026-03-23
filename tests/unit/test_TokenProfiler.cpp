// test_TokenProfiler.cpp — GTest unit tests for TokenProfiler.

#include "TokenProfiler.h"
#include <gtest/gtest.h>

using namespace llmquant;

// ─── Basic feed tests ────────────────────────────────────────────────────────

TEST(TokenProfilerTest, EmptyProfilerHasZeroTotals) {
    TokenProfiler p;
    EXPECT_EQ(p.total_tokens(), 0u);
    EXPECT_EQ(p.unique_count(), 0u);
}

TEST(TokenProfilerTest, FeedSingleTokenIncrementsTotals) {
    TokenProfiler p;
    p.feed("hello");
    EXPECT_EQ(p.total_tokens(), 1u);
    EXPECT_EQ(p.unique_count(), 1u);
}

TEST(TokenProfilerTest, FeedSameTokenRepeatedly) {
    TokenProfiler p;
    p.feed("word");
    p.feed("word");
    p.feed("word");
    EXPECT_EQ(p.total_tokens(), 3u);
    EXPECT_EQ(p.unique_count(), 1u);
}

TEST(TokenProfilerTest, FeedBatchIncrementsTotals) {
    TokenProfiler p;
    p.feed_batch({"a", "b", "c", "a"});
    EXPECT_EQ(p.total_tokens(), 4u);
    EXPECT_EQ(p.unique_count(), 3u);
}

TEST(TokenProfilerTest, ResetClearsAllState) {
    TokenProfiler p;
    p.feed_batch({"x", "y", "z"});
    p.reset();
    EXPECT_EQ(p.total_tokens(), 0u);
    EXPECT_EQ(p.unique_count(), 0u);
}

// ─── Entropy tests ───────────────────────────────────────────────────────────

TEST(TokenProfilerTest, EntropyZeroWhenEmpty) {
    TokenProfiler p;
    EXPECT_DOUBLE_EQ(p.entropy(), 0.0);
}

TEST(TokenProfilerTest, EntropyZeroForSingleUniqueToken) {
    TokenProfiler p;
    p.feed("only");
    p.feed("only");
    // P(only)=1.0, entropy = -1*log2(1) = 0
    EXPECT_NEAR(p.entropy(), 0.0, 1e-9);
}

TEST(TokenProfilerTest, EntropyMaxForUniformDistribution) {
    TokenProfiler p;
    // 4 unique tokens each once -> entropy = log2(4) = 2
    p.feed_batch({"a", "b", "c", "d"});
    EXPECT_NEAR(p.entropy(), 2.0, 1e-9);
}

TEST(TokenProfilerTest, EntropyIncreasesWithMoreUniqueTokens) {
    TokenProfiler p1, p2;
    p1.feed_batch({"a", "b"});       // 2 unique -> entropy=1.0
    p2.feed_batch({"a", "b", "c", "d"});  // 4 unique -> entropy=2.0
    EXPECT_LT(p1.entropy(), p2.entropy());
}

// ─── Profile tests ───────────────────────────────────────────────────────────

TEST(TokenProfilerTest, ProfileTotalTokensCorrect) {
    TokenProfiler p;
    p.feed_batch({"x", "y", "x", "z", "x"});
    auto sp = p.profile();
    EXPECT_EQ(sp.total_tokens, 5u);
}

TEST(TokenProfilerTest, ProfileUniqueTokensCorrect) {
    TokenProfiler p;
    p.feed_batch({"x", "y", "z"});
    auto sp = p.profile();
    EXPECT_EQ(sp.unique_tokens, 3u);
}

TEST(TokenProfilerTest, ProfileTopKLimitedToK) {
    TokenProfiler p;
    for (int i = 0; i < 20; ++i) {
        p.feed("token_" + std::to_string(i));
    }
    auto sp = p.profile(5);
    EXPECT_EQ(sp.top_k_tokens.size(), 5u);
}

TEST(TokenProfilerTest, ProfileTopKSortedByFrequencyDesc) {
    TokenProfiler p;
    p.feed("rare");
    p.feed("common");
    p.feed("common");
    p.feed("common");
    auto sp = p.profile(2);
    ASSERT_GE(sp.top_k_tokens.size(), 2u);
    EXPECT_GE(sp.top_k_tokens[0].frequency, sp.top_k_tokens[1].frequency);
}

TEST(TokenProfilerTest, ProfileFirstAndLastSeenCorrect) {
    TokenProfiler p;
    p.feed("a");  // idx 0
    p.feed("b");  // idx 1
    p.feed("a");  // idx 2
    auto sp = p.profile(10);
    const TokenProfile* a_prof = nullptr;
    for (const auto& tp : sp.top_k_tokens) {
        if (tp.token == "a") { a_prof = &tp; break; }
    }
    ASSERT_NE(a_prof, nullptr);
    EXPECT_EQ(a_prof->first_seen_idx, 0u);
    EXPECT_EQ(a_prof->last_seen_idx, 2u);
}

TEST(TokenProfilerTest, ProfileAvgPositionCorrect) {
    TokenProfiler p;
    p.feed("t");  // pos 0
    p.feed("t");  // pos 1
    // avg_position = (0+1)/2 = 0.5
    auto sp = p.profile(1);
    ASSERT_EQ(sp.top_k_tokens.size(), 1u);
    EXPECT_NEAR(sp.top_k_tokens[0].avg_position, 0.5, 1e-9);
}

TEST(TokenProfilerTest, FeedBatchEmptyDoesNothing) {
    TokenProfiler p;
    p.feed_batch({});
    EXPECT_EQ(p.total_tokens(), 0u);
}

TEST(TokenProfilerTest, ProfileAfterResetStartsFresh) {
    TokenProfiler p;
    p.feed_batch({"a", "b", "a"});
    p.reset();
    p.feed("c");
    auto sp = p.profile();
    EXPECT_EQ(sp.total_tokens, 1u);
    EXPECT_EQ(sp.unique_tokens, 1u);
}
