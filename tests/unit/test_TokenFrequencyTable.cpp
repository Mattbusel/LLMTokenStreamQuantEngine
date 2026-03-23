// test_TokenFrequencyTable.cpp — GTest unit tests for TokenFrequencyTable.

#include "TokenFrequencyTable.h"
#include <gtest/gtest.h>

using namespace llmquant;

// ─── Basic unigram tests ─────────────────────────────────────────────────────

TEST(TokenFrequencyTableTest, EmptyTableHasZeroTotals) {
    TokenFrequencyTable tft;
    EXPECT_EQ(tft.total_tokens(), 0u);
    EXPECT_EQ(tft.unique_count(), 0u);
}

TEST(TokenFrequencyTableTest, AddSingleTokenIncrementsTotals) {
    TokenFrequencyTable tft;
    tft.add_token("hello");
    EXPECT_EQ(tft.total_tokens(), 1u);
    EXPECT_EQ(tft.unique_count(), 1u);
    EXPECT_EQ(tft.frequency("hello"), 1u);
}

TEST(TokenFrequencyTableTest, AddSameTokenRepeatedly) {
    TokenFrequencyTable tft;
    tft.add_token("word");
    tft.add_token("word");
    tft.add_token("word");
    EXPECT_EQ(tft.total_tokens(), 3u);
    EXPECT_EQ(tft.unique_count(), 1u);
    EXPECT_EQ(tft.frequency("word"), 3u);
}

TEST(TokenFrequencyTableTest, FrequencyOfUnseenTokenIsZero) {
    TokenFrequencyTable tft;
    EXPECT_EQ(tft.frequency("ghost"), 0u);
}

TEST(TokenFrequencyTableTest, MultipleDistinctTokens) {
    TokenFrequencyTable tft;
    tft.add_token("a");
    tft.add_token("b");
    tft.add_token("c");
    tft.add_token("a");
    EXPECT_EQ(tft.total_tokens(), 4u);
    EXPECT_EQ(tft.unique_count(), 3u);
    EXPECT_EQ(tft.frequency("a"), 2u);
    EXPECT_EQ(tft.frequency("b"), 1u);
}

// ─── N-gram tests ─────────────────────────────────────────────────────────────

TEST(TokenFrequencyTableTest, AddBigramAndRetrieve) {
    TokenFrequencyTable tft;
    std::vector<std::string> bigram{"the", "cat"};
    tft.add_ngram(bigram, 2);
    EXPECT_EQ(tft.ngram_frequency(bigram), 1u);
}

TEST(TokenFrequencyTableTest, UnseenNgramFrequencyIsZero) {
    TokenFrequencyTable tft;
    EXPECT_EQ(tft.ngram_frequency({"foo", "bar"}), 0u);
}

TEST(TokenFrequencyTableTest, FeedSequenceBuildsUnigrams) {
    TokenFrequencyTable tft;
    tft.feed_sequence({"a", "b", "c"}, 1);
    EXPECT_EQ(tft.frequency("a"), 1u);
    EXPECT_EQ(tft.frequency("b"), 1u);
    EXPECT_EQ(tft.frequency("c"), 1u);
    EXPECT_EQ(tft.total_tokens(), 3u);
}

TEST(TokenFrequencyTableTest, FeedSequenceBuildsNgrams) {
    TokenFrequencyTable tft;
    tft.feed_sequence({"the", "cat", "sat"}, 3);
    EXPECT_GE(tft.ngram_frequency({"the", "cat"}), 1u);
    EXPECT_GE(tft.ngram_frequency({"cat", "sat"}), 1u);
    EXPECT_GE(tft.ngram_frequency({"the", "cat", "sat"}), 1u);
}

// ─── Top-K tests ─────────────────────────────────────────────────────────────

TEST(TokenFrequencyTableTest, TopKReturnsMostFrequent) {
    TokenFrequencyTable tft;
    tft.add_token("common");
    tft.add_token("common");
    tft.add_token("common");
    tft.add_token("rare");
    auto top = tft.top_k(1);
    ASSERT_EQ(top.size(), 1u);
    EXPECT_EQ(top[0].first, "common");
    EXPECT_EQ(top[0].second, 3u);
}

TEST(TokenFrequencyTableTest, TopKLargerThanVocab) {
    TokenFrequencyTable tft;
    tft.add_token("x");
    tft.add_token("y");
    auto top = tft.top_k(100);
    EXPECT_EQ(top.size(), 2u);
}

// ─── Conditional probability tests ───────────────────────────────────────────

TEST(TokenFrequencyTableTest, ConditionalProbabilityWithContext) {
    TokenFrequencyTable tft;
    // Feed "the cat" three times and "the dog" once
    tft.feed_sequence({"the", "cat"}, 2);
    tft.feed_sequence({"the", "cat"}, 2);
    tft.feed_sequence({"the", "cat"}, 2);
    tft.feed_sequence({"the", "dog"}, 2);

    // P(cat | the) = 3 / 4 = 0.75
    double p = tft.conditional_probability("cat", {"the"});
    EXPECT_NEAR(p, 0.75, 1e-9);
}

TEST(TokenFrequencyTableTest, ConditionalProbabilityUnseenContext) {
    TokenFrequencyTable tft;
    double p = tft.conditional_probability("word", {"unseen"});
    EXPECT_EQ(p, 0.0);
}

// ─── Reset test ───────────────────────────────────────────────────────────────

TEST(TokenFrequencyTableTest, ResetClearsAllState) {
    TokenFrequencyTable tft;
    tft.add_token("x");
    tft.add_token("y");
    tft.reset();
    EXPECT_EQ(tft.total_tokens(), 0u);
    EXPECT_EQ(tft.unique_count(), 0u);
    EXPECT_EQ(tft.frequency("x"), 0u);
}
