// test_LogitsProcessor.cpp — GTest suite for LogitsProcessor (15 tests).

#include "LogitsProcessor.h"

#include <cmath>
#include <gtest/gtest.h>
#include <numeric>
#include <vector>

using namespace llmquant;

// ---------------------------------------------------------------------------
// Fixture
// ---------------------------------------------------------------------------

class LogitsProcessorTest : public ::testing::Test {
protected:
    LogitsProcessor proc;

    std::vector<float> uniform_logits(size_t n, float val = 0.0f) {
        return std::vector<float>(n, val);
    }
};

// ---------------------------------------------------------------------------
// apply_temperature
// ---------------------------------------------------------------------------

TEST_F(LogitsProcessorTest, TemperatureOneIsNoop) {
    std::vector<float> logits = {1.0f, 2.0f, 3.0f};
    auto original = logits;
    proc.apply_temperature(logits, 1.0f);
    EXPECT_EQ(logits, original);
}

TEST_F(LogitsProcessorTest, TemperatureHalfDoublesLogits) {
    std::vector<float> logits = {2.0f, 4.0f};
    proc.apply_temperature(logits, 0.5f);
    EXPECT_FLOAT_EQ(logits[0], 4.0f);
    EXPECT_FLOAT_EQ(logits[1], 8.0f);
}

TEST_F(LogitsProcessorTest, TemperatureTwoHalvesLogits) {
    std::vector<float> logits = {4.0f, 8.0f};
    proc.apply_temperature(logits, 2.0f);
    EXPECT_FLOAT_EQ(logits[0], 2.0f);
    EXPECT_FLOAT_EQ(logits[1], 4.0f);
}

TEST_F(LogitsProcessorTest, TemperatureZeroThrows) {
    std::vector<float> logits = {1.0f, 2.0f};
    EXPECT_THROW(proc.apply_temperature(logits, 0.0f), std::invalid_argument);
}

// ---------------------------------------------------------------------------
// apply_repetition_penalty
// ---------------------------------------------------------------------------

TEST_F(LogitsProcessorTest, RepetitionPenaltyReducesPositiveLogit) {
    std::vector<float> logits = {2.0f, 1.0f, 0.5f};
    proc.apply_repetition_penalty(logits, {0}, 2.0f);
    EXPECT_FLOAT_EQ(logits[0], 1.0f);   // 2.0 / 2.0
    EXPECT_FLOAT_EQ(logits[1], 1.0f);   // unchanged
}

TEST_F(LogitsProcessorTest, RepetitionPenaltyAmplifiedNegativeLogit) {
    std::vector<float> logits = {-1.0f, 1.0f};
    proc.apply_repetition_penalty(logits, {0}, 2.0f);
    EXPECT_FLOAT_EQ(logits[0], -2.0f);  // -1.0 * 2.0
}

TEST_F(LogitsProcessorTest, RepetitionPenaltyOneIsNoop) {
    std::vector<float> logits = {3.0f, -2.0f};
    auto original = logits;
    proc.apply_repetition_penalty(logits, {0, 1}, 1.0f);
    EXPECT_EQ(logits, original);
}

// ---------------------------------------------------------------------------
// apply_top_k
// ---------------------------------------------------------------------------

TEST_F(LogitsProcessorTest, TopKKeepsKLargest) {
    std::vector<float> logits = {1.0f, 5.0f, 3.0f, 2.0f, 4.0f};
    proc.apply_top_k(logits, 2);
    // Expect that exactly 2 logits are not -inf
    int finite_count = 0;
    for (float l : logits) {
        if (l > -1e20f) ++finite_count;
    }
    EXPECT_EQ(finite_count, 2);
}

TEST_F(LogitsProcessorTest, TopKLargerThanSizeIsNoop) {
    std::vector<float> logits = {1.0f, 2.0f, 3.0f};
    auto original = logits;
    proc.apply_top_k(logits, 10);
    EXPECT_EQ(logits, original);
}

// ---------------------------------------------------------------------------
// apply_top_p
// ---------------------------------------------------------------------------

TEST_F(LogitsProcessorTest, TopPOneIsNoop) {
    std::vector<float> logits = {1.0f, 2.0f, 3.0f};
    auto original = logits;
    proc.apply_top_p(logits, 1.0f);
    EXPECT_EQ(logits, original);
}

TEST_F(LogitsProcessorTest, TopPMasksLowProbabilityTokens) {
    // One very high logit and many tiny ones — top-p 0.5 should keep only the big one
    std::vector<float> logits = {10.0f, -5.0f, -5.0f, -5.0f, -5.0f};
    proc.apply_top_p(logits, 0.5f);
    // The first token should survive
    EXPECT_GT(logits[0], -1e20f);
}

// ---------------------------------------------------------------------------
// softmax
// ---------------------------------------------------------------------------

TEST_F(LogitsProcessorTest, SoftmaxSumsToOne) {
    std::vector<float> logits = {1.0f, 2.0f, 3.0f};
    auto probs = proc.softmax(logits);
    float sum = 0.0f;
    for (float p : probs) sum += p;
    EXPECT_NEAR(sum, 1.0f, 1e-5f);
}

TEST_F(LogitsProcessorTest, SoftmaxUniformLogitsSameProbs) {
    std::vector<float> logits = {0.0f, 0.0f, 0.0f, 0.0f};
    auto probs = proc.softmax(logits);
    for (float p : probs) {
        EXPECT_NEAR(p, 0.25f, 1e-5f);
    }
}

// ---------------------------------------------------------------------------
// sample / greedy
// ---------------------------------------------------------------------------

TEST_F(LogitsProcessorTest, GreedyReturnsArgmax) {
    std::vector<float> logits = {0.1f, 0.9f, 0.5f};
    EXPECT_EQ(proc.greedy(logits), 1u);
}

TEST_F(LogitsProcessorTest, SampleReturnsBoundedIndex) {
    std::vector<float> probs = {0.2f, 0.5f, 0.3f};
    size_t idx = proc.sample(probs, 12345ULL);
    EXPECT_LT(idx, probs.size());
}
