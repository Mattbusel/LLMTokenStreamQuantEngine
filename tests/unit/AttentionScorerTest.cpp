#include "AttentionScorer.h"

#include <gtest/gtest.h>
#include <cmath>
#include <numeric>
#include <vector>

using namespace llmquant;

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

static std::vector<float> make_key_flat(int seq_len, int head_dim, float fill = 1.0f) {
    return std::vector<float>(static_cast<size_t>(seq_len * head_dim), fill);
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

TEST(AttentionScorerTest, ConstructsWithoutError) {
    EXPECT_NO_THROW(AttentionScorer scorer(4, 8));
}

TEST(AttentionScorerTest, ScaledDotProductLengthMatchesSeqLen) {
    AttentionScorer scorer(1, 4);
    std::vector<float> q(4, 1.0f);
    auto k = make_key_flat(5, 4, 1.0f);
    auto scores = scorer.scaled_dot_product(q, k);
    EXPECT_EQ(static_cast<int>(scores.size()), 5);
}

TEST(AttentionScorerTest, ScaledDotProductScalesCorrectly) {
    // q = [1,0,...], k = [1,0,...] row 0 → dot = 1, scaled = 1/sqrt(d_k)
    AttentionScorer scorer(1, 4);
    std::vector<float> q = {1.0f, 0.0f, 0.0f, 0.0f};
    auto k = make_key_flat(1, 4, 0.0f);
    k[0] = 1.0f;
    auto scores = scorer.scaled_dot_product(q, k);
    ASSERT_EQ(scores.size(), 1u);
    EXPECT_NEAR(scores[0], 1.0f / std::sqrt(4.0f), 1e-5f);
}

TEST(AttentionScorerTest, SoftmaxSumsToOne) {
    AttentionScorer scorer(2, 4);
    std::vector<float> raw = {1.0f, 2.0f, 3.0f, 0.5f};
    auto weights = scorer.softmax(raw);
    float total = 0.0f;
    for (float w : weights) total += w;
    EXPECT_NEAR(total, 1.0f, 1e-5f);
}

TEST(AttentionScorerTest, SoftmaxAllNonNegative) {
    AttentionScorer scorer(2, 4);
    auto weights = scorer.softmax({-10.0f, -5.0f, 0.0f, 5.0f});
    for (float w : weights) {
        EXPECT_GE(w, 0.0f);
    }
}

TEST(AttentionScorerTest, SoftmaxNumericalStability) {
    AttentionScorer scorer(1, 2);
    // Very large values should not produce NaN or Inf.
    auto weights = scorer.softmax({1e30f, 1e30f, 1e30f});
    for (float w : weights) {
        EXPECT_TRUE(std::isfinite(w));
    }
}

TEST(AttentionScorerTest, CausalMaskKeepsCurrentAndPast) {
    AttentionScorer scorer(1, 4);
    // For seq_len=4, query at position 3: positions 0..3 are visible.
    std::vector<float> scores = {1.0f, 2.0f, 3.0f, 4.0f};
    auto masked = scorer.apply_causal_mask(scores, 4);
    // No positions are masked (query is the last position).
    EXPECT_FLOAT_EQ(masked[0], 1.0f);
    EXPECT_FLOAT_EQ(masked[3], 4.0f);
}

TEST(AttentionScorerTest, CausalMaskBlocksFuturePositions) {
    AttentionScorer scorer(1, 4);
    // Score vector of length 6 but seq_len=3 → positions 3,4,5 are future.
    std::vector<float> scores(6, 1.0f);
    auto masked = scorer.apply_causal_mask(scores, 3);
    EXPECT_TRUE(std::isinf(masked[3]) && masked[3] < 0.0f);
    EXPECT_TRUE(std::isinf(masked[5]) && masked[5] < 0.0f);
}

TEST(AttentionScorerTest, ApplyAttentionBiasAddsElementWise) {
    AttentionScorer scorer(1, 4);
    std::vector<float> scores = {1.0f, 2.0f, 3.0f};
    std::vector<float> bias   = {0.5f, -0.5f, 1.0f};
    auto result = scorer.apply_attention_bias(scores, bias);
    EXPECT_NEAR(result[0], 1.5f, 1e-5f);
    EXPECT_NEAR(result[1], 1.5f, 1e-5f);
    EXPECT_NEAR(result[2], 4.0f, 1e-5f);
}

TEST(AttentionScorerTest, AttentionEntropyUniformIsMaximal) {
    AttentionScorer scorer(2, 4);
    // Uniform distribution has maximum entropy.
    std::vector<float> uniform = {0.25f, 0.25f, 0.25f, 0.25f};
    std::vector<float> peaked  = {0.97f, 0.01f, 0.01f, 0.01f};
    float h_uniform = scorer.attention_entropy(uniform);
    float h_peaked  = scorer.attention_entropy(peaked);
    EXPECT_GT(h_uniform, h_peaked);
}

TEST(AttentionScorerTest, AttentionEntropyNonNegative) {
    AttentionScorer scorer(1, 4);
    auto w = scorer.softmax({1.0f, 2.0f, 0.5f});
    EXPECT_GE(scorer.attention_entropy(w), 0.0f);
}

TEST(AttentionScorerTest, IsAttendingUniformlyDetectsUniform) {
    AttentionScorer scorer(2, 4);
    std::vector<float> uniform(8, 0.125f);
    EXPECT_TRUE(scorer.is_attending_uniformly(uniform, 0.1f));
}

TEST(AttentionScorerTest, IsAttendingUniformlyDetectsPeaked) {
    AttentionScorer scorer(2, 4);
    std::vector<float> peaked = {0.9f, 0.05f, 0.03f, 0.02f};
    EXPECT_FALSE(scorer.is_attending_uniformly(peaked, 0.1f));
}

TEST(AttentionScorerTest, MultiHeadAttentionReturnsCorrectShape) {
    AttentionScorer scorer(3, 8);
    std::vector<std::vector<float>> queries = {
        std::vector<float>(8, 1.0f),
        std::vector<float>(8, 0.5f),
        std::vector<float>(8, 0.0f),
    };
    // Each "key" for a head: flat seq_len=4 keys of head_dim=8.
    std::vector<std::vector<float>> keys = {
        std::vector<float>(4 * 8, 1.0f),
        std::vector<float>(4 * 8, 0.5f),
        std::vector<float>(4 * 8, 0.0f),
    };
    auto result = scorer.multi_head_attention(queries, keys);
    ASSERT_EQ(result.size(), 3u);
    for (const auto& head_scores : result) {
        EXPECT_EQ(static_cast<int>(head_scores.size()), 4);
    }
}

TEST(AttentionScorerTest, EmptyInputsHandledGracefully) {
    AttentionScorer scorer(1, 4);
    EXPECT_TRUE(scorer.scaled_dot_product({}, {}).empty());
    EXPECT_TRUE(scorer.softmax({}).empty());
    EXPECT_FLOAT_EQ(scorer.attention_entropy({}), 0.0f);
    EXPECT_TRUE(scorer.is_attending_uniformly({}, 0.1f));
}
