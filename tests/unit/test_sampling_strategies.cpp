#include <gtest/gtest.h>

#ifdef LLMQUANT_SAMPLING_STRATEGIES_ENABLED

#include "SamplingStrategies.h"

#include <cmath>
#include <numeric>
#include <vector>

using namespace llmquant;

namespace {

// ============================================================
// Softmax
// ============================================================

TEST(SamplingStrategies, SoftmaxSumsToOne) {
    std::vector<float> logits = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f};
    auto probs = SamplingStrategies::softmax(logits, 1.0f);

    ASSERT_EQ(probs.size(), logits.size());
    float total = std::accumulate(probs.begin(), probs.end(), 0.0f);
    EXPECT_NEAR(total, 1.0f, 1e-5f);
}

TEST(SamplingStrategies, SoftmaxSumsToOneWithTemperature) {
    std::vector<float> logits = {0.5f, -1.0f, 2.3f, 0.1f};
    auto probs = SamplingStrategies::softmax(logits, 0.5f);

    ASSERT_EQ(probs.size(), logits.size());
    float total = std::accumulate(probs.begin(), probs.end(), 0.0f);
    EXPECT_NEAR(total, 1.0f, 1e-5f);

    // Lower temperature → sharper distribution (higher max probability).
    auto probs_hot = SamplingStrategies::softmax(logits, 2.0f);
    float max_cold = *std::max_element(probs.begin(), probs.end());
    float max_hot  = *std::max_element(probs_hot.begin(), probs_hot.end());
    EXPECT_GT(max_cold, max_hot);
}

TEST(SamplingStrategies, SoftmaxEmptyReturnsEmpty) {
    auto probs = SamplingStrategies::softmax({}, 1.0f);
    EXPECT_TRUE(probs.empty());
}

// ============================================================
// Greedy
// ============================================================

TEST(SamplingStrategies, GreedyReturnsArgmax) {
    std::vector<float> logits = {0.1f, 3.5f, 1.2f, 0.8f};
    int result = SamplingStrategies::greedy(logits);
    EXPECT_EQ(result, 1); // index 1 has highest logit
}

TEST(SamplingStrategies, GreedyEmptyReturnsMinusOne) {
    EXPECT_EQ(SamplingStrategies::greedy({}), -1);
}

TEST(SamplingStrategies, GreedyNegativeLogits) {
    std::vector<float> logits = {-5.0f, -2.0f, -3.0f};
    int result = SamplingStrategies::greedy(logits);
    EXPECT_EQ(result, 1);
}

// ============================================================
// Top-k sampling
// ============================================================

TEST(SamplingStrategies, TopKReducesCandidates) {
    // With a large vocab and k=2, the sampled token must be one of the top-2.
    std::vector<float> logits = {0.1f, 0.2f, 10.0f, 9.0f, 0.3f}; // top-2 at indices 2 and 3
    uint64_t rng_state = 42;

    // Sample many times and verify all results are in {2, 3}.
    for (int i = 0; i < 100; ++i) {
        int tok = SamplingStrategies::top_k_sample(logits, 2, rng_state);
        EXPECT_TRUE(tok == 2 || tok == 3)
            << "top_k=2 sampled token " << tok << " which is not in top-2";
    }
}

TEST(SamplingStrategies, TopKWithKEqualsOne) {
    std::vector<float> logits = {1.0f, 5.0f, 2.0f};
    uint64_t rng_state = 99;
    // k=1 → always greedy.
    for (int i = 0; i < 20; ++i) {
        int tok = SamplingStrategies::top_k_sample(logits, 1, rng_state);
        EXPECT_EQ(tok, 1);
    }
}

// ============================================================
// Top-p (nucleus) sampling
// ============================================================

TEST(SamplingStrategies, TopPSamplesValidToken) {
    std::vector<float> logits = {1.0f, 2.0f, 3.0f};
    uint64_t rng_state = 7;
    for (int i = 0; i < 50; ++i) {
        int tok = SamplingStrategies::top_p_sample(logits, 0.95f, rng_state);
        EXPECT_GE(tok, 0);
        EXPECT_LT(tok, static_cast<int>(logits.size()));
    }
}

// ============================================================
// Repetition penalty
// ============================================================

TEST(SamplingStrategies, RepetitionPenaltyReducesRepeatedTokenLogit) {
    std::vector<float> logits = {1.0f, 2.0f, 3.0f, 4.0f};
    std::vector<int> recent = {2, 3}; // tokens 2 and 3 are repeated
    float penalty = 1.5f;

    float original_2 = logits[2];
    float original_3 = logits[3];

    SamplingStrategies::apply_repetition_penalty(logits, recent, penalty);

    // Positive logits should be divided by the penalty.
    EXPECT_NEAR(logits[2], original_2 / penalty, 1e-5f);
    EXPECT_NEAR(logits[3], original_3 / penalty, 1e-5f);

    // Unaffected tokens should be unchanged.
    EXPECT_NEAR(logits[0], 1.0f, 1e-5f);
    EXPECT_NEAR(logits[1], 2.0f, 1e-5f);
}

TEST(SamplingStrategies, RepetitionPenaltyNoPenaltyAtOne) {
    std::vector<float> logits = {1.0f, 2.0f, 3.0f};
    std::vector<int> recent = {0, 1, 2};
    std::vector<float> original = logits;

    SamplingStrategies::apply_repetition_penalty(logits, recent, 1.0f);

    for (std::size_t i = 0; i < logits.size(); ++i) {
        EXPECT_NEAR(logits[i], original[i], 1e-6f);
    }
}

TEST(SamplingStrategies, RepetitionPenaltyNegativeLogit) {
    std::vector<float> logits = {-3.0f, 1.0f};
    std::vector<int> recent = {0};
    float penalty = 2.0f;

    float original = logits[0];
    SamplingStrategies::apply_repetition_penalty(logits, recent, penalty);

    // Negative logit should be multiplied by penalty (more negative).
    EXPECT_NEAR(logits[0], original * penalty, 1e-5f);
}

// ============================================================
// LCG RNG
// ============================================================

TEST(SamplingStrategies, LcgFloatInRange) {
    uint64_t state = 12345;
    for (int i = 0; i < 10000; ++i) {
        float val = SamplingStrategies::lcg_float(state);
        EXPECT_GE(val, 0.0f);
        EXPECT_LT(val, 1.0f + 1e-6f); // allow tiny float imprecision
    }
}

TEST(SamplingStrategies, LcgFloatDeterministic) {
    uint64_t s1 = 777, s2 = 777;
    float v1 = SamplingStrategies::lcg_float(s1);
    float v2 = SamplingStrategies::lcg_float(s2);
    EXPECT_FLOAT_EQ(v1, v2);
}

// ============================================================
// Unified dispatcher
// ============================================================

TEST(SamplingStrategies, SampleGreedyDispatch) {
    std::vector<float> logits = {0.1f, 5.0f, 0.5f};
    SamplingConfig cfg{1.0f, 0.95f, 0, 1.0f, false};
    uint64_t rng = 0;
    int tok = SamplingStrategies::sample(logits, cfg, rng);
    EXPECT_EQ(tok, 1); // greedy → argmax
}

TEST(SamplingStrategies, SampleTopKDispatch) {
    std::vector<float> logits = {0.1f, 0.2f, 10.0f, 9.0f, 0.3f};
    SamplingConfig cfg{1.0f, 1.0f, 2, 1.0f, true};
    uint64_t rng = 13;
    for (int i = 0; i < 50; ++i) {
        int tok = SamplingStrategies::sample(logits, cfg, rng);
        EXPECT_TRUE(tok == 2 || tok == 3);
    }
}

} // namespace

#else  // LLMQUANT_SAMPLING_STRATEGIES_ENABLED

// Stub test so the binary always contains at least one test case.
TEST(SamplingStrategiesDisabled, FeatureFlagOff) {
    SUCCEED() << "SamplingStrategies feature flag is OFF — tests skipped.";
}

#endif // LLMQUANT_SAMPLING_STRATEGIES_ENABLED
