#include <gtest/gtest.h>
#include "TokenSampler.h"

using namespace llmquant;

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

static std::vector<float> uniform_logits(int n)
{
    return std::vector<float>(n, 0.0f);
}

// ---------------------------------------------------------------------------
// Construction
// ---------------------------------------------------------------------------

TEST(TokenSampler, ConstructDefault)
{
    SamplingConfig cfg;
    TokenSampler sampler(cfg);
    // Should not throw
    SUCCEED();
}

// ---------------------------------------------------------------------------
// Temperature
// ---------------------------------------------------------------------------

TEST(TokenSampler, TemperatureScalesLogits)
{
    SamplingConfig cfg;
    cfg.temperature = 2.0f;
    TokenSampler sampler(cfg);
    std::vector<float> logits = {2.0f, 4.0f, 6.0f};
    auto out = sampler.apply_temperature(logits);
    ASSERT_EQ(out.size(), logits.size());
    EXPECT_NEAR(out[0], 1.0f, 1e-5f);
    EXPECT_NEAR(out[1], 2.0f, 1e-5f);
    EXPECT_NEAR(out[2], 3.0f, 1e-5f);
}

TEST(TokenSampler, TemperatureEmptyInput)
{
    SamplingConfig cfg;
    TokenSampler sampler(cfg);
    auto out = sampler.apply_temperature({});
    EXPECT_TRUE(out.empty());
}

// ---------------------------------------------------------------------------
// Top-k
// ---------------------------------------------------------------------------

TEST(TokenSampler, TopKKeepsTopTokens)
{
    SamplingConfig cfg;
    TokenSampler sampler(cfg);
    std::vector<float> probs = {0.1f, 0.5f, 0.3f, 0.05f, 0.05f};
    auto out = sampler.apply_top_k(probs, 2);
    ASSERT_EQ(out.size(), probs.size());
    // Indices 1 and 2 should be non-zero
    EXPECT_GT(out[1], 0.0f);
    EXPECT_GT(out[2], 0.0f);
    // Others should be zero
    EXPECT_FLOAT_EQ(out[0], 0.0f);
    EXPECT_FLOAT_EQ(out[3], 0.0f);
    EXPECT_FLOAT_EQ(out[4], 0.0f);
}

TEST(TokenSampler, TopKNormalisesToOne)
{
    SamplingConfig cfg;
    TokenSampler sampler(cfg);
    std::vector<float> probs = {0.1f, 0.4f, 0.3f, 0.2f};
    auto out = sampler.apply_top_k(probs, 2);
    float sum = 0.0f;
    for (float v : out) sum += v;
    EXPECT_NEAR(sum, 1.0f, 1e-5f);
}

// ---------------------------------------------------------------------------
// Top-p
// ---------------------------------------------------------------------------

TEST(TokenSampler, TopPNormalisesToOne)
{
    SamplingConfig cfg;
    TokenSampler sampler(cfg);
    std::vector<float> probs = {0.05f, 0.45f, 0.35f, 0.15f};
    auto out = sampler.apply_top_p(probs, 0.9f);
    float sum = 0.0f;
    for (float v : out) sum += v;
    EXPECT_NEAR(sum, 1.0f, 1e-5f);
}

TEST(TokenSampler, TopPWithP1KeepsAll)
{
    SamplingConfig cfg;
    TokenSampler sampler(cfg);
    std::vector<float> probs = {0.25f, 0.25f, 0.25f, 0.25f};
    auto out = sampler.apply_top_p(probs, 1.0f);
    for (float v : out) EXPECT_GT(v, 0.0f);
}

// ---------------------------------------------------------------------------
// Repetition penalty
// ---------------------------------------------------------------------------

TEST(TokenSampler, RepetitionPenaltyReducesPositiveLogit)
{
    SamplingConfig cfg;
    cfg.repetition_penalty = 2.0f;
    TokenSampler sampler(cfg);
    std::vector<float> logits = {1.0f, 2.0f, 3.0f};
    auto out = sampler.apply_repetition_penalty(logits, {1});
    EXPECT_NEAR(out[1], 1.0f, 1e-5f);  // 2.0 / 2.0
    EXPECT_NEAR(out[0], 1.0f, 1e-5f);  // unchanged
}

// ---------------------------------------------------------------------------
// Greedy
// ---------------------------------------------------------------------------

TEST(TokenSampler, GreedyReturnsArgmax)
{
    SamplingConfig cfg;
    TokenSampler sampler(cfg);
    std::vector<float> logits = {0.1f, 0.9f, 0.3f};
    EXPECT_EQ(sampler.greedy(logits), 1);
}

// ---------------------------------------------------------------------------
// Sampling
// ---------------------------------------------------------------------------

TEST(TokenSampler, SampleReturnsValidIndex)
{
    SamplingConfig cfg;
    cfg.seed = 777;
    TokenSampler sampler(cfg);
    std::vector<float> probs = {0.1f, 0.5f, 0.4f};
    for (int i = 0; i < 100; ++i) {
        int tok = sampler.sample(probs);
        EXPECT_GE(tok, 0);
        EXPECT_LT(tok, static_cast<int>(probs.size()));
    }
}

TEST(TokenSampler, SampleFromLogitsReturnsValidIndex)
{
    SamplingConfig cfg;
    cfg.seed = 42;
    cfg.temperature = 1.0f;
    cfg.top_k = 10;
    cfg.top_p = 0.95f;
    TokenSampler sampler(cfg);
    std::vector<float> logits(20, 0.0f);
    logits[5] = 3.0f;
    for (int i = 0; i < 50; ++i) {
        int tok = sampler.sample_from_logits(logits, {});
        EXPECT_GE(tok, 0);
        EXPECT_LT(tok, 20);
    }
}

TEST(TokenSampler, SetSeedProducesDeterministicOutput)
{
    SamplingConfig cfg;
    cfg.temperature = 1.0f;
    cfg.top_k = 5;
    cfg.top_p = 1.0f;
    TokenSampler s1(cfg), s2(cfg);
    s1.set_seed(999);
    s2.set_seed(999);
    std::vector<float> logits = {1.0f, 2.0f, 1.5f, 0.5f, 3.0f};
    EXPECT_EQ(s1.sample_from_logits(logits, {}), s2.sample_from_logits(logits, {}));
}
