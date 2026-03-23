#include <gtest/gtest.h>
#include "BeamSearchDecoderV2.h"

#include <algorithm>
#include <cmath>

using namespace llmquant;

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// A simple next_logits_fn that always returns uniform logits.
static std::function<std::vector<float>(const std::vector<int>&)>
uniform_next_fn(int vocab_size)
{
    return [vocab_size](const std::vector<int>&) -> std::vector<float> {
        return std::vector<float>(vocab_size, 0.0f);
    };
}

/// Always returns logits favouring token `tok`.
static std::function<std::vector<float>(const std::vector<int>&)>
biased_next_fn(int vocab_size, int tok)
{
    return [vocab_size, tok](const std::vector<int>&) -> std::vector<float> {
        std::vector<float> l(vocab_size, 0.0f);
        l[tok] = 10.0f;
        return l;
    };
}

// ---------------------------------------------------------------------------
// Construction
// ---------------------------------------------------------------------------

TEST(BeamSearchDecoderV2, ConstructValid)
{
    EXPECT_NO_THROW(BeamSearchDecoderV2(4, 128, 2));
}

TEST(BeamSearchDecoderV2, ConstructInvalidBeamWidth)
{
    EXPECT_THROW(BeamSearchDecoderV2(0, 128, 2), std::invalid_argument);
}

// ---------------------------------------------------------------------------
// Softmax
// ---------------------------------------------------------------------------

TEST(BeamSearchDecoderV2, SoftmaxSumsToOne)
{
    BeamSearchDecoderV2 dec(2, 10, 1);
    std::vector<float> logits = {1.0f, 2.0f, 3.0f, 4.0f};
    auto probs = dec.softmax(logits);
    float sum = 0.0f;
    for (float p : probs) sum += p;
    EXPECT_NEAR(sum, 1.0f, 1e-5f);
}

TEST(BeamSearchDecoderV2, SoftmaxMonotone)
{
    BeamSearchDecoderV2 dec(2, 10, 1);
    std::vector<float> logits = {1.0f, 2.0f, 3.0f};
    auto probs = dec.softmax(logits);
    EXPECT_LT(probs[0], probs[1]);
    EXPECT_LT(probs[1], probs[2]);
}

// ---------------------------------------------------------------------------
// Length-penalised score
// ---------------------------------------------------------------------------

TEST(BeamSearchDecoderV2, LengthPenaltyReducesLongBeams)
{
    BeamSearchDecoderV2 dec(4, 128, 2, 1.0f);
    Beam short_beam, long_beam;
    short_beam.tokens = {1, 2};
    short_beam.log_prob = -2.0f;
    long_beam.tokens = {1, 2, 3, 4};
    long_beam.log_prob = -2.0f;

    float s_short = dec.length_penalized_score(short_beam);
    float s_long = dec.length_penalized_score(long_beam);
    // log_prob / len: -2/2 = -1 vs -2/4 = -0.5; shorter beam has lower score
    EXPECT_LT(s_short, s_long);
}

// ---------------------------------------------------------------------------
// best_sequence
// ---------------------------------------------------------------------------

TEST(BeamSearchDecoderV2, BestSequencePicksHighestScore)
{
    BeamSearchDecoderV2 dec(2, 10, 1);
    Beam a, b;
    a.tokens = {1, 2};
    a.log_prob = -1.0f;
    a.score = -0.5f;
    b.tokens = {3, 4};
    b.log_prob = -0.5f;
    b.score = -0.25f;
    auto seq = dec.best_sequence({a, b});
    EXPECT_EQ(seq[0], 3);
    EXPECT_EQ(seq[1], 4);
}

// ---------------------------------------------------------------------------
// Full search
// ---------------------------------------------------------------------------

TEST(BeamSearchDecoderV2, SearchProducesNonEmptyOutput)
{
    BeamSearchDecoderV2 dec(2, 5, 3);
    std::vector<float> init_logits = {0.5f, 1.0f, 0.2f, 0.1f, 0.8f};
    auto beams = dec.search(init_logits, uniform_next_fn(5));
    EXPECT_FALSE(beams.empty());
}

TEST(BeamSearchDecoderV2, SearchEndsAtEos)
{
    // EOS = token 2, always biased toward it
    int vocab = 5;
    BeamSearchDecoderV2 dec(2, 20, 2);
    std::vector<float> init_logits(vocab, 0.0f);
    init_logits[2] = 10.0f;
    auto beams = dec.search(init_logits, biased_next_fn(vocab, 2));
    // All beams should be finished
    for (const auto& b : beams)
        EXPECT_TRUE(b.finished || b.tokens.size() >= 20u);
}

TEST(BeamSearchDecoderV2, SearchBeamWidthRespected)
{
    BeamSearchDecoderV2 dec(3, 10, 99);
    std::vector<float> init_logits(10, 0.0f);
    auto beams = dec.search(init_logits, uniform_next_fn(10));
    EXPECT_LE(static_cast<int>(beams.size()), 3);
}

TEST(BeamSearchDecoderV2, SearchOutputSortedByScore)
{
    BeamSearchDecoderV2 dec(4, 8, 99);
    std::vector<float> init_logits(8, 0.0f);
    auto beams = dec.search(init_logits, uniform_next_fn(8));
    for (std::size_t i = 1; i < beams.size(); ++i)
        EXPECT_GE(beams[i - 1].score, beams[i].score);
}

TEST(BeamSearchDecoderV2, BestSequenceFromSearch)
{
    BeamSearchDecoderV2 dec(2, 5, 99);
    std::vector<float> init_logits = {0.0f, 10.0f, 0.0f, 0.0f};
    auto beams = dec.search(init_logits, biased_next_fn(4, 1));
    auto seq = dec.best_sequence(beams);
    EXPECT_FALSE(seq.empty());
    // First token should be token 1 (highest logit)
    EXPECT_EQ(seq[0], 1);
}

TEST(BeamSearchDecoderV2, EmptyLogitsReturnsEmpty)
{
    BeamSearchDecoderV2 dec(2, 10, 1);
    auto beams = dec.search({}, uniform_next_fn(0));
    EXPECT_TRUE(beams.empty());
}
