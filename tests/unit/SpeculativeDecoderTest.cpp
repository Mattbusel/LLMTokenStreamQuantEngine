#include <gtest/gtest.h>
#include "SpeculativeDecoder.h"

#include <algorithm>
#include <cmath>
#include <functional>
#include <numeric>
#include <vector>

using namespace llmquant;

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

static constexpr int kVocabSize = 32;

static SpeculativeDecoder::ModelFn make_uniform_fn() {
    return [](const std::vector<int>&) -> std::vector<float> {
        return std::vector<float>(kVocabSize, 0.0f);
    };
}

static SpeculativeDecoder::ModelFn make_biased_fn(int preferred_token) {
    return [preferred_token](const std::vector<int>&) -> std::vector<float> {
        std::vector<float> logits(kVocabSize, 0.0f);
        logits[preferred_token] = 10.0f;
        return logits;
    };
}

// ---------------------------------------------------------------------------
// Construction
// ---------------------------------------------------------------------------

TEST(SpeculativeDecoder, ConstructValid) {
    SpeculativeConfig cfg;
    cfg.draft_steps = 4;
    EXPECT_NO_THROW(SpeculativeDecoder(make_uniform_fn(), make_uniform_fn(), cfg));
}

TEST(SpeculativeDecoder, ConstructInvalidDraftSteps) {
    SpeculativeConfig cfg;
    cfg.draft_steps = 0;
    EXPECT_THROW(SpeculativeDecoder(make_uniform_fn(), make_uniform_fn(), cfg),
                 std::invalid_argument);
}

TEST(SpeculativeDecoder, ConstructNullDraftModel) {
    SpeculativeConfig cfg;
    EXPECT_THROW(SpeculativeDecoder(nullptr, make_uniform_fn(), cfg),
                 std::invalid_argument);
}

// ---------------------------------------------------------------------------
// decode_step — draft tokens produced
// ---------------------------------------------------------------------------

TEST(SpeculativeDecoder, DecodeStepProducesTokens) {
    SpeculativeConfig cfg;
    cfg.draft_steps = 4;
    SpeculativeDecoder dec(make_biased_fn(3), make_biased_fn(3), cfg);
    uint64_t rng = 42;
    auto tokens = dec.decode_step({1, 2, 3}, rng);
    EXPECT_FALSE(tokens.empty());
    for (int t : tokens) {
        EXPECT_GE(t, 0);
        EXPECT_LT(t, kVocabSize);
    }
}

TEST(SpeculativeDecoder, DecodeStepMaxTokensBounded) {
    SpeculativeConfig cfg;
    cfg.draft_steps = 4;
    SpeculativeDecoder dec(make_uniform_fn(), make_uniform_fn(), cfg);
    uint64_t rng = 7;
    // draft_steps=4 drafts + 1 correction = max 5 tokens
    auto tokens = dec.decode_step({}, rng);
    EXPECT_LE(static_cast<int>(tokens.size()), cfg.draft_steps + 1);
}

// ---------------------------------------------------------------------------
// Acceptance rate
// ---------------------------------------------------------------------------

TEST(SpeculativeDecoder, AcceptanceRateZeroInitially) {
    SpeculativeConfig cfg;
    SpeculativeDecoder dec(make_uniform_fn(), make_uniform_fn(), cfg);
    auto stats = dec.get_stats();
    EXPECT_EQ(stats.total_draft_tokens, 0u);
    EXPECT_EQ(stats.accepted_tokens, 0u);
}

TEST(SpeculativeDecoder, AcceptanceRateAfterSteps) {
    SpeculativeConfig cfg;
    cfg.draft_steps = 4;
    // Identical models → high acceptance
    SpeculativeDecoder dec(make_biased_fn(5), make_biased_fn(5), cfg);
    uint64_t rng = 12345;
    for (int i = 0; i < 10; ++i) {
        dec.decode_step({0, 1, 2}, rng);
    }
    auto stats = dec.get_stats();
    EXPECT_GT(stats.total_draft_tokens, 0u);
    float rate = stats.acceptance_rate;
    EXPECT_GE(rate, 0.0f);
    EXPECT_LE(rate, 1.0f);
}

TEST(SpeculativeDecoder, AcceptanceRateLowWhenModelsDiverge) {
    SpeculativeConfig cfg;
    cfg.draft_steps = 4;
    // Draft prefers token 0, target prefers token 31 → low acceptance
    SpeculativeDecoder dec(make_biased_fn(0), make_biased_fn(31), cfg);
    uint64_t rng = 99999;
    for (int i = 0; i < 20; ++i) {
        dec.decode_step({0}, rng);
    }
    auto stats = dec.get_stats();
    EXPECT_GT(stats.total_draft_tokens, 0u);
    // Rate should be < 1.0 when models diverge significantly
    EXPECT_LT(stats.acceptance_rate, 1.0f);
}

// ---------------------------------------------------------------------------
// Determinism
// ---------------------------------------------------------------------------

TEST(SpeculativeDecoder, SameRngProducesSameTokens) {
    SpeculativeConfig cfg;
    cfg.draft_steps = 3;
    SpeculativeDecoder dec1(make_biased_fn(7), make_biased_fn(7), cfg);
    SpeculativeDecoder dec2(make_biased_fn(7), make_biased_fn(7), cfg);
    uint64_t rng1 = 42;
    uint64_t rng2 = 42;
    auto t1 = dec1.decode_step({1, 2}, rng1);
    auto t2 = dec2.decode_step({1, 2}, rng2);
    EXPECT_EQ(t1, t2);
}
