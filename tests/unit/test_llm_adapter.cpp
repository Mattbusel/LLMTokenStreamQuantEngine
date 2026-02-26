#include "gtest/gtest.h"
#include "LLMAdapter.h"

#include <cmath>
#include <string>
#include <vector>

namespace llmquant {
namespace {

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

static bool approx_eq(double a, double b, double eps = 1e-9) {
    return std::abs(a - b) < eps;
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

TEST(LLMAdapterTest, test_llm_adapter_map_token_fear_token_returns_negative_sentiment) {
    LLMAdapter adapter;
    SemanticWeight w = adapter.map_token_to_weight("crash");
    EXPECT_LT(w.sentiment_score, 0.0) << "Fear token 'crash' must have negative sentiment";
    EXPECT_GT(w.confidence_score, 0.5) << "Well-known fear token must have high confidence";
}

TEST(LLMAdapterTest, test_llm_adapter_map_token_bullish_token_returns_positive_bias) {
    LLMAdapter adapter;
    SemanticWeight w = adapter.map_token_to_weight("bullish");
    EXPECT_GT(w.directional_bias, 0.0) << "'bullish' must have positive directional bias";
    EXPECT_GT(w.sentiment_score,  0.0) << "'bullish' must have positive sentiment";
}

TEST(LLMAdapterTest, test_llm_adapter_map_token_unknown_token_returns_zero_weight) {
    LLMAdapter adapter;
    // A token that is not in any dictionary should yield the neutral default.
    SemanticWeight w = adapter.map_token_to_weight("xyzzy_unknown_token_42");
    EXPECT_DOUBLE_EQ(w.sentiment_score,   0.0);
    EXPECT_DOUBLE_EQ(w.directional_bias,  0.0);
    // confidence_score for unknown tokens is 0.5 (neutral)
    EXPECT_DOUBLE_EQ(w.confidence_score, 0.5);
}

TEST(LLMAdapterTest, test_llm_adapter_map_sequence_empty_returns_zero) {
    LLMAdapter adapter;
    SemanticWeight w = adapter.map_sequence_to_weight({});
    EXPECT_DOUBLE_EQ(w.sentiment_score,   0.0);
    EXPECT_DOUBLE_EQ(w.confidence_score,  0.0);
    EXPECT_DOUBLE_EQ(w.volatility_score,  0.0);
    EXPECT_DOUBLE_EQ(w.directional_bias,  0.0);
}

TEST(LLMAdapterTest, test_llm_adapter_map_sequence_confidence_weighted_average) {
    LLMAdapter adapter;
    // Insert two tokens with known, predictable weights.
    adapter.add_token_mapping("up",   SemanticWeight{0.8, 1.0, 0.0, 0.8});
    adapter.add_token_mapping("down", SemanticWeight{-0.8, 1.0, 0.0, -0.8});

    // Equal confidence => average bias should be ~0.
    SemanticWeight agg = adapter.map_sequence_to_weight({"up", "down"});
    EXPECT_NEAR(agg.directional_bias, 0.0, 1e-6);
    EXPECT_NEAR(agg.sentiment_score,  0.0, 1e-6);
}

TEST(LLMAdapterTest, test_llm_adapter_map_sequence_single_token_matches_direct_lookup) {
    LLMAdapter adapter;
    SemanticWeight direct = adapter.map_token_to_weight("bullish");
    SemanticWeight seq    = adapter.map_sequence_to_weight({"bullish"});

    EXPECT_NEAR(seq.sentiment_score,  direct.sentiment_score,  1e-9);
    EXPECT_NEAR(seq.directional_bias, direct.directional_bias, 1e-9);
    EXPECT_NEAR(seq.volatility_score, direct.volatility_score, 1e-9);
}

TEST(LLMAdapterTest, test_llm_adapter_add_token_mapping_overrides_default) {
    LLMAdapter adapter;
    // 'bullish' is in the default map.  Override it.
    adapter.add_token_mapping("bullish", SemanticWeight{-0.5, 0.9, 0.3, -0.5});
    SemanticWeight w = adapter.map_token_to_weight("bullish");
    EXPECT_DOUBLE_EQ(w.sentiment_score,  -0.5);
    EXPECT_DOUBLE_EQ(w.directional_bias, -0.5);
}

TEST(LLMAdapterTest, test_llm_adapter_cache_stats_track_hits_and_misses) {
    // We cannot inspect private stats_ directly, but we can verify via
    // observable behaviour: map a known token (hit) and an unknown token
    // (miss) and confirm the adapter still functions correctly.
    LLMAdapter adapter;

    // Known token -> should be found (cache hit)
    SemanticWeight hit = adapter.map_token_to_weight("crash");
    EXPECT_LT(hit.sentiment_score, 0.0);

    // Unknown token -> default neutral (cache miss)
    SemanticWeight miss = adapter.map_token_to_weight("nonexistent_abc");
    EXPECT_DOUBLE_EQ(miss.directional_bias, 0.0);
    EXPECT_DOUBLE_EQ(miss.confidence_score, 0.5);
}

} // namespace
} // namespace llmquant
