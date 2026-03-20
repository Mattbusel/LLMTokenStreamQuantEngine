#include "gtest/gtest.h"
#include "LLMAdapter.h"

#include <cmath>
#include <cstdio>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace llmquant {
namespace {

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

[[maybe_unused]] static bool approx_eq(double a, double b, double eps = 1e-9) {
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

TEST(LLMAdapterTest, test_llm_adapter_simd_empty_returns_zero) {
    LLMAdapter adapter;
    auto w = adapter.map_sequence_simd({});
    EXPECT_DOUBLE_EQ(w.sentiment_score,  0.0);
    EXPECT_DOUBLE_EQ(w.confidence_score, 0.0);
    EXPECT_DOUBLE_EQ(w.volatility_score, 0.0);
    EXPECT_DOUBLE_EQ(w.directional_bias, 0.0);
}

TEST(LLMAdapterTest, test_llm_adapter_simd_matches_scalar_for_single_token) {
    LLMAdapter adapter;
    auto scalar = adapter.map_sequence_to_weight({"bullish"});
    auto simd   = adapter.map_sequence_simd({"bullish"});
    EXPECT_NEAR(scalar.sentiment_score,  simd.sentiment_score,  1e-9);
    EXPECT_NEAR(scalar.directional_bias, simd.directional_bias, 1e-9);
    EXPECT_NEAR(scalar.volatility_score, simd.volatility_score, 1e-9);
    EXPECT_NEAR(scalar.confidence_score, simd.confidence_score, 1e-9);
}

TEST(LLMAdapterTest, test_llm_adapter_simd_matches_scalar_for_four_tokens) {
    LLMAdapter adapter;
    std::vector<std::string> tokens{"bullish", "crash", "volatile", "rally"};
    auto scalar = adapter.map_sequence_to_weight(tokens);
    auto simd   = adapter.map_sequence_simd(tokens);
    EXPECT_NEAR(scalar.sentiment_score,  simd.sentiment_score,  1e-9);
    EXPECT_NEAR(scalar.directional_bias, simd.directional_bias, 1e-9);
    EXPECT_NEAR(scalar.volatility_score, simd.volatility_score, 1e-9);
    EXPECT_NEAR(scalar.confidence_score, simd.confidence_score, 1e-9);
}

// Improvement #15: empty-sequence test verifying stats and neutral output.
TEST(LLMAdapterTest, EmptySequence) {
    LLMAdapter adapter;
    auto stats_before = adapter.get_stats();

    SemanticWeight w = adapter.map_sequence_to_weight({});

    // All output fields must be zero / neutral for an empty input.
    EXPECT_DOUBLE_EQ(w.sentiment_score,   0.0);
    EXPECT_DOUBLE_EQ(w.confidence_score,  0.0);
    EXPECT_DOUBLE_EQ(w.volatility_score,  0.0);
    EXPECT_DOUBLE_EQ(w.directional_bias,  0.0);

    // map_sequence_to_weight({}) returns early before calling map_token_to_weight,
    // so tokens_processed must not have changed.
    auto stats_after = adapter.get_stats();
    EXPECT_EQ(stats_after.tokens_processed, stats_before.tokens_processed)
        << "Empty sequence must not increment tokens_processed";
}

TEST(LLMAdapterTest, test_llm_adapter_simd_odd_count_matches_scalar) {
    LLMAdapter adapter;
    std::vector<std::string> tokens{"crash", "panic", "bullish", "rally", "volatile"};
    auto scalar = adapter.map_sequence_to_weight(tokens);
    auto simd   = adapter.map_sequence_simd(tokens);
    EXPECT_NEAR(scalar.sentiment_score,  simd.sentiment_score,  1e-9);
    EXPECT_NEAR(scalar.directional_bias, simd.directional_bias, 1e-9);
}

// ---------------------------------------------------------------------------
// Tests for financial tokens added in the expanded dictionary
// ---------------------------------------------------------------------------

TEST(LLMAdapterTest, test_llm_adapter_calls_positive_puts_negative_directional_bias) {
    LLMAdapter adapter;
    SemanticWeight calls = adapter.map_token_to_weight("calls");
    SemanticWeight puts  = adapter.map_token_to_weight("puts");
    EXPECT_GT(calls.directional_bias, 0.0) << "'calls' should have positive directional bias";
    EXPECT_LT(puts.directional_bias,  0.0) << "'puts' should have negative directional bias";
    // Symmetry: magnitude should be similar
    EXPECT_NEAR(std::abs(calls.directional_bias), std::abs(puts.directional_bias), 0.2);
}

TEST(LLMAdapterTest, test_llm_adapter_overbought_oversold_opposite_bias) {
    LLMAdapter adapter;
    SemanticWeight ob  = adapter.map_token_to_weight("overbought");
    SemanticWeight os_ = adapter.map_token_to_weight("oversold");
    EXPECT_LT(ob.directional_bias,  0.0) << "'overbought' should signal bearish (reversal risk)";
    EXPECT_GT(os_.directional_bias, 0.0) << "'oversold' should signal bullish (bounce potential)";
}

TEST(LLMAdapterTest, test_llm_adapter_capitulation_strong_bearish) {
    LLMAdapter adapter;
    SemanticWeight w = adapter.map_token_to_weight("capitulation");
    EXPECT_LT(w.sentiment_score,  -0.5) << "'capitulation' must be strongly bearish";
    EXPECT_GT(w.volatility_score,  0.7) << "'capitulation' must signal high volatility";
}

TEST(LLMAdapterTest, test_llm_adapter_accumulation_bullish_low_vol) {
    LLMAdapter adapter;
    SemanticWeight w = adapter.map_token_to_weight("accumulation");
    EXPECT_GT(w.directional_bias, 0.0) << "'accumulation' should be bullish";
    EXPECT_LT(w.volatility_score, 0.6) << "'accumulation' implies controlled, low-vol buying";
}

TEST(LLMAdapterTest, test_llm_adapter_recession_strongly_bearish) {
    LLMAdapter adapter;
    SemanticWeight w = adapter.map_token_to_weight("recession");
    EXPECT_LT(w.sentiment_score,  -0.5) << "'recession' must be strongly bearish";
    EXPECT_LT(w.directional_bias, -0.5) << "'recession' must have strong negative bias";
}

TEST(LLMAdapterTest, test_llm_adapter_case_insensitive_lookup) {
    LLMAdapter adapter;
    // All three forms should resolve to the same mapping.
    SemanticWeight lower = adapter.map_token_to_weight("bullish");
    SemanticWeight upper = adapter.map_token_to_weight("BULLISH");
    SemanticWeight mixed = adapter.map_token_to_weight("Bullish");
    EXPECT_DOUBLE_EQ(lower.sentiment_score, upper.sentiment_score);
    EXPECT_DOUBLE_EQ(lower.sentiment_score, mixed.sentiment_score);
    EXPECT_DOUBLE_EQ(lower.directional_bias, upper.directional_bias);
}

TEST(LLMAdapterTest, test_llm_adapter_get_stats_hit_miss_sum_equals_processed) {
    LLMAdapter adapter;
    adapter.map_token_to_weight("bullish");    // hit
    adapter.map_token_to_weight("unknown_xyz"); // miss
    auto s = adapter.get_stats();
    EXPECT_EQ(s.cache_hits + s.cache_misses, s.tokens_processed)
        << "cache_hits + cache_misses must equal tokens_processed";
}

TEST(LLMAdapterTest, test_llm_adapter_crypto_bearish_tokens_negative_sentiment) {
    LLMAdapter adapter;
    for (const std::string& token : {"rug", "rekt", "fud"}) {
        SemanticWeight w = adapter.map_token_to_weight(token);
        EXPECT_LT(w.sentiment_score, 0.0)
            << "Crypto bearish token '" << token << "' must have negative sentiment";
        EXPECT_GT(w.confidence_score, 0.5)
            << "Crypto bearish token '" << token << "' must have above-neutral confidence";
    }
}

TEST(LLMAdapterTest, test_llm_adapter_crypto_bullish_tokens_positive_sentiment) {
    LLMAdapter adapter;
    SemanticWeight pump = adapter.map_token_to_weight("pump");
    SemanticWeight ath  = adapter.map_token_to_weight("ath");
    EXPECT_GT(pump.sentiment_score, 0.0) << "'pump' must have positive sentiment";
    EXPECT_GT(pump.directional_bias, 0.0) << "'pump' must have positive directional bias";
    EXPECT_GT(ath.sentiment_score, 0.0)  << "'ath' must have positive sentiment";
}

TEST(LLMAdapterTest, test_llm_adapter_get_dictionary_size_is_large_enough) {
    LLMAdapter adapter;
    EXPECT_GE(adapter.get_dictionary_size(), 60u)
        << "Default dictionary must contain at least 60 unique mappings";
}

TEST(LLMAdapterTest, test_llm_adapter_clear_custom_mappings_empties_dictionary) {
    LLMAdapter adapter;
    ASSERT_GE(adapter.get_dictionary_size(), 1u);
    adapter.clear_custom_mappings();
    EXPECT_EQ(adapter.get_dictionary_size(), 0u);
    // After clearing, any lookup returns the neutral default.
    SemanticWeight w = adapter.map_token_to_weight("bullish");
    EXPECT_DOUBLE_EQ(w.sentiment_score,  0.0);
    EXPECT_DOUBLE_EQ(w.directional_bias, 0.0);
    EXPECT_DOUBLE_EQ(w.confidence_score, 0.5);
}

TEST(LLMAdapterTest, test_llm_adapter_tokens_processed_increments_per_call) {
    LLMAdapter adapter;
    auto before = adapter.get_stats();
    adapter.map_token_to_weight("crash");
    adapter.map_token_to_weight("panic");
    adapter.map_token_to_weight("xyzzy_unknown_42");
    auto after = adapter.get_stats();
    EXPECT_EQ(after.tokens_processed, before.tokens_processed + 3u)
        << "tokens_processed must increment by exactly 1 per map_token_to_weight call";
}

// ---------------------------------------------------------------------------
// load_sentiment_dictionary tests (zero coverage before this cycle)
// ---------------------------------------------------------------------------

TEST(LLMAdapterTest, test_llm_adapter_load_sentiment_dictionary_valid_file) {
    const std::string tmp = "/tmp/llmquant_test_dict.txt";
    {
        std::ofstream f(tmp);
        f << "moon     0.8  0.9  0.6  0.85\n";
        f << "doom    -0.9  0.9  0.8 -0.90\n";
        f << "  \n";                         // blank line — must be skipped
        f << "# comment line\n";            // comment — must be skipped (iss >> fails)
    }

    LLMAdapter adapter;
    adapter.clear_custom_mappings();
    ASSERT_NO_THROW(adapter.load_sentiment_dictionary(tmp));

    SemanticWeight moon = adapter.map_token_to_weight("moon");
    EXPECT_NEAR(moon.sentiment_score,  0.8,  1e-9);
    EXPECT_NEAR(moon.confidence_score, 0.9,  1e-9);
    EXPECT_NEAR(moon.directional_bias, 0.85, 1e-9);

    SemanticWeight doom = adapter.map_token_to_weight("doom");
    EXPECT_LT(doom.sentiment_score,  0.0);
    EXPECT_LT(doom.directional_bias, 0.0);

    std::remove(tmp.c_str());
}

TEST(LLMAdapterTest, test_llm_adapter_load_sentiment_dictionary_nonexistent_file_throws) {
    LLMAdapter adapter;
    EXPECT_THROW(
        adapter.load_sentiment_dictionary("/does/not/exist/vocab.txt"),
        std::runtime_error);
}

TEST(LLMAdapterTest, test_llm_adapter_load_sentiment_dictionary_case_normalises_tokens) {
    const std::string tmp = "/tmp/llmquant_test_dict_case.txt";
    {
        std::ofstream f(tmp);
        f << "SURGE  0.5 0.8 0.7 0.6\n";  // uppercase — must normalize to "surge"
    }

    LLMAdapter adapter;
    adapter.clear_custom_mappings();
    adapter.load_sentiment_dictionary(tmp);

    // Lookup via lowercase must find the mapping.
    SemanticWeight w = adapter.map_token_to_weight("surge");
    EXPECT_NEAR(w.sentiment_score, 0.5, 1e-9);

    std::remove(tmp.c_str());
}

TEST(LLMAdapterTest, test_llm_adapter_reset_stats_clears_counters) {
    LLMAdapter adapter;

    // Process some tokens to populate the stats.
    adapter.map_token_to_weight("bullish");    // cache hit
    adapter.map_token_to_weight("unknown_xyz"); // cache miss
    ASSERT_GT(adapter.get_stats().tokens_processed, 0u);

    adapter.reset_stats();

    auto stats = adapter.get_stats();
    EXPECT_EQ(stats.tokens_processed, 0u);
    EXPECT_EQ(stats.cache_hits,       0u);
    EXPECT_EQ(stats.cache_misses,     0u);

    // Must still work normally after reset.
    adapter.map_token_to_weight("bearish");
    EXPECT_EQ(adapter.get_stats().tokens_processed, 1u);
}

TEST(LLMAdapterTest, test_llm_adapter_reset_stats_preserves_dictionary) {
    LLMAdapter adapter;
    size_t dict_before = adapter.get_dictionary_size();
    ASSERT_GT(dict_before, 0u);

    // Process some tokens to populate counters, then reset.
    adapter.map_token_to_weight("bullish");
    adapter.reset_stats();

    // Dictionary must be unchanged — reset_stats only clears counters.
    EXPECT_EQ(adapter.get_dictionary_size(), dict_before);
    // Known tokens still return the correct weight after stat reset.
    SemanticWeight w = adapter.map_token_to_weight("bullish");
    EXPECT_GT(w.directional_bias, 0.0);
}

} // namespace
} // namespace llmquant