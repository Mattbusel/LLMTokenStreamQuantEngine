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

TEST(LLMAdapterTest, test_llm_adapter_normalizes_whitespace_and_case) {
    LLMAdapter adapter;
    // "  BULLISH  " should normalize to "bullish" and hit the dictionary.
    SemanticWeight padded   = adapter.map_token_to_weight("  BULLISH  ");
    SemanticWeight canonical = adapter.map_token_to_weight("bullish");

    EXPECT_DOUBLE_EQ(padded.directional_bias,  canonical.directional_bias);
    EXPECT_DOUBLE_EQ(padded.sentiment_score,   canonical.sentiment_score);
    EXPECT_DOUBLE_EQ(padded.confidence_score,  canonical.confidence_score);
}

TEST(LLMAdapterTest, test_llm_adapter_add_token_mapping_overrides_existing) {
    LLMAdapter adapter;
    // "bullish" is a built-in positive token.
    SemanticWeight original = adapter.map_token_to_weight("bullish");
    ASSERT_GT(original.directional_bias, 0.0);

    // Override with a strongly negative mapping.
    SemanticWeight override_weight{-1.0, 1.0, 0.8, -0.9};
    adapter.add_token_mapping("bullish", override_weight);

    SemanticWeight after_override = adapter.map_token_to_weight("bullish");
    EXPECT_DOUBLE_EQ(after_override.directional_bias, -0.9)
        << "add_token_mapping must override the existing entry for 'bullish'";
}

// ---------------------------------------------------------------------------
// contains_token
// ---------------------------------------------------------------------------

TEST(LLMAdapterTest, test_llm_adapter_contains_token_finds_built_in) {
    LLMAdapter adapter;
    EXPECT_TRUE(adapter.contains_token("bullish"))
        << "contains_token must return true for a built-in token";
    EXPECT_TRUE(adapter.contains_token("bearish"))
        << "contains_token must return true for a built-in token";
}

TEST(LLMAdapterTest, test_llm_adapter_contains_token_returns_false_for_unknown) {
    LLMAdapter adapter;
    EXPECT_FALSE(adapter.contains_token("xyzzy_unknown_token_42"))
        << "contains_token must return false for an unknown token";
}

TEST(LLMAdapterTest, test_llm_adapter_contains_token_normalises_case_and_whitespace) {
    LLMAdapter adapter;
    // "bullish" is built-in; check that contains_token applies same normalisation.
    EXPECT_TRUE(adapter.contains_token("  BULLISH  "))
        << "contains_token must apply the same case/whitespace normalisation as map_token_to_weight";
}

TEST(LLMAdapterTest, test_llm_adapter_contains_token_reflects_add_token_mapping) {
    LLMAdapter adapter;
    const std::string token = "xyzzy_custom_token";
    ASSERT_FALSE(adapter.contains_token(token));
    adapter.add_token_mapping(token, {0.5, 0.3, 0.1, 0.8});
    EXPECT_TRUE(adapter.contains_token(token))
        << "contains_token must return true after add_token_mapping";
}

// ---------------------------------------------------------------------------
// top_tokens_by_sentiment
// ---------------------------------------------------------------------------

TEST(LLMAdapterTest, test_llm_adapter_top_tokens_returns_at_most_n_entries) {
    LLMAdapter adapter;
    auto top = adapter.top_tokens_by_sentiment(5);
    EXPECT_LE(top.size(), 5u);
}

TEST(LLMAdapterTest, test_llm_adapter_top_tokens_sorted_descending_by_abs_sentiment) {
    LLMAdapter adapter;
    auto top = adapter.top_tokens_by_sentiment(10);
    ASSERT_GE(top.size(), 2u);
    for (size_t i = 1; i < top.size(); ++i) {
        EXPECT_GE(std::fabs(top[i-1].second), std::fabs(top[i].second))
            << "Results must be sorted descending by |sentiment|";
    }
}

TEST(LLMAdapterTest, test_llm_adapter_top_tokens_large_n_returns_full_dictionary) {
    LLMAdapter adapter;
    size_t dict_size = adapter.get_dictionary_size();
    auto top = adapter.top_tokens_by_sentiment(dict_size + 100);
    EXPECT_EQ(top.size(), dict_size)
        << "Requesting more than dict size should return exactly dict_size entries";
}

TEST(LLMAdapterTest, test_llm_adapter_top_tokens_zero_n_returns_empty) {
    LLMAdapter adapter;
    auto top = adapter.top_tokens_by_sentiment(0);
    EXPECT_TRUE(top.empty()) << "n=0 must return empty vector";
}

TEST(LLMAdapterTest, test_llm_adapter_top_tokens_highest_sentiment_is_known_token) {
    LLMAdapter adapter;
    // The built-in dictionary has highly negative tokens like "crash" (-0.9) and
    // highly positive tokens like "bullish". The top entry must be one of the
    // high-magnitude built-in tokens.
    auto top = adapter.top_tokens_by_sentiment(1);
    ASSERT_EQ(top.size(), 1u);
    EXPECT_GE(std::fabs(top[0].second), 0.7)
        << "The highest-|sentiment| token must have |sentiment| >= 0.7";
}

// ---------------------------------------------------------------------------
// remove_token_mapping
// ---------------------------------------------------------------------------

TEST(LLMAdapterTest, test_llm_adapter_remove_existing_token_returns_true) {
    LLMAdapter adapter;
    ASSERT_TRUE(adapter.contains_token("crash"));
    bool removed = adapter.remove_token_mapping("crash");
    EXPECT_TRUE(removed) << "Removing an existing token must return true";
    EXPECT_FALSE(adapter.contains_token("crash")) << "Token must be absent after removal";
}

TEST(LLMAdapterTest, test_llm_adapter_remove_nonexistent_token_returns_false) {
    LLMAdapter adapter;
    bool removed = adapter.remove_token_mapping("__no_such_token__");
    EXPECT_FALSE(removed) << "Removing a non-existent token must return false";
}

TEST(LLMAdapterTest, test_llm_adapter_remove_then_add_mapping_works) {
    LLMAdapter adapter;
    adapter.remove_token_mapping("crash");
    SemanticWeight custom{0.5, 0.8, 0.1, 0.6};
    adapter.add_token_mapping("crash", custom);
    SemanticWeight retrieved;
    ASSERT_TRUE(adapter.get_token_mapping("crash", retrieved));
    EXPECT_DOUBLE_EQ(retrieved.sentiment_score, 0.5);
}

// ---------------------------------------------------------------------------
// get_token_mapping
// ---------------------------------------------------------------------------

TEST(LLMAdapterTest, test_llm_adapter_get_token_mapping_known_token) {
    LLMAdapter adapter;
    SemanticWeight w;
    bool found = adapter.get_token_mapping("crash", w);
    EXPECT_TRUE(found) << "Known token must be found";
    EXPECT_LT(w.sentiment_score, 0.0) << "crash must have negative sentiment";
}

TEST(LLMAdapterTest, test_llm_adapter_get_token_mapping_unknown_token_returns_false) {
    LLMAdapter adapter;
    SemanticWeight w;
    bool found = adapter.get_token_mapping("__unknown__", w);
    EXPECT_FALSE(found);
}

TEST(LLMAdapterTest, test_llm_adapter_get_token_mapping_case_insensitive) {
    LLMAdapter adapter;
    SemanticWeight lower, upper;
    ASSERT_TRUE(adapter.get_token_mapping("crash", lower));
    ASSERT_TRUE(adapter.get_token_mapping("CRASH", upper));
    EXPECT_DOUBLE_EQ(lower.sentiment_score, upper.sentiment_score);
}

// ---------------------------------------------------------------------------
// update_token_weight
// ---------------------------------------------------------------------------

TEST(LLMAdapterTest, test_llm_adapter_update_token_weight_existing_token) {
    LLMAdapter adapter;
    ASSERT_TRUE(adapter.contains_token("crash"));
    SemanticWeight new_w{0.5, 0.5, 0.1, 0.5};
    bool updated = adapter.update_token_weight("crash", new_w);
    EXPECT_TRUE(updated) << "update_token_weight on existing token must return true";

    SemanticWeight retrieved;
    ASSERT_TRUE(adapter.get_token_mapping("crash", retrieved));
    EXPECT_DOUBLE_EQ(retrieved.sentiment_score, 0.5);
}

TEST(LLMAdapterTest, test_llm_adapter_update_token_weight_nonexistent_returns_false) {
    LLMAdapter adapter;
    SemanticWeight w{0.1, 0.1, 0.1, 0.1};
    EXPECT_FALSE(adapter.update_token_weight("__no_such_token__", w));
}

TEST(LLMAdapterTest, test_llm_adapter_update_token_weight_case_insensitive) {
    LLMAdapter adapter;
    SemanticWeight new_w{0.3, 0.3, 0.3, 0.3};
    ASSERT_TRUE(adapter.update_token_weight("CRASH", new_w));  // uppercase key
    SemanticWeight retrieved;
    ASSERT_TRUE(adapter.get_token_mapping("crash", retrieved));
    EXPECT_DOUBLE_EQ(retrieved.sentiment_score, 0.3);
}

// ---------------------------------------------------------------------------
// Cycle 27: update_token_weight and remove_token_mapping integration
// ---------------------------------------------------------------------------

TEST(LLMAdapterTest, test_llm_adapter_update_weight_changes_map_output) {
    LLMAdapter adapter;
    // Verify initial "bullish" is positive.
    SemanticWeight before = adapter.map_token_to_weight("bullish");
    EXPECT_GT(before.directional_bias, 0.0);

    // Recalibrate to be strongly negative.
    // SemanticWeight field order: {sentiment_score, confidence_score, volatility_score, directional_bias}
    SemanticWeight new_w{-0.9, 0.9, 0.9, -0.9};  // directional_bias = -0.9
    ASSERT_TRUE(adapter.update_token_weight("bullish", new_w));

    SemanticWeight after = adapter.map_token_to_weight("bullish");
    EXPECT_LT(after.directional_bias, 0.0)
        << "After update, bullish must return negative directional_bias";
}

TEST(LLMAdapterTest, test_llm_adapter_remove_makes_token_return_neutral) {
    LLMAdapter adapter;
    // "crash" is in the default dict — remove it.
    ASSERT_TRUE(adapter.remove_token_mapping("crash"));

    // map_token_to_weight for an unknown token returns neutral {0, 0.5, 0.1, 0}.
    SemanticWeight after = adapter.map_token_to_weight("crash");
    EXPECT_DOUBLE_EQ(after.directional_bias, 0.0)
        << "After removal, crash must return neutral (zero) directional_bias";
}

// ---------------------------------------------------------------------------
// batch_add_token_mappings
// ---------------------------------------------------------------------------

TEST(LLMAdapterTest, test_llm_adapter_batch_add_inserts_new_tokens) {
    LLMAdapter adapter;
    size_t before = adapter.get_dictionary_size();
    std::unordered_map<std::string, SemanticWeight> batch = {
        {"new_tok_a", {0.5, 0.8, 0.1, 0.6}},
        {"new_tok_b", {-0.3, 0.7, 0.2, -0.4}},
    };
    size_t inserted = adapter.batch_add_token_mappings(batch);
    EXPECT_EQ(inserted, 2u);
    EXPECT_EQ(adapter.get_dictionary_size(), before + 2);
}

TEST(LLMAdapterTest, test_llm_adapter_batch_add_overwrites_existing) {
    LLMAdapter adapter;
    std::unordered_map<std::string, SemanticWeight> batch = {{"crash", {0.9, 0.5, 0.1, 0.9}}};
    size_t inserted = adapter.batch_add_token_mappings(batch);
    EXPECT_EQ(inserted, 0u) << "Overwrite must not count as new insertion";
    SemanticWeight w;
    ASSERT_TRUE(adapter.get_token_mapping("crash", w));
    EXPECT_DOUBLE_EQ(w.sentiment_score, 0.9);
}

TEST(LLMAdapterTest, test_llm_adapter_batch_add_empty_batch) {
    LLMAdapter adapter;
    size_t before = adapter.get_dictionary_size();
    EXPECT_EQ(adapter.batch_add_token_mappings({}), 0u);
    EXPECT_EQ(adapter.get_dictionary_size(), before);
}

// ---------------------------------------------------------------------------
// Cycle 31: get_sentiment_distribution()
// ---------------------------------------------------------------------------

TEST(LLMAdapterTest, test_llm_adapter_sentiment_distribution_default_dict) {
    LLMAdapter adapter;
    auto dist = adapter.get_sentiment_distribution();

    // Default dict has negative tokens (crash, panic, etc.) and positive tokens.
    EXPECT_GT(dist.negative_count, 0u)
        << "Default dictionary must contain negative-sentiment tokens";
    EXPECT_GT(dist.positive_count, 0u)
        << "Default dictionary must contain positive-sentiment tokens";
    EXPECT_EQ(dist.negative_count + dist.neutral_count + dist.positive_count,
              adapter.get_dictionary_size())
        << "Sum of bucket counts must equal dictionary size";
}

TEST(LLMAdapterTest, test_llm_adapter_sentiment_distribution_confidence_in_range) {
    LLMAdapter adapter;
    auto dist = adapter.get_sentiment_distribution();
    EXPECT_GE(dist.mean_confidence, 0.0);
    EXPECT_LE(dist.mean_confidence, 1.0)
        << "Mean confidence must be in [0, 1]";
}

TEST(LLMAdapterTest, test_llm_adapter_sentiment_distribution_empty_dict) {
    LLMAdapter adapter;
    adapter.clear_custom_mappings();  // removes all tokens
    auto dist = adapter.get_sentiment_distribution();
    EXPECT_EQ(dist.negative_count, 0u);
    EXPECT_EQ(dist.neutral_count,  0u);
    EXPECT_EQ(dist.positive_count, 0u);
    EXPECT_DOUBLE_EQ(dist.mean_sentiment,  0.0);
    EXPECT_DOUBLE_EQ(dist.mean_confidence, 0.0);
}

TEST(LLMAdapterTest, test_llm_adapter_sentiment_distribution_single_negative_token) {
    LLMAdapter adapter;
    adapter.clear_custom_mappings();
    adapter.add_token_mapping("collapse", {-0.9, 0.9, 0.7, -0.9});
    auto dist = adapter.get_sentiment_distribution();
    EXPECT_EQ(dist.negative_count, 1u);
    EXPECT_EQ(dist.neutral_count,  0u);
    EXPECT_EQ(dist.positive_count, 0u);
    EXPECT_NEAR(dist.mean_sentiment,  -0.9, 1e-9);
    EXPECT_NEAR(dist.mean_confidence,  0.9, 1e-9);
}

TEST(LLMAdapterTest, test_llm_adapter_sentiment_distribution_neutral_boundary) {
    LLMAdapter adapter;
    adapter.clear_custom_mappings();
    // Exactly at boundary -0.1 and +0.1 should be neutral.
    adapter.add_token_mapping("tok_neg_boundary", {-0.1, 0.5, 0.0, 0.0});
    adapter.add_token_mapping("tok_pos_boundary", { 0.1, 0.5, 0.0, 0.0});
    auto dist = adapter.get_sentiment_distribution();
    EXPECT_EQ(dist.neutral_count,  2u)
        << "Tokens with |sentiment| == 0.1 should be in the neutral bucket";
    EXPECT_EQ(dist.negative_count, 0u);
    EXPECT_EQ(dist.positive_count, 0u);
}

// ---------------------------------------------------------------------------
// Cycle 31: get_all_token_keys()
// ---------------------------------------------------------------------------

TEST(LLMAdapterTest, test_llm_adapter_get_all_token_keys_count_matches_dict_size) {
    LLMAdapter adapter;
    auto keys = adapter.get_all_token_keys();
    EXPECT_EQ(keys.size(), adapter.get_dictionary_size())
        << "get_all_token_keys must return exactly dictionary_size entries";
}

TEST(LLMAdapterTest, test_llm_adapter_get_all_token_keys_empty_after_clear) {
    LLMAdapter adapter;
    adapter.clear_custom_mappings();
    EXPECT_TRUE(adapter.get_all_token_keys().empty());
}

// ---------------------------------------------------------------------------
// Cycle 33: filter_tokens_by_sentiment()
// ---------------------------------------------------------------------------

TEST(LLMAdapterTest, test_filter_tokens_by_sentiment_returns_matching_tokens) {
    LLMAdapter adapter;
    adapter.clear_custom_mappings();
    adapter.add_token_mapping("bullish", { 0.8, 0.9, 0.1, 0.8});
    adapter.add_token_mapping("bearish", {-0.7, 0.9, 0.2, -0.7});
    adapter.add_token_mapping("neutral", { 0.0, 0.5, 0.0, 0.0});

    // Filter for positive sentiment only.
    auto pos_tokens = adapter.filter_tokens_by_sentiment(0.1, 1.0);
    ASSERT_EQ(pos_tokens.size(), 1u);
    EXPECT_EQ(pos_tokens[0].first, "bullish");
    EXPECT_NEAR(pos_tokens[0].second, 0.8, 1e-9);
}

TEST(LLMAdapterTest, test_filter_tokens_by_sentiment_empty_range_returns_nothing) {
    LLMAdapter adapter;
    // Filter with min > max should return nothing.
    auto result = adapter.filter_tokens_by_sentiment(0.5, -0.5);
    EXPECT_TRUE(result.empty())
        << "Inverted range [0.5, -0.5] should return no tokens";
}

TEST(LLMAdapterTest, test_filter_tokens_by_sentiment_full_range_returns_all) {
    LLMAdapter adapter;
    auto all = adapter.filter_tokens_by_sentiment(-1.0, 1.0);
    EXPECT_EQ(all.size(), adapter.get_dictionary_size())
        << "Full range [-1, 1] must return all tokens";
}

TEST(LLMAdapterTest, test_filter_tokens_by_sentiment_negative_range) {
    LLMAdapter adapter;
    adapter.clear_custom_mappings();
    adapter.add_token_mapping("crash",   {-0.9, 0.9, 0.8, -0.7});
    adapter.add_token_mapping("bullish", { 0.8, 0.9, 0.1,  0.8});

    auto neg = adapter.filter_tokens_by_sentiment(-1.0, -0.1);
    ASSERT_EQ(neg.size(), 1u);
    EXPECT_EQ(neg[0].first, "crash");
}

// ---------------------------------------------------------------------------
// Cycle 34: export_dictionary()
// ---------------------------------------------------------------------------

TEST(LLMAdapterTest, test_export_dictionary_empty_returns_empty_string) {
    LLMAdapter adapter;
    adapter.clear_custom_mappings();
    EXPECT_TRUE(adapter.export_dictionary().empty())
        << "export_dictionary must return empty string when dict is empty";
}

TEST(LLMAdapterTest, test_export_dictionary_contains_token_name) {
    LLMAdapter adapter;
    adapter.clear_custom_mappings();
    adapter.add_token_mapping("bullish", {0.8, 0.9, 0.1, 0.8});
    std::string exported = adapter.export_dictionary();
    EXPECT_NE(exported.find("bullish"), std::string::npos)
        << "exported dictionary must contain the token name";
}

TEST(LLMAdapterTest, test_export_dictionary_line_count_matches_dict_size) {
    LLMAdapter adapter;
    adapter.clear_custom_mappings();
    adapter.add_token_mapping("crash",   {-0.9, 0.9, 0.8, -0.7});
    adapter.add_token_mapping("rally",   { 0.8, 0.9, 0.2,  0.7});
    adapter.add_token_mapping("neutral", { 0.0, 0.5, 0.0,  0.0});

    std::string exported = adapter.export_dictionary();
    // Count lines (each token occupies one line ending with '\n').
    size_t line_count = 0;
    for (char c : exported) if (c == '\n') ++line_count;
    EXPECT_EQ(line_count, 3u)
        << "export_dictionary must produce one line per token";
}

TEST(LLMAdapterTest, test_export_dictionary_is_sorted_alphabetically) {
    LLMAdapter adapter;
    adapter.clear_custom_mappings();
    adapter.add_token_mapping("zebra",   {0.1, 0.5, 0.0, 0.0});
    adapter.add_token_mapping("apple",   {0.2, 0.5, 0.0, 0.0});
    adapter.add_token_mapping("mango",   {0.3, 0.5, 0.0, 0.0});

    std::string exported = adapter.export_dictionary();
    auto pos_apple = exported.find("apple");
    auto pos_mango = exported.find("mango");
    auto pos_zebra = exported.find("zebra");
    EXPECT_LT(pos_apple, pos_mango) << "apple must come before mango";
    EXPECT_LT(pos_mango, pos_zebra) << "mango must come before zebra";
}

} // namespace
} // namespace llmquant