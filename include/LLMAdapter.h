#pragma once

#include <string>
#include <utility>
#include <vector>
#include <unordered_map>
#include <atomic>
#include <stdexcept>
#include <immintrin.h>  // SSE2/AVX2 intrinsics

namespace llmquant {

/**
 * @brief Normalised semantic weight extracted from a single token or token sequence.
 *
 * All fields are in the range [-1.0, 1.0] except confidence_score which is
 * in [0.0, 1.0]. A fully neutral token has all fields at 0.0 except
 * confidence_score which defaults to 0.5.
 */
struct SemanticWeight {
    /** @brief Overall sentiment polarity: negative = bearish/fearful, positive = bullish. */
    double sentiment_score{0.0};
    /** @brief How strongly the model believes this mapping is accurate (0 = none, 1 = certain). */
    double confidence_score{0.5};
    /** @brief Implied market volatility contribution (0 = calm, 1 = high volatility). */
    double volatility_score{0.0};
    /** @brief Directional market bias: negative = sell pressure, positive = buy pressure. */
    double directional_bias{0.0};
};

/**
 * @brief Maps raw LLM tokens to their quantitative SemanticWeight representations.
 *
 * A default dictionary of ~40 high-signal tokens is loaded at construction.
 * Additional mappings can be injected at runtime via add_token_mapping() or
 * loaded in bulk from a tab-separated dictionary file.
 *
 * Thread safety: map_token_to_weight() and map_sequence_to_weight() are
 * safe to call from multiple threads concurrently (atomic stat counters,
 * read-only map access after initialisation). Mutation methods
 * (add_token_mapping, load_sentiment_dictionary) must not be called
 * concurrently with read methods.
 */
class LLMAdapter {
public:
    /**
     * @brief Construct an adapter pre-loaded with the built-in default token dictionary.
     */
    LLMAdapter();

    // LLMAdapter is non-copyable and non-movable due to atomic members.
    LLMAdapter(const LLMAdapter&) = delete;
    LLMAdapter& operator=(const LLMAdapter&) = delete;
    LLMAdapter(LLMAdapter&&) = delete;
    LLMAdapter& operator=(LLMAdapter&&) = delete;

    /**
     * @brief Look up the SemanticWeight for a single token.
     *
     * @param token Raw token string (case-sensitive).
     * @return The registered SemanticWeight, or a neutral weight
     *         {0.0, 0.5, 0.1, 0.0} if the token is not in the dictionary.
     */
    SemanticWeight map_token_to_weight(const std::string& token) const;

    /**
     * @brief Compute a confidence-weighted aggregate SemanticWeight for a token sequence.
     *
     * Each token is looked up individually; the results are averaged with each
     * token's confidence_score used as its weight.
     *
     * @param tokens Ordered list of raw token strings.
     * @return Aggregated SemanticWeight, or a zero weight if tokens is empty.
     */
    SemanticWeight map_sequence_to_weight(const std::vector<std::string>& tokens) const;

    /**
     * @brief Load additional token-to-weight mappings from a whitespace-delimited file.
     *
     * Each line must contain: token sentiment confidence volatility bias
     *
     * @param filepath Path to the dictionary file.
     * @throws std::runtime_error if the file cannot be opened.
     */
    void load_sentiment_dictionary(const std::string& filepath);

    /**
     * @brief Insert or overwrite a single token mapping.
     *
     * @param token  Raw token string (case-sensitive).
     * @param weight SemanticWeight to associate with the token.
     */
    void add_token_mapping(const std::string& token, const SemanticWeight& weight);

    /**
     * @brief Batch-score a sequence of tokens using SSE2-accelerated aggregation.
     *
     * Processes token pairs with SSE2 intrinsics, accumulating four
     * confidence-weighted sums (sentiment, volatility, directional bias, and
     * total confidence) across two 128-bit registers. A scalar tail loop
     * handles any odd remaining token. The final result is the
     * confidence-weighted average of all four fields, identical in semantics
     * to map_sequence_to_weight() but faster on sequences of two or more tokens.
     *
     * @param tokens Tokens to score; may be empty (returns zero weight).
     * @return Confidence-weighted average SemanticWeight computed via SSE2.
     */
    SemanticWeight map_sequence_simd(const std::vector<std::string>& tokens) const;

    /**
     * @brief Returns the number of token-to-weight mappings currently loaded.
     *
     * @return Total number of entries in the token dictionary (built-in + custom).
     */
    size_t get_dictionary_size() const;

    /**
     * @brief Removes all mappings from the dictionary.
     *
     * Note: this clears both built-in and custom mappings. After calling this
     * method, all tokens will return the neutral default weight until new
     * mappings are added via add_token_mapping() or load_sentiment_dictionary().
     */
    void clear_custom_mappings();

    /**
     * @brief Return true if the dictionary contains a mapping for the given token.
     *
     * Applies the same normalisation (lowercase + strip whitespace) as
     * map_token_to_weight(). Does NOT update the stats counters.
     *
     * @param token Raw token string.
     * @return true if a mapping exists; false otherwise.
     */
    bool contains_token(const std::string& token) const;

    /**
     * @brief Remove a single token mapping from the dictionary.
     *
     * If the token does not exist, this is a no-op.
     * Applies the same normalisation as map_token_to_weight().
     *
     * @param token Raw token string to remove.
     * @return true if a mapping was removed; false if it did not exist.
     */
    bool remove_token_mapping(const std::string& token);

    /**
     * @brief Update the numeric fields of an existing token mapping in-place.
     *
     * Only modifies an entry if it already exists in the dictionary.
     * Useful for live recalibration without the remove + add roundtrip.
     * Applies the same normalisation as map_token_to_weight().
     *
     * @param token  Raw token string.
     * @param weight New weight values to store.
     * @return true if the entry was found and updated; false if not found.
     */
    bool update_token_weight(const std::string& token, const SemanticWeight& weight);

    /**
     * @brief Retrieve the SemanticWeight associated with a token.
     *
     * Applies the same normalisation as map_token_to_weight().
     * Does NOT update stats counters.
     *
     * @param token  Raw token string.
     * @param weight Output parameter populated with the mapping on success.
     * @return true if found; false if the token is not in the dictionary
     *         (weight is left unchanged on false return).
     */
    bool get_token_mapping(const std::string& token, SemanticWeight& weight) const;

    /**
     * @brief Return the top N tokens sorted by absolute sentiment score.
     *
     * Useful for auditing which tokens are most influential in driving
     * signal generation.  Ties are broken by dictionary insertion order
     * (unspecified).
     *
     * @param n Maximum number of tokens to return (default: 10).
     *          If n >= dictionary size, all tokens are returned.
     * @return Vector of (token, sentiment_score) pairs, sorted descending
     *         by |sentiment_score|.
     */
    std::vector<std::pair<std::string, double>>
        top_tokens_by_sentiment(size_t n = 10) const;

    /**
     * @brief Return a snapshot of internal processing statistics.
     *
     * @return Struct with tokens_processed, cache_hits and cache_misses counts.
     */
    struct Stats {
        uint64_t tokens_processed;
        uint64_t cache_hits;
        uint64_t cache_misses;
    };

    /** @brief Return current processing statistics. */
    Stats get_stats() const noexcept {
        return Stats{
            stats_.tokens_processed.load(),
            stats_.cache_hits.load(),
            stats_.cache_misses.load()
        };
    }

    /**
     * @brief Reset all processing statistics (tokens_processed, cache_hits, cache_misses) to zero.
     *
     * Thread-safe: each counter is independently atomic.
     * Useful when starting a new session without restarting the process.
     */
    void reset_stats() noexcept {
        stats_.tokens_processed.store(0, std::memory_order_relaxed);
        stats_.cache_hits.store(0, std::memory_order_relaxed);
        stats_.cache_misses.store(0, std::memory_order_relaxed);
    }

private:
    void initialize_default_mappings();

    /**
     * @brief Strip leading/trailing whitespace and lowercase a token string.
     *
     * Used by both map_token_to_weight() and add_token_mapping() to ensure
     * lookups and insertions use the same canonical form.
     *
     * @param token Raw token string.
     * @return Normalized lowercase token with no leading/trailing whitespace.
     */
    static std::string normalize_token(const std::string& token);

    /**
     * @brief Scalar fallback for confidence-weighted aggregation over [begin, end).
     *
     * @param weights Vector of SemanticWeight values.
     * @param begin   Start index (inclusive).
     * @param end     End index (exclusive).
     * @return Confidence-weighted average over the specified range.
     */
    static SemanticWeight aggregate_scalar(const std::vector<SemanticWeight>& weights,
                                           size_t begin, size_t end);

    std::unordered_map<std::string, SemanticWeight> token_weights_;

    /** @brief Internal statistics; mutable so const query methods can update them. */
    mutable struct {
        std::atomic<uint64_t> tokens_processed{0};
        std::atomic<uint64_t> cache_hits{0};
        std::atomic<uint64_t> cache_misses{0};
    } stats_;
};

} // namespace llmquant
