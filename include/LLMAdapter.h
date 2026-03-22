#pragma once

#include <cinttypes>
#include <cstdio>
#include <string>
#include <utility>
#include <vector>
#include <unordered_map>
#include <atomic>
#include <memory>
#include <stdexcept>

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
    [[nodiscard]] SemanticWeight map_token_to_weight(const std::string& token) const;

    /**
     * @brief Compute a confidence-weighted aggregate SemanticWeight for a token sequence.
     *
     * Each token is looked up individually; the results are averaged with each
     * token's confidence_score used as its weight.
     *
     * @param tokens Ordered list of raw token strings.
     * @return Aggregated SemanticWeight, or a zero weight if tokens is empty.
     */
    [[nodiscard]] SemanticWeight map_sequence_to_weight(const std::vector<std::string>& tokens) const;

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
    [[nodiscard]] SemanticWeight map_sequence_simd(const std::vector<std::string>& tokens) const;

    /**
     * @brief Map a list of tokens to their individual SemanticWeights in one call.
     *
     * Returns a weight per token using the same lookup logic as
     * map_token_to_weight().  Tokens not found in the dictionary are mapped
     * to a neutral SemanticWeight (all zeros).  Useful when per-token weights
     * are needed rather than a single aggregated sequence weight.
     *
     * @param tokens Vector of raw token strings (normalised internally).
     * @return Vector of SemanticWeights, one per input token, in the same order.
     */
    std::vector<SemanticWeight> batch_map_tokens_to_weights(
        const std::vector<std::string>& tokens) const;

    /**
     * @brief Returns the number of token-to-weight mappings currently loaded.
     *
     * @return Total number of entries in the token dictionary (built-in + custom).
     */
    size_t get_dictionary_size() const;

    /**
     * @brief Remove all entries from the token dictionary.
     *
     * After calling this method, all token lookups will return the neutral
     * fallback weight until new mappings are added via add_token_mapping() or
     * load_sentiment_dictionary().  Primarily useful in unit tests that need
     * a clean-slate adapter.
     *
     * Not safe to call concurrently with map_token_to_weight() or any other
     * method that reads token_weights_.
     */
    void clear_dictionary();

    /**
     * @brief Return all token keys currently in the dictionary.
     *
     * Order is unspecified (depends on the underlying hash map). Useful for
     * serialisation, inspection, and testing dictionary state.
     *
     * @return Vector of normalised token strings.
     */
    [[nodiscard]] std::vector<std::string> get_all_token_keys() const;

    /**
     * @brief Count tokens with positive sentiment_score (> 0).
     * @return Number of bullish tokens in the dictionary.
     */
    [[nodiscard]] size_t count_bullish_tokens() const;

    /**
     * @brief Count tokens with negative sentiment_score (< 0).
     * @return Number of bearish tokens in the dictionary.
     */
    [[nodiscard]] size_t count_bearish_tokens() const;

    /**
     * @brief Count tokens with sentiment_score == 0.0 (neutral).
     * @return Number of neutral tokens in the dictionary.
     */
    [[nodiscard]] size_t count_neutral_tokens() const;

    /**
     * @brief Return the mean confidence_score across all tokens in the dictionary.
     *
     * Returns 0.0 if the dictionary is empty.
     *
     * @return Average confidence_score in [0.0, 1.0].
     */
    [[nodiscard]] double get_avg_confidence() const;

    /**
     * @brief Return the minimum confidence_score across all dictionary tokens.
     *
     * Returns 0.0 if the dictionary is empty.
     *
     * @return Minimum confidence_score in [0.0, 1.0].
     */
    [[nodiscard]] double get_min_confidence() const;

    /**
     * @brief Return the maximum confidence_score across all dictionary tokens.
     *
     * Returns 0.0 if the dictionary is empty.
     *
     * @return Maximum confidence_score in [0.0, 1.0].
     */
    [[nodiscard]] double get_max_confidence() const;

    /**
     * @brief Return the spread between the highest and lowest confidence_score in the dictionary.
     *
     * Computed as get_max_confidence() - get_min_confidence().
     * Returns 0.0 if the dictionary is empty.
     *
     * @return Confidence range (always >= 0).
     */
    [[nodiscard]] double get_confidence_range() const;

    /**
     * @brief Return the mean sentiment_score across all tokens in the dictionary.
     *
     * Returns 0.0 if the dictionary is empty.
     *
     * @return Average sentiment_score in [-1.0, 1.0].
     */
    [[nodiscard]] double get_avg_sentiment() const;

    /**
     * @brief Return the mean volatility_score across all tokens in the dictionary.
     *
     * Returns 0.0 if the dictionary is empty.
     *
     * @return Average volatility_score in [0.0, 1.0].
     */
    [[nodiscard]] double get_avg_volatility() const;

    /**
     * @brief Return the mean directional_bias across all tokens in the dictionary.
     *
     * Returns 0.0 if the dictionary is empty.
     * A positive value indicates an overall bullish dictionary bias;
     * a negative value indicates a bearish bias.
     *
     * @return Average directional_bias in [-1.0, 1.0].
     */
    [[nodiscard]] double get_avg_directional_bias() const;

    /**
     * @brief Count tokens with volatility_score strictly above @p threshold.
     *
     * @param threshold Volatility threshold in [0.0, 1.0].
     * @return Number of tokens where volatility_score > threshold.
     */
    [[nodiscard]] size_t count_tokens_above_volatility(double threshold) const;

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
    [[nodiscard]] bool contains_token(const std::string& token) const;

    /**
     * @brief Return true if the dictionary contains at least one of the given tokens.
     *
     * Short-circuits on the first match; tokens are normalised the same way as
     * map_token_to_weight().  Returns false for an empty input list.
     *
     * @param tokens Candidate token strings (raw / unnormalised).
     * @return true if any of @p tokens is found in the dictionary.
     */
    [[nodiscard]] bool contains_any_of(const std::vector<std::string>& tokens) const;

    /**
     * @brief Remove a single token mapping from the dictionary.
     *
     * If the token does not exist, this is a no-op.
     * Applies the same normalisation as map_token_to_weight().
     *
     * @param token Raw token string to remove.
     * @return true if a mapping was removed; false if it did not exist.
     */
    [[nodiscard]] bool remove_token_mapping(const std::string& token);

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
    [[nodiscard]] bool update_token_weight(const std::string& token, const SemanticWeight& weight);

    /**
     * @brief Insert or overwrite multiple token mappings in one call.
     *
     * Equivalent to calling add_token_mapping() for each entry but avoids
     * repeated normalisation overhead.
     *
     * @param mappings Map of raw token string to SemanticWeight.
     * @return Number of new entries inserted (not counting overwrites).
     */
    [[nodiscard]] size_t batch_add_token_mappings(
        const std::unordered_map<std::string, SemanticWeight>& mappings);

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
    [[nodiscard]] bool get_token_mapping(const std::string& token, SemanticWeight& weight) const;

    /**
     * @brief Multiply every token's confidence_score by @p factor.
     *
     * Useful for simulating information aging in long-running sessions: after
     * calling this, older sentiment contributions have less influence on new
     * predictions until refreshed.  @p factor is clamped to [0.0, 1.0] so
     * calling with factor > 1.0 does not amplify confidence beyond the valid
     * range.  Factor < 0 is treated as 0 (zeroes all confidence scores).
     *
     * @param factor Decay multiplier in [0.0, 1.0]; clamped if out of range.
     */
    void decay_all_weights(double factor);

    /**
     * @brief Distribution of tokens across sentiment polarity buckets.
     */
    struct SentimentDistribution {
        size_t negative_count{0};  ///< Tokens with sentiment_score < -0.1.
        size_t neutral_count{0};   ///< Tokens with |sentiment_score| <= 0.1.
        size_t positive_count{0};  ///< Tokens with sentiment_score > 0.1.
        double mean_sentiment{0.0};   ///< Mean sentiment_score across all tokens.
        double mean_confidence{0.0};  ///< Mean confidence_score across all tokens.
    };

    /**
     * @brief Compute the sentiment distribution across all loaded dictionary tokens.
     *
     * Iterates the token dictionary and categorises each entry by sentiment_score
     * into negative (<-0.1), neutral ([-0.1, 0.1]), or positive (>0.1) buckets.
     * Also computes the mean sentiment and confidence across all tokens.
     *
     * @return SentimentDistribution with bucket counts and aggregate stats.
     */
    SentimentDistribution get_sentiment_distribution() const;

    /**
     * @brief Return all tokens whose sentiment_score falls within [min_sentiment, max_sentiment].
     *
     * Useful for debugging and auditing which tokens will drive bullish or
     * bearish signal energy.  Both bounds are inclusive.
     *
     * @param min_sentiment Lower bound (inclusive) on sentiment_score.
     * @param max_sentiment Upper bound (inclusive) on sentiment_score.
     * @return Vector of (token, sentiment_score) pairs in unspecified order.
     */
    [[nodiscard]] std::vector<std::pair<std::string, double>>
        filter_tokens_by_sentiment(double min_sentiment, double max_sentiment) const;

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
    [[nodiscard]] std::vector<std::pair<std::string, double>>
        top_tokens_by_sentiment(size_t n = 10) const;

    /**
     * @brief Return all tokens whose confidence_score is within [min_confidence, max_confidence].
     *
     * Useful for auditing low-confidence or high-confidence token sets.
     *
     * @param min_confidence Lower bound (inclusive) on confidence_score.
     * @param max_confidence Upper bound (inclusive) on confidence_score.
     * @return Vector of (token, confidence_score) pairs in unspecified order.
     */
    [[nodiscard]] std::vector<std::pair<std::string, double>>
        filter_tokens_by_confidence(double min_confidence, double max_confidence) const;

    /**
     * @brief Return all tokens whose volatility_score falls within [min_volatility, max_volatility].
     *
     * Useful for identifying tokens in a specific volatility band, e.g. all
     * medium-volatility tokens (0.4–0.7) for regime-specific filtering.
     * Both bounds are inclusive.
     *
     * @param min_volatility Lower bound (inclusive) on volatility_score.
     * @param max_volatility Upper bound (inclusive) on volatility_score.
     * @return Vector of (token, volatility_score) pairs in unspecified order.
     */
    [[nodiscard]] std::vector<std::pair<std::string, double>>
        filter_tokens_by_volatility(double min_volatility, double max_volatility) const;

    /**
     * @brief Return the top N tokens sorted by volatility_score descending.
     *
     * Identifies the most volatility-contributing tokens in the dictionary.
     *
     * @param n Maximum number of tokens to return (default: 10).
     *          If n >= dictionary size, all tokens are returned.
     * @return Vector of (token, volatility_score) pairs, sorted descending.
     */
    [[nodiscard]] std::vector<std::pair<std::string, double>>
        top_tokens_by_volatility(size_t n = 10) const;

    /**
     * @brief Return all tokens whose directional_bias falls within [min_bias, max_bias].
     *
     * Useful for isolating strongly bullish (> 0.5) or strongly bearish (< -0.5) tokens,
     * or finding near-neutral tokens ([-0.1, 0.1]) for noise analysis.
     * Both bounds are inclusive.
     *
     * @param min_bias Lower bound (inclusive) on directional_bias.
     * @param max_bias Upper bound (inclusive) on directional_bias.
     * @return Vector of (token, directional_bias) pairs in unspecified order.
     */
    [[nodiscard]] std::vector<std::pair<std::string, double>>
        filter_tokens_by_directional_bias(double min_bias, double max_bias) const;

    /**
     * @brief Return the top N tokens sorted by absolute directional_bias descending.
     *
     * Identifies the most directionally influential tokens regardless of sign.
     *
     * @param n Maximum number of tokens to return (default: 10).
     *          If n >= dictionary size, all tokens are returned.
     * @return Vector of (token, directional_bias) pairs, sorted descending by |directional_bias|.
     */
    [[nodiscard]] std::vector<std::pair<std::string, double>>
        top_tokens_by_directional_bias(size_t n = 10) const;

    /**
     * @brief Return the top N tokens by a composite influence score.
     *
     * The composite score blends normalised hit-frequency with absolute
     * directional_bias:
     *   score = 0.5 * (hit_count / max_hit_count) + 0.5 * |directional_bias|
     *
     * Tokens with a high score are both frequently encountered AND strongly
     * directional, making them the most impactful tokens in the current run.
     * Useful for debugging runaway bias or prioritising dictionary tuning.
     *
     * If the dictionary is empty or all hit-counts are zero the frequency
     * component is treated as 0 for every token.
     *
     * @param n Maximum number of tokens to return (default: 10).
     * @return Vector of (token, composite_score) sorted descending by score.
     */
    std::vector<std::pair<std::string, double>>
        export_hot_tokens(size_t n = 10) const;

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
     * @brief Return the number of times a specific dictionary token has been looked up
     *        and found (cache hit) since construction or the last reset_frequency_counts().
     *
     * Applies the same normalisation as map_token_to_weight().
     * Returns 0 if the token is not in the dictionary or has not yet been hit.
     * Thread-safe (atomic load via the per-token counter).
     *
     * @param token Raw token string.
     * @return Hit count for the token.
     */
    [[nodiscard]] uint64_t get_token_hit_count(const std::string& token) const;

    /**
     * @brief Return the top N most frequently hit dictionary tokens, sorted descending.
     *
     * Only tokens that have been looked up and found at least once will have
     * non-zero counts.  Tokens with zero hits are included in the returned
     * vector only if n is large enough to encompass the full dictionary.
     *
     * @param n Maximum number of entries to return (default: 10).
     * @return Vector of (token, hit_count) pairs sorted by hit_count descending.
     */
    [[nodiscard]] std::vector<std::pair<std::string, uint64_t>> top_tokens_by_frequency(size_t n = 10) const;

    /**
     * @brief Reset all per-token hit counters to zero without affecting statistics.
     *
     * Thread-safe: each counter is reset atomically.
     */
    void reset_frequency_counts();

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

    /**
     * @brief Return the fraction of token lookups served from the cache, in [0.0, 1.0].
     *
     * Returns 0.0 if no tokens have been processed yet.
     * Thread-safe: reads atomic counters with relaxed ordering.
     *
     * @return Cache hit rate in [0.0, 1.0].
     */
    double get_cache_hit_rate() const noexcept {
        uint64_t processed = stats_.tokens_processed.load(std::memory_order_relaxed);
        if (processed == 0) return 0.0;
        return static_cast<double>(stats_.cache_hits.load(std::memory_order_relaxed))
             / static_cast<double>(processed);
    }

    /**
     * @brief Return a single-line human-readable summary of processing statistics.
     *
     * Format: "tokens=<n> hits=<n> misses=<n> hit_rate=<rate> dict_size=<n>"
     * If no tokens have been processed, returns "tokens=0 (no data)".
     *
     * Thread-safe (reads atomic counters with relaxed ordering).
     *
     * @return Single-line stats summary string.
     */
    [[nodiscard]] std::string format_stats() const;

    /**
     * @brief Serialise the current adapter statistics to a JSON string.
     *
     * Produces a JSON object with tokens_processed, cache_hits, cache_misses,
     * hit_rate_pct (0–100), and dictionary_size.
     * Thread-safe (reads atomic counters with relaxed ordering).
     *
     * @return JSON object as std::string.
     */
    std::string to_stats_json() const noexcept {
        uint64_t processed = stats_.tokens_processed.load(std::memory_order_relaxed);
        uint64_t hits      = stats_.cache_hits.load(std::memory_order_relaxed);
        uint64_t misses    = processed - hits;
        double   hit_pct   = (processed > 0) ? (static_cast<double>(hits) * 100.0 / static_cast<double>(processed)) : 0.0;
        size_t   dict_size = get_dictionary_size();
        char buf[256];
        std::snprintf(buf, sizeof(buf),
            "{\"tokens_processed\":%" PRIu64
            ",\"cache_hits\":%" PRIu64
            ",\"cache_misses\":%" PRIu64
            ",\"hit_rate_pct\":%.2f"
            ",\"dictionary_size\":%zu}",
            processed, hits, misses, hit_pct, dict_size);
        return buf;
    }

    /**
     * @brief Export the entire token dictionary as a tab-separated string.
     *
     * Each line represents one token in the format:
     *   token\tsentiment\tconfidence\tvolatility\tdirectional_bias
     *
     * Values are formatted to 6 decimal places.  Lines are sorted
     * alphabetically by token for deterministic output.
     *
     * @return Multi-line TSV string; empty string if the dictionary is empty.
     */
    /**
     * @brief Return the minimum sentiment_score across all dictionary tokens.
     *
     * Returns 0.0 if the dictionary is empty.
     *
     * @return Minimum sentiment_score.
     */
    double get_min_sentiment() const;

    /**
     * @brief Return the maximum sentiment_score across all dictionary tokens.
     *
     * Returns 0.0 if the dictionary is empty.
     *
     * @return Maximum sentiment_score.
     */
    double get_max_sentiment() const;

    /**
     * @brief Return the spread between the most bullish and most bearish token sentiments.
     *
     * Computed as get_max_sentiment() - get_min_sentiment().
     * Returns 0.0 if the dictionary is empty.
     *
     * @return Sentiment range (always >= 0).
     */
    double get_sentiment_range() const;

    /**
     * @brief Return the minimum volatility_score across all dictionary tokens.
     *
     * Returns 0.0 if the dictionary is empty.
     *
     * @return Minimum volatility_score in [0.0, 1.0].
     */
    double get_min_volatility() const;

    /**
     * @brief Return the maximum volatility_score across all dictionary tokens.
     *
     * Returns 0.0 if the dictionary is empty.
     *
     * @return Maximum volatility_score in [0.0, 1.0].
     */
    double get_max_volatility() const;

    /**
     * @brief Return the minimum directional_bias across all dictionary tokens.
     *
     * Returns 0.0 if the dictionary is empty.
     *
     * @return Minimum directional_bias in [-1.0, 1.0].
     */
    double get_min_directional_bias() const;

    /**
     * @brief Return the maximum directional_bias across all dictionary tokens.
     *
     * Returns 0.0 if the dictionary is empty.
     *
     * @return Maximum directional_bias in [-1.0, 1.0].
     */
    double get_max_directional_bias() const;

    /**
     * @brief Return the spread between the highest and lowest volatility_score in the dictionary.
     *
     * Computed as get_max_volatility() - get_min_volatility().
     * Returns 0.0 if the dictionary is empty.
     *
     * @return Volatility range (always >= 0).
     */
    double get_volatility_range() const;

    /**
     * @brief Return the spread between the most bullish and most bearish directional_bias
     *        in the dictionary.
     *
     * Computed as get_max_directional_bias() - get_min_directional_bias().
     * Returns 0.0 if the dictionary is empty.
     *
     * @return Directional bias range (always >= 0).
     */
    double get_directional_bias_range() const;

    std::string export_dictionary() const;

    /**
     * @brief Import token mappings from a TSV string produced by export_dictionary().
     *
     * Parses each line of the format:
     *   token\tsentiment\tconfidence\tvolatility\tdirectional_bias
     *
     * Existing mappings are replaced for tokens already in the dictionary;
     * new tokens are added.  Lines that cannot be parsed (wrong field count
     * or non-numeric values) are silently skipped.
     *
     * @param tsv_data Multi-line TSV string (as returned by export_dictionary()).
     * @return Number of token entries successfully imported.
     */
    size_t load_dictionary_from_tsv(const std::string& tsv_data);

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

    /**
     * @brief Per-token hit counters; populated in parallel with token_weights_.
     *
     * Each entry is a unique_ptr so the pointed-to atomic is stable (no
     * rehash relocation) and can be incremented without a mutex in
     * map_token_to_weight().  Entries are added whenever token_weights_ is
     * updated and are never removed while the adapter is running.
     */
    mutable std::unordered_map<std::string, std::unique_ptr<std::atomic<uint64_t>>> token_hit_counts_;

    /** @brief Internal statistics; mutable so const query methods can update them. */
    mutable struct {
        std::atomic<uint64_t> tokens_processed{0};
        std::atomic<uint64_t> cache_hits{0};
        std::atomic<uint64_t> cache_misses{0};
    } stats_;
};

} // namespace llmquant
