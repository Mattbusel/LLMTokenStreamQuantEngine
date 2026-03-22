#pragma once

#include <cstdint>
#include <functional>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

namespace llmquant {

/**
 * @brief Tracks each token's cumulative contribution to the signal bias.
 *
 * In a token-stream engine the mapping from token text to semantic weight is
 * static (from the LLM adapter's dictionary), but the *impact* of a token
 * depends on how often it appears and when.  TokenBiasHeatmap accumulates
 * per-token statistics so operators can answer questions like:
 *
 *   - Which tokens are driving the most bullish/bearish signal energy?
 *   - Is the bias dominated by one recurring token, indicating a potential
 *     adversarial injection or data quality issue?
 *
 * ## Usage
 * ```cpp
 * TokenBiasHeatmap heatmap;
 * // In process_token lambda, after weight lookup:
 * heatmap.record(token_text, weight.directional_bias);
 * // Query:
 * auto top5 = heatmap.top_by_abs_contribution(5);
 * ```
 *
 * ## Thread safety
 * All methods are thread-safe.
 *
 * ## Feature flag
 * Controlled by `LLMQUANT_ENABLE_TOKEN_BIAS_HEATMAP` (default ON).
 */
class TokenBiasHeatmap {
public:
    struct TokenStats {
        std::string token;
        double      total_bias{0.0};       ///< Algebraic sum of all bias deltas.
        double      abs_contribution{0.0}; ///< Sum of |bias| over all appearances.
        uint64_t    count{0};              ///< How many times token was recorded.
        double      mean_bias{0.0};        ///< total_bias / count.
    };

    struct Config {
        /**
         * @brief Maximum distinct tokens to track.  Oldest entry (by first-seen
         * insertion order) is evicted when capacity is exceeded.
         * Default 4096.
         */
        uint32_t max_tokens{4096};
    };

    explicit TokenBiasHeatmap(Config cfg = Config{}) : cfg_(cfg) {}

    // Non-copyable.
    TokenBiasHeatmap(const TokenBiasHeatmap&)            = delete;
    TokenBiasHeatmap& operator=(const TokenBiasHeatmap&) = delete;

    /**
     * @brief Record a token's bias contribution.
     *
     * @param token Text of the token.
     * @param bias  Signed bias delta from this token occurrence.
     */
    void record(const std::string& token, double bias);

    /**
     * @brief Return top-N tokens sorted descending by |total_contribution|.
     *
     * Thread-safe snapshot copy.
     */
    [[nodiscard]] std::vector<TokenStats>
    top_by_abs_contribution(uint32_t n) const;

    /**
     * @brief Return top-N tokens sorted descending by algebraic total_bias
     *        (most bullish first).
     */
    [[nodiscard]] std::vector<TokenStats>
    top_bullish(uint32_t n) const;

    /**
     * @brief Return top-N tokens sorted ascending by algebraic total_bias
     *        (most bearish first).
     */
    [[nodiscard]] std::vector<TokenStats>
    top_bearish(uint32_t n) const;

    /**
     * @brief Total number of record() calls.
     */
    [[nodiscard]] uint64_t total_records() const noexcept {
        return total_records_.load(std::memory_order_relaxed);
    }

    /**
     * @brief Current number of distinct tokens tracked.
     */
    [[nodiscard]] uint32_t distinct_tokens() const;

    /**
     * @brief Clear all accumulated state.
     */
    void reset();

    /**
     * @brief Serialise top-10 tokens to a JSON array string.
     */
    [[nodiscard]] std::string to_stats_json() const;

private:
    struct Entry {
        double   total_bias{0.0};
        double   abs_contribution{0.0};
        uint64_t count{0};
        uint64_t insert_order{0};  // for eviction of oldest
    };

    Config cfg_;
    mutable std::mutex mtx_;
    std::unordered_map<std::string, Entry> map_;
    uint64_t insert_counter_{0};
    std::atomic<uint64_t> total_records_{0};

    // Build a sorted snapshot; caller holds no lock (uses lock internally).
    [[nodiscard]] std::vector<TokenStats> snapshot_sorted(
        bool by_abs, bool ascending) const;
};

} // namespace llmquant
