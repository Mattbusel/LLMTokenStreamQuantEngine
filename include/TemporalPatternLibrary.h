#pragma once

#include <atomic>
#include <cstdint>
#include <functional>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

namespace llmquant {

/**
 * @brief Trie-based multi-token pattern matcher for pre-computed sentiment boosts.
 *
 * Individual token weights (as in LLMAdapter) miss the compound meaning of
 * multi-token sequences.  For example:
 *  - "beat"        → mild positive signal
 *  - "earnings"    → neutral
 *  - "earnings beat" → strong positive signal (much more than the sum of parts)
 *
 * This library stores a dictionary of token sequences mapped to sentiment
 * boosts.  A sliding window automaton advances through the trie on each
 * `push_token()` call.  When a registered sequence is fully matched, the
 * corresponding boost is fired via callback and `pop_match()` returns it.
 *
 * ## Complexity
 * - push_token(): O(max_pattern_length) per token — effectively O(1) in practice.
 * - Memory: O(total tokens across all patterns × pointer overhead).
 *
 * ## Thread safety
 * - `register_pattern()` and `push_token()` are both mutex-protected.
 * - The library is designed for single-threaded hot-path use; the mutex
 *   overhead is negligible compared to LLM inference.
 *
 * ## Feature flag
 * Controlled by `LLMQUANT_ENABLE_TEMPORAL_PATTERN` (default ON).
 */
class TemporalPatternLibrary {
public:
    /**
     * @brief Callback invoked when a registered pattern is matched.
     *
     * Parameters: (pattern_name, boost_value).
     * Called from the thread invoking push_token().
     */
    using MatchCallback = std::function<void(const std::string& name, double boost)>;

    /**
     * @brief A pattern entry returned by pop_match().
     */
    struct Match {
        std::string name;   ///< Human-readable pattern name.
        double      boost;  ///< Sentiment boost to apply.
    };

    /**
     * @brief Configuration.
     */
    struct Config {
        /**
         * @brief Maximum sequence length tracked in the sliding window.
         *
         * Sequences longer than this are silently truncated at registration.
         * Default 8.
         */
        int max_pattern_length{8};

        /**
         * @brief Callback fired on every match (in addition to pop_match()).
         *
         * Optional.  Default null.
         */
        MatchCallback on_match;
    };

    // -----------------------------------------------------------------------
    // Construction
    // -----------------------------------------------------------------------

    explicit TemporalPatternLibrary(Config config = Config{});

    // -----------------------------------------------------------------------
    // Pattern management
    // -----------------------------------------------------------------------

    /**
     * @brief Register a token sequence → boost mapping.
     *
     * Sequences are case-sensitive.  Registering the same sequence twice
     * overwrites the previous boost.  Thread-safe.
     *
     * @param name      Human-readable label for the pattern.
     * @param tokens    Token sequence (e.g. {"earnings", "beat"}).
     * @param boost     Sentiment boost to apply when matched (can be negative).
     */
    void register_pattern(const std::string& name,
                          const std::vector<std::string>& tokens,
                          double boost);

    /**
     * @brief Remove a previously registered pattern by name.
     *
     * No-op if the pattern does not exist.  Thread-safe.
     *
     * @param name  The name used during register_pattern().
     */
    void remove_pattern(const std::string& name);

    // -----------------------------------------------------------------------
    // Streaming
    // -----------------------------------------------------------------------

    /**
     * @brief Push a new token from the LLM stream.
     *
     * Advances all active automata states.  Any pattern that reaches a
     * terminal state is recorded as a pending match.  Thread-safe.
     *
     * @param token  The incoming token string.
     * @return       true if at least one pattern matched on this token.
     */
    bool push_token(const std::string& token);

    /**
     * @brief Pop the next pending match (FIFO).
     *
     * Returns nullopt if no matches are pending.  Thread-safe.
     */
    [[nodiscard]] bool pop_match(Match& out);

    /**
     * @brief Number of matches pending in the queue.
     *
     * Thread-safe.
     */
    [[nodiscard]] int pending_matches() const noexcept;

    // -----------------------------------------------------------------------
    // Statistics
    // -----------------------------------------------------------------------

    /** @brief Total tokens processed by push_token(). */
    [[nodiscard]] uint64_t total_tokens() const noexcept {
        return total_tokens_.load(std::memory_order_relaxed);
    }

    /** @brief Total pattern matches since construction (or last reset). */
    [[nodiscard]] uint64_t total_matches() const noexcept {
        return total_matches_.load(std::memory_order_relaxed);
    }

    /** @brief Number of registered patterns. */
    [[nodiscard]] int pattern_count() const noexcept;

    /**
     * @brief Reset streaming state and pending-match queue.
     *
     * Does NOT remove registered patterns.  Thread-safe.
     */
    void reset();

    /**
     * @brief Clear all registered patterns and streaming state.
     *
     * Thread-safe.
     */
    void clear();

    /**
     * @brief Update configuration.  Does NOT clear registered patterns.
     *
     * Thread-safe.
     */
    void update_config(const Config& config);

    /**
     * @brief Serialise state to a JSON object string.
     */
    [[nodiscard]] std::string to_stats_json() const;

private:
    mutable std::mutex mutex_;
    Config config_;

    // Trie node.
    struct TrieNode {
        std::unordered_map<std::string, int> children; // token → child index
        bool   is_terminal{false};
        double boost{0.0};
        std::string name;
    };

    std::vector<TrieNode> trie_;      // node 0 is the root
    int                   next_id_{1}; // next trie node index to allocate

    // A partial match in progress: {trie node index, depth}.
    struct ActiveState {
        int node_idx{0};
        int depth{0};
    };

    std::vector<ActiveState> active_;   // partial matches
    std::vector<Match>       pending_;  // completed matches queue

    // Pattern registry for deduplication (name → leaf node index).
    std::unordered_map<std::string, int> registry_;

    std::atomic<uint64_t> total_tokens_{0};
    std::atomic<uint64_t> total_matches_{0};
};

} // namespace llmquant
