#pragma once

#include <cstdint>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

namespace llmquant {

/**
 * @brief Trie-based token sequence pattern matcher with signal multipliers.
 *
 * Maintains a trie of known multi-token "phrase patterns", each associated
 * with a signal multiplier. As tokens arrive from the LLM stream, the matcher
 * advances a set of active cursors through the trie. When a cursor reaches a
 * terminal node, it fires the associated multiplier.
 *
 * ## Example patterns
 * | Pattern tokens                   | Multiplier | Rationale                         |
 * |----------------------------------|------------|-----------------------------------|
 * | ["better", "than", "expected"]   |  +3.0      | Strong positive earnings beat     |
 * | ["guidance", "lowered"]          |  -5.0      | Forward guidance cut              |
 * | ["ahead", "of", "expectations"]  |  +2.5      | Beat consensus                    |
 * | ["missed", "estimates"]          |  -4.0      | Earnings miss                     |
 * | ["raised", "guidance"]           |  +4.0      | Bullish forward guidance          |
 * | ["disappointing", "results"]     |  -3.5      | Vague negative framing            |
 *
 * ## Design
 * - The trie stores token strings (lowercased) at each edge.
 * - Multiple patterns can be active simultaneously (cursor set).
 * - A new cursor is spawned at the root for every incoming token (allowing
 *   patterns to match starting at any position in the stream).
 * - Cursors that fail to match are pruned immediately.
 * - When a pattern completes, its multiplier is returned from `feed()`.
 *
 * ## Thread safety
 * All methods are thread-safe.
 *
 * ## Feature flag
 * Controlled by `LLMQUANT_ENABLE_PATTERN_MATCHER` (default ON).
 */
class PatternMatcher {
public:
    /** @brief A single pattern entry: token sequence + multiplier. */
    struct Pattern {
        /** @brief Ordered sequence of tokens (lowercased) forming the phrase. */
        std::vector<std::string> tokens;
        /**
         * @brief Signal multiplier applied when this pattern is matched.
         * Positive values amplify bullish signals; negative values amplify
         * bearish signals.
         */
        double multiplier{1.0};
        /** @brief Optional human-readable label for logging. */
        std::string label;
    };

    /**
     * @brief Configuration.
     */
    struct Config {
        /**
         * @brief Maximum number of simultaneously active cursors.
         *
         * Bounds memory usage when the stream contains many potential pattern
         * prefixes. When the limit is exceeded, oldest cursors are pruned.
         * Default: 256.
         */
        size_t max_active_cursors{256};

        /**
         * @brief Lowercase incoming tokens before matching.
         *
         * Recommended; LLM tokenisers produce mixed-case output.
         * Default: true.
         */
        bool lowercase_tokens{true};
    };

    explicit PatternMatcher(Config cfg = Config{});

    // Non-copyable.
    PatternMatcher(const PatternMatcher&)            = delete;
    PatternMatcher& operator=(const PatternMatcher&) = delete;

    /**
     * @brief Add a phrase pattern with an associated signal multiplier.
     *
     * Thread-safe.
     *
     * @param pattern  Pattern definition (tokens + multiplier + label).
     */
    void add_pattern(const Pattern& pattern);

    /**
     * @brief Remove all patterns whose label equals `label`.
     *
     * Thread-safe.
     *
     * @param label  The label string to match (exact).
     * @return Number of patterns removed.
     */
    size_t remove_pattern_by_label(const std::string& label);

    /**
     * @brief Feed the next token from the LLM stream.
     *
     * Advances all active cursors. Returns any multipliers from patterns that
     * completed on this token. An empty vector means no pattern matched.
     *
     * Thread-safe.
     *
     * @param token  The next token string (will be lowercased if configured).
     * @return Vector of (multiplier, label) pairs for all patterns that fired.
     */
    [[nodiscard]] std::vector<std::pair<double, std::string>> feed(const std::string& token);

    /**
     * @brief Reset all active cursors (but keep patterns).
     *
     * Thread-safe.
     */
    void reset_cursors();

    /**
     * @brief Remove all patterns and reset cursors.
     *
     * Thread-safe.
     */
    void clear();

    /**
     * @brief Number of registered patterns.
     *
     * Thread-safe.
     */
    [[nodiscard]] size_t pattern_count() const;

    /**
     * @brief Number of currently active cursors.
     *
     * Thread-safe.
     */
    [[nodiscard]] size_t active_cursor_count() const;

    /**
     * @brief Total pattern matches since construction or last clear().
     *
     * Thread-safe.
     */
    [[nodiscard]] uint64_t total_matches() const noexcept;

    /**
     * @brief Total tokens fed since construction or last clear().
     *
     * Thread-safe.
     */
    [[nodiscard]] uint64_t total_tokens_fed() const noexcept;

    /**
     * @brief Serialise state to a JSON object string.
     */
    [[nodiscard]] std::string to_stats_json() const;

private:
    // ----------------------------------------------------------------
    // Trie implementation
    // ----------------------------------------------------------------

    struct TrieNode {
        std::unordered_map<std::string, size_t> children;  // token → child index
        bool is_terminal{false};
        double multiplier{1.0};
        std::string label;
    };

    // ----------------------------------------------------------------
    // Cursor: tracks one in-progress pattern match.
    // ----------------------------------------------------------------

    struct Cursor {
        size_t node_index{0};  // current position in trie
        size_t depth{0};       // how many tokens have matched so far
    };

    // ----------------------------------------------------------------
    // Fields
    // ----------------------------------------------------------------

    mutable std::mutex mtx_;
    Config cfg_;

    std::vector<TrieNode> trie_;   // node 0 = root
    std::vector<Cursor>   cursors_;

    uint64_t total_matches_{0};
    uint64_t total_tokens_fed_{0};

    // ----------------------------------------------------------------
    // Helpers
    // ----------------------------------------------------------------

    void insert_pattern_locked(const Pattern& pattern);
    [[nodiscard]] static std::string lowercase(const std::string& s);
};

} // namespace llmquant
