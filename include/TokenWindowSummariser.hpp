#pragma once

#include "LLMAdapter.h"

#include <cstddef>
#include <deque>
#include <mutex>
#include <string>

namespace llmquant {

/**
 * @brief Rolling window over SemanticWeight entries that emits a summary
 *        signal when the window reaches capacity.
 *
 * Tokens are pushed in one at a time.  Each push ages every existing entry by
 * multiplying its contribution by `decay` (0 < decay ≤ 1), then appends the
 * new weight.  When the deque reaches `window_size` entries the window is
 * considered "ready" and `summarise()` returns a populated `WindowSummary`.
 *
 * ## Design intent
 * Short-horizon signals (e.g. last 20 tokens) often carry more actionable
 * information than a raw accumulated bias.  `TokenWindowSummariser` gives the
 * pipeline a lightweight way to track recency-weighted directional bias,
 * average confidence, and second-moment volatility over a fixed look-back.
 *
 * ## Feature flag
 * Controlled by `LLMQUANT_ENABLE_TOKEN_WINDOW_SUMMARISER` (default ON).
 *
 * Thread safety: all public methods are protected by an internal mutex.
 */
class TokenWindowSummariser {
public:
    // -----------------------------------------------------------------------
    // Summary result
    // -----------------------------------------------------------------------

    /**
     * @brief Snapshot summary produced by `summarise()`.
     *
     * @param net_bias        Recency-weighted mean directional bias (signed,
     *                        positive = bullish, negative = bearish).
     * @param confidence      Mean confidence score over the current window.
     * @param volatility_est  Unbiased standard deviation of directional_bias
     *                        values in the window — a proxy for near-term
     *                        signal uncertainty.
     * @param token_count     Number of entries currently in the window.
     */
    struct WindowSummary {
        float  net_bias       {0.0f};
        float  confidence     {0.0f};
        float  volatility_est {0.0f};
        size_t token_count    {0};
    };

    // -----------------------------------------------------------------------
    // Construction
    // -----------------------------------------------------------------------

    /**
     * @brief Construct a summariser.
     *
     * @param window_size Capacity of the rolling window (≥ 1).
     * @param decay       Per-push age decay factor applied to existing entries
     *                    before the new entry is appended.  Range (0, 1].
     *                    1.0 = no decay (uniform window).
     */
    explicit TokenWindowSummariser(size_t window_size = 20,
                                   float  decay        = 0.95f);

    // Non-copyable; move is fine.
    TokenWindowSummariser(const TokenWindowSummariser&)            = delete;
    TokenWindowSummariser& operator=(const TokenWindowSummariser&) = delete;

    // -----------------------------------------------------------------------
    // Data intake
    // -----------------------------------------------------------------------

    /**
     * @brief Push a new SemanticWeight into the window.
     *
     * All existing entries are multiplied by `decay_` before the new entry is
     * appended.  If the deque already contains `window_size_` entries the
     * oldest entry is evicted after the decay pass.
     */
    void push(const SemanticWeight& w);

    // -----------------------------------------------------------------------
    // Summary
    // -----------------------------------------------------------------------

    /**
     * @brief Compute and return the current window summary.
     *
     * Safe to call at any time.  If the window is empty all fields are zero.
     * The volatility_est field is the sample standard deviation (Bessel-
     * corrected) of directional_bias values; it is 0 when fewer than two
     * entries are present.
     */
    [[nodiscard]] WindowSummary summarise() const;

    // -----------------------------------------------------------------------
    // Accessors
    // -----------------------------------------------------------------------

    /**
     * @brief Returns true when the window contains at least `window_size_` entries.
     */
    [[nodiscard]] bool is_ready() const noexcept;

    /**
     * @brief Number of entries currently in the window (≤ window_size).
     */
    [[nodiscard]] size_t size() const noexcept;

    /**
     * @brief Configured capacity.
     */
    [[nodiscard]] size_t window_size() const noexcept { return window_size_; }

    /**
     * @brief Configured decay factor.
     */
    [[nodiscard]] float decay() const noexcept { return decay_; }

    // -----------------------------------------------------------------------
    // Control
    // -----------------------------------------------------------------------

    /**
     * @brief Clear all entries.  is_ready() will return false until the window
     *        fills again.
     */
    void reset();

    /**
     * @brief JSON diagnostics snapshot.
     */
    [[nodiscard]] std::string to_stats_json() const;

private:
    // Each stored entry is the decayed directional_bias, the raw
    // confidence_score, and the raw volatility_score at the time of push.
    struct Entry {
        float bias;
        float confidence;
        float volatility;
    };

    size_t window_size_;
    float  decay_;

    // Entries stored oldest-first; new entries are pushed to the back.
    std::deque<Entry> window_;

    mutable std::mutex mutex_;
};

} // namespace llmquant
