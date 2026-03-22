#pragma once

#include <cstdint>
#include <functional>
#include <mutex>
#include <string>
#include <vector>

namespace llmquant {

/**
 * @brief Shapley-inspired attribution of token contributions to trade signals.
 *
 * As tokens arrive they are pushed into a fixed-size rolling window alongside
 * the signed weight each token applied to the current bias.  When
 * `attribute()` is called (e.g. at order generation time), the attributor
 * computes each token's *marginal contribution* — the change in the cumulative
 * bias that is explained by that token relative to the population mean.
 *
 * This is a lightweight linear-attribution model (not full Shapley), but it is
 * O(N) and exact for any linearly-additive signal pipeline:
 *
 *   influence_i = weight_i − mean_weight
 *
 * Results are sorted by |influence| descending so the most influential tokens
 * appear first.  Operators can use this to audit individual decisions, detect
 * model-gaming attempts, and verify that no single token dominates a signal.
 *
 * ## Feature flag
 * Controlled by `LLMQUANT_ENABLE_TOKEN_INFLUENCE` (default ON).
 *
 * Thread safety: all public methods are thread-safe.
 */
class TokenInfluenceAttributor {
public:
    // -----------------------------------------------------------------------
    // Data types
    // -----------------------------------------------------------------------

    struct TokenRecord {
        std::string token;
        double      weight;     ///< Signed weight applied by the LLM adapter.
        uint64_t    sequence;   ///< Monotonic arrival sequence number.
    };

    struct AttributionResult {
        std::string token;
        double      weight;         ///< Raw weight applied.
        double      influence;      ///< weight − mean_weight (marginal contribution).
        double      abs_influence;  ///< |influence|
        double      cumulative_fraction; ///< Fraction of |total influence| explained.
        uint64_t    sequence;
    };

    // -----------------------------------------------------------------------
    // Configuration
    // -----------------------------------------------------------------------

    struct Config {
        /**
         * @brief Rolling window size (number of tokens retained for attribution).
         *
         * Older tokens are evicted as new ones arrive.  Default 64.
         */
        std::size_t window_size{64};

        /**
         * @brief Maximum number of tokens returned by attribute().
         *
         * Default 10.
         */
        std::size_t top_k{10};

        /**
         * @brief Fired when |influence| of a single token exceeds this fraction
         * of total absolute influence.  Signals a potential dominant-token event.
         *
         * Set to 1.0 to disable.  Default 0.5.
         */
        double dominant_token_fraction{0.5};

        /**
         * @brief Called when a dominant token is detected.
         *
         * Parameters: (token, influence_fraction).
         * Called outside the internal lock.
         */
        std::function<void(const std::string& token, double fraction)>
            on_dominant_token;
    };

    // -----------------------------------------------------------------------
    // Construction
    // -----------------------------------------------------------------------

    explicit TokenInfluenceAttributor(Config cfg = Config{});

    // -----------------------------------------------------------------------
    // Token intake
    // -----------------------------------------------------------------------

    /**
     * @brief Record a token and its signed weight in the rolling window.
     *
     * @param token  Raw token string.
     * @param weight Signed weight applied by the LLM adapter to bias.
     */
    void record(const std::string& token, double weight);

    // -----------------------------------------------------------------------
    // Attribution
    // -----------------------------------------------------------------------

    /**
     * @brief Compute attribution for the current window.
     *
     * Returns up to `top_k` tokens sorted by |influence| descending.
     * `cumulative_fraction` is filled relative to the total |influence| of
     * the window so callers can draw Pareto / waterfall charts.
     */
    [[nodiscard]] std::vector<AttributionResult> attribute() const;

    /**
     * @brief Compute attribution and call `visitor` for each result.
     *
     * Avoids heap allocation when the caller processes results inline.
     */
    void attribute(std::function<void(const AttributionResult&)> visitor) const;

    // -----------------------------------------------------------------------
    // Accessors
    // -----------------------------------------------------------------------

    /**
     * @brief Number of tokens currently in the window.
     */
    [[nodiscard]] std::size_t window_size() const noexcept;

    /**
     * @brief Total tokens recorded since construction or last reset.
     */
    [[nodiscard]] uint64_t total_recorded() const noexcept {
        return total_recorded_.load(std::memory_order_relaxed);
    }

    /**
     * @brief Total dominant-token events fired.
     */
    [[nodiscard]] uint64_t dominant_events() const noexcept {
        return dominant_events_.load(std::memory_order_relaxed);
    }

    // -----------------------------------------------------------------------
    // Control
    // -----------------------------------------------------------------------

    /**
     * @brief Clear the rolling window; retain configuration.
     */
    void reset();

    /**
     * @brief Update configuration without clearing the window.
     */
    void update_config(const Config& cfg);

    /**
     * @brief JSON diagnostics (top-k attribution of the current window).
     */
    [[nodiscard]] std::string to_stats_json() const;

private:
    Config cfg_;

    std::vector<TokenRecord> window_;  // circular buffer
    std::size_t head_{0};              // next write position
    std::size_t count_{0};             // tokens in buffer
    uint64_t    seq_{0};               // arrival counter

    std::atomic<uint64_t> total_recorded_{0};
    std::atomic<uint64_t> dominant_events_{0};

    mutable std::mutex mutex_;

    // Internal: compute results inside the lock.
    [[nodiscard]] std::vector<AttributionResult> compute_attribution_locked() const;
};

} // namespace llmquant
