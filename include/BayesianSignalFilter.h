#pragma once

#include <atomic>
#include <cstdint>
#include <functional>
#include <mutex>
#include <string>

namespace llmquant {

/**
 * @brief Bayesian posterior confidence filter for trade signals.
 *
 * Maintains per-direction (bullish / bearish) prior and likelihood estimates
 * and updates them with each signal outcome.  The resulting posterior
 * probability represents the current belief that the active signal direction
 * is correct, conditioned on the historical hit rate and the current regime.
 *
 * ## Model
 * Let P(correct) be the prior probability that a signal in the current
 * direction is correct.  After each resolved signal (passed + eventual
 * profit/loss outcome), the filter updates:
 *
 *   P(correct | hit) ∝ P(correct) * P(hit | correct)
 *   P(correct | miss) ∝ P(correct) * P(miss | correct)
 *
 * Using a Beta-Bernoulli conjugate: Beta(α, β) where
 *   α = prior_alpha + hits, β = prior_beta + misses
 * The posterior mean = α / (α + β).
 *
 * ## Usage
 * @code
 * BayesianSignalFilter filter;
 * // After risk evaluation:
 * if (passed) {
 *     // Record that a signal was emitted.
 *     filter.record_signal(bias > 0);
 * }
 * // After trade outcome:
 * filter.record_outcome(bias > 0, profit > 0);
 * // Gate on posterior confidence:
 * if (filter.posterior_confidence(bias > 0) < min_confidence) skip;
 * @endcode
 *
 * ## Feature flag
 * Controlled by `LLMQUANT_ENABLE_BAYESIAN_FILTER` (default ON).
 */
class BayesianSignalFilter {
public:
    /**
     * @brief Configuration.
     */
    struct Config {
        /**
         * @brief Beta prior alpha (pseudo-hits before any observations).
         *
         * Higher alpha → more optimistic prior.  Default 2.0 (weakly informative).
         */
        double prior_alpha{2.0};

        /**
         * @brief Beta prior beta (pseudo-misses before any observations).
         *
         * Default 2.0 (50/50 uninformative prior).
         */
        double prior_beta{2.0};

        /**
         * @brief Learning rate for exponential forgetting of old outcomes.
         *
         * At each update: alpha *= forgetting; beta *= forgetting.
         * 1.0 = no forgetting (accumulate all history).
         * 0.99 = gentle forgetting, adapts to regime changes.
         * Default 0.99.
         */
        double forgetting{0.99};

        /**
         * @brief Minimum posterior confidence to emit a callback.
         *
         * When posterior drops below this threshold, on_low_confidence fires.
         * Default 0.4.
         */
        double low_confidence_threshold{0.4};

        /**
         * @brief Callback when posterior confidence drops below threshold.
         *
         * Parameters: (is_bullish, posterior_confidence).
         */
        std::function<void(bool, double)> on_low_confidence;
    };

    // -----------------------------------------------------------------------
    // Construction
    // -----------------------------------------------------------------------

    explicit BayesianSignalFilter(Config config = Config{});

    // -----------------------------------------------------------------------
    // Update
    // -----------------------------------------------------------------------

    /**
     * @brief Record that a signal was emitted in the given direction.
     *
     * Must be paired with a subsequent record_outcome() call.
     * Thread-safe.
     *
     * @param bullish  true = bullish signal, false = bearish.
     */
    void record_signal(bool bullish);

    /**
     * @brief Record the outcome (win/loss) for the last signal in this direction.
     *
     * Updates the Beta posterior.  If forgetting < 1.0, the existing
     * alpha/beta counts are decayed before adding the new observation.
     * Thread-safe.
     *
     * @param bullish  Direction of the resolved signal.
     * @param win      true = profitable outcome, false = loss.
     */
    void record_outcome(bool bullish, bool win);

    // -----------------------------------------------------------------------
    // Query
    // -----------------------------------------------------------------------

    /**
     * @brief Posterior mean confidence E[P(correct)] = α / (α + β).
     *
     * @param bullish  Which direction to query.
     * @return Value in (0, 1).
     */
    [[nodiscard]] double posterior_confidence(bool bullish) const noexcept;

    /**
     * @brief 95% credible interval half-width (±delta around the posterior mean).
     *
     * Approximated as 1.96 * sqrt(α*β / ((α+β)²*(α+β+1))).
     */
    [[nodiscard]] double credible_interval_half_width(bool bullish) const noexcept;

    /**
     * @brief Posterior precision: number of effective observations = α + β.
     *
     * Higher precision → tighter credible interval.
     */
    [[nodiscard]] double posterior_precision(bool bullish) const noexcept;

    // -----------------------------------------------------------------------
    // Statistics
    // -----------------------------------------------------------------------

    /** @brief Total signals recorded (both directions). */
    [[nodiscard]] uint64_t total_signals() const noexcept {
        return total_signals_.load(std::memory_order_relaxed);
    }

    /** @brief Total outcomes recorded (both directions). */
    [[nodiscard]] uint64_t total_outcomes() const noexcept {
        return total_outcomes_.load(std::memory_order_relaxed);
    }

    /** @brief Times the low-confidence callback fired. */
    [[nodiscard]] uint64_t low_confidence_events() const noexcept {
        return low_confidence_events_.load(std::memory_order_relaxed);
    }

    /**
     * @brief Reset all posterior state to the configured prior.
     *
     * Thread-safe.
     */
    void reset();

    /**
     * @brief Update configuration; resets posteriors to the new prior.
     *
     * Thread-safe.
     */
    void update_config(const Config& config);

    /**
     * @brief Serialise current state to a JSON object string.
     */
    [[nodiscard]] std::string to_stats_json() const;

private:
    mutable std::mutex mutex_;
    Config config_;

    // Per-direction Beta(alpha, beta) posteriors.
    double bull_alpha_{2.0}, bull_beta_{2.0};
    double bear_alpha_{2.0}, bear_beta_{2.0};

    std::atomic<uint64_t> total_signals_{0};
    std::atomic<uint64_t> total_outcomes_{0};
    std::atomic<uint64_t> low_confidence_events_{0};

    void update_posterior(double& alpha, double& beta, bool win);
};

} // namespace llmquant
