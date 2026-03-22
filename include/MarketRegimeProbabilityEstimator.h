#pragma once

#include <array>
#include <atomic>
#include <cstdint>
#include <functional>
#include <mutex>
#include <string>

namespace llmquant {

/**
 * @brief Online two-state Hidden Markov Model for market regime estimation.
 *
 * Maintains a posterior probability distribution over two latent regimes:
 *   - State 0: Risk-OFF  (bearish sentiment, elevated volatility)
 *   - State 1: Risk-ON   (bullish sentiment, suppressed volatility)
 *
 * On each observation (sentiment score + volatility), the filter performs a
 * Bayes update:
 *
 *   1. Predict: π_t = A^T · π_{t-1}          (transition step)
 *   2. Update:  π_t ∝ B(obs_t) ⊙ π_t         (emission likelihood)
 *   3. Normalise to ensure π_t[0] + π_t[1] = 1
 *
 * Emission likelihoods use Gaussian densities whose means and variances are
 * estimated online via exponential moving averages.
 *
 * ## Interpretation
 * - `prob_risk_on()` ∈ [0,1]: probability the market is in the risk-on state.
 * - When this crosses `transition_threshold` upward, `on_regime_change` fires
 *   with `is_risk_on = true`.
 *
 * ## Feature flag
 * Controlled by `LLMQUANT_ENABLE_REGIME_PROB` (default ON).
 *
 * Thread safety: all public methods are thread-safe.
 */
class MarketRegimeProbabilityEstimator {
public:
    // Number of hidden states (compile-time constant).
    static constexpr int kStates = 2;

    // -----------------------------------------------------------------------
    // Configuration
    // -----------------------------------------------------------------------

    struct Config {
        /**
         * @brief EMA alpha for online emission parameter estimation.
         *
         * Smaller = slower adaptation (more stable estimates).  Default 0.05.
         */
        double emission_alpha{0.05};

        /**
         * @brief State transition probability (stay → switch).
         *
         * P(switch state) = transition_prob per observation.  Default 0.02.
         */
        double transition_prob{0.02};

        /**
         * @brief Minimum emission variance (prevents divide-by-zero).
         *
         * Default 1e-4.
         */
        double min_variance{1e-4};

        /**
         * @brief Probability threshold for declaring a regime change.
         *
         * `on_regime_change` fires when `prob_risk_on` crosses this threshold.
         * Default 0.7 (upward) / 0.3 (downward).
         */
        double transition_threshold{0.7};

        /**
         * @brief Minimum observations before regime transitions are reported.
         */
        int min_observations{20};

        /**
         * @brief Fired when the dominant regime changes.
         *
         * Parameters: (prob_risk_on, is_risk_on).
         * Called outside the internal lock.
         */
        std::function<void(double prob_risk_on, bool is_risk_on)> on_regime_change;
    };

    // -----------------------------------------------------------------------
    // Construction
    // -----------------------------------------------------------------------

    explicit MarketRegimeProbabilityEstimator(Config cfg = Config{});

    // -----------------------------------------------------------------------
    // Observation intake
    // -----------------------------------------------------------------------

    /**
     * @brief Feed one observation into the filter.
     *
     * @param sentiment Normalised sentiment score ∈ [-1, 1].
     * @param volatility Non-negative volatility estimate (e.g. realised vol).
     */
    void update(double sentiment, double volatility) noexcept;

    // -----------------------------------------------------------------------
    // Accessors
    // -----------------------------------------------------------------------

    /**
     * @brief Posterior probability of the risk-ON regime ∈ [0, 1].
     */
    [[nodiscard]] double prob_risk_on()  const noexcept;

    /**
     * @brief Posterior probability of the risk-OFF regime ∈ [0, 1].
     */
    [[nodiscard]] double prob_risk_off() const noexcept;

    /**
     * @brief True when the dominant regime is currently risk-ON.
     *
     * Defined as prob_risk_on() > transition_threshold.
     */
    [[nodiscard]] bool is_risk_on() const noexcept;

    /**
     * @brief Number of observations fed so far.
     */
    [[nodiscard]] uint64_t observation_count() const noexcept {
        return obs_count_.load(std::memory_order_relaxed);
    }

    /**
     * @brief Total regime transition events fired since last reset.
     */
    [[nodiscard]] uint64_t transition_count() const noexcept {
        return transition_count_.load(std::memory_order_relaxed);
    }

    // -----------------------------------------------------------------------
    // Control
    // -----------------------------------------------------------------------

    /**
     * @brief Reset filter to uniform prior; retain emission parameters.
     */
    void reset();

    /**
     * @brief Update configuration without resetting state.
     */
    void update_config(const Config& cfg);

    /**
     * @brief JSON diagnostics snapshot.
     */
    [[nodiscard]] std::string to_stats_json() const;

private:
    Config cfg_;

    // Filter state (protected by mutex_).
    std::array<double, kStates> pi_{};         // posterior π[0]=OFF, π[1]=ON
    std::array<double, kStates> mu_sent_{};    // emission mean: sentiment
    std::array<double, kStates> var_sent_{};   // emission variance: sentiment
    std::array<double, kStates> mu_vol_{};     // emission mean: volatility
    std::array<double, kStates> var_vol_{};    // emission variance: volatility
    bool prev_risk_on_{false};

    std::atomic<uint64_t> obs_count_{0};
    std::atomic<uint64_t> transition_count_{0};
    std::atomic<double>   prob_risk_on_{0.5};

    mutable std::mutex mutex_;

    // Gaussian log-likelihood (numerically safe).
    [[nodiscard]] static double gaussian_pdf(double x, double mu, double var,
                                              double min_var) noexcept;
};

} // namespace llmquant
