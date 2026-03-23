#pragma once

#include <atomic>
#include <cstdint>
#include <functional>
#include <mutex>
#include <string>
#include <vector>

namespace llmquant {

/**
 * @brief Hidden Markov Model (HMM) regime switcher for signal weight tables.
 *
 * Maintains a 2-state HMM (risk-on / risk-off) and switches between two
 * semantic weight tables depending on the inferred regime. The state
 * posterior is updated via the Baum-Welch forward algorithm (online,
 * single-pass) using observed signal statistics as emissions.
 *
 * ## Regime semantics
 *
 * | Regime   | Token weight adjustment                                       |
 * |----------|---------------------------------------------------------------|
 * | RiskOn   | Positive tokens get their nominal weight; neutral baseline.   |
 * | RiskOff  | Negative tokens are amplified by `risk_off_neg_multiplier`;   |
 * |          | positive tokens are discounted by `risk_off_pos_discount`.    |
 *
 * ## HMM transition model
 * The transition matrix is parameterised by two probabilities:
 * - `p_stay_risk_on`  — probability of remaining in RiskOn  given RiskOn.
 * - `p_stay_risk_off` — probability of remaining in RiskOff given RiskOff.
 *
 * Emission likelihood is modelled as a Bernoulli on the sign of the latest
 * signal: positive signals are more likely under RiskOn, negative under RiskOff.
 *
 * ## Usage
 * ```cpp
 * RegimeSwitcher switcher;
 * switcher.update(signal_value);
 * double weight = switcher.apply_weight(base_weight, token_is_positive);
 * ```
 *
 * ## Thread safety
 * All methods are thread-safe.
 *
 * ## Feature flag
 * Controlled by `LLMQUANT_ENABLE_REGIME_SWITCHER` (default ON).
 */
class RegimeSwitcher {
public:
    /** @brief The two HMM regime states. */
    enum class Regime { RiskOn = 0, RiskOff = 1 };

    /** @brief Callback invoked when the inferred regime changes. */
    using RegimeCallback = std::function<void(Regime new_regime, Regime old_regime)>;

    /**
     * @brief Configuration.
     */
    struct Config {
        /**
         * @brief Probability of staying in RiskOn given currently RiskOn.
         * Default: 0.95 (high persistence — regimes are sticky).
         */
        double p_stay_risk_on{0.95};

        /**
         * @brief Probability of staying in RiskOff given currently RiskOff.
         * Default: 0.90.
         */
        double p_stay_risk_off{0.90};

        /**
         * @brief Emission probability of observing a positive signal in RiskOn.
         * Default: 0.70.
         */
        double p_pos_given_risk_on{0.70};

        /**
         * @brief Emission probability of observing a positive signal in RiskOff.
         * Default: 0.30.
         */
        double p_pos_given_risk_off{0.30};

        /**
         * @brief Multiplier applied to negative-token base weights in RiskOff regime.
         * Values > 1.0 amplify negative signals (more defensive / short bias).
         * Default: 1.50.
         */
        double risk_off_neg_multiplier{1.50};

        /**
         * @brief Discount factor applied to positive-token base weights in RiskOff regime.
         * Values < 1.0 reduce bullish signal strength.
         * Default: 0.60.
         */
        double risk_off_pos_discount{0.60};

        /**
         * @brief Minimum posterior probability of RiskOff before the regime
         * officially flips to RiskOff.  Hysteresis prevents rapid flapping.
         * Default: 0.65.
         */
        double risk_off_threshold{0.65};

        /**
         * @brief Minimum posterior probability of RiskOn before recovering from RiskOff.
         * Default: 0.70.
         */
        double risk_on_threshold{0.70};
    };

    explicit RegimeSwitcher(Config cfg = Config{});

    // Non-copyable.
    RegimeSwitcher(const RegimeSwitcher&)            = delete;
    RegimeSwitcher& operator=(const RegimeSwitcher&) = delete;

    /**
     * @brief Feed a new signal observation and update regime posterior.
     *
     * @param signal_value  Latest raw signal (positive → bullish, negative → bearish).
     *
     * Thread-safe.
     */
    void update(double signal_value);

    /**
     * @brief Apply regime-aware weighting to a base token weight.
     *
     * In RiskOn regime: returns `base_weight` unchanged.
     * In RiskOff regime:
     *   - If `token_is_positive`: returns `base_weight * risk_off_pos_discount`.
     *   - Otherwise:              returns `base_weight * risk_off_neg_multiplier`.
     *
     * Thread-safe.
     *
     * @param base_weight      Nominal weight for this token.
     * @param token_is_positive true if the token has positive semantic polarity.
     * @return Adjusted weight.
     */
    [[nodiscard]] double apply_weight(double base_weight, bool token_is_positive) const;

    /**
     * @brief Return the current inferred regime.
     *
     * Thread-safe.
     */
    [[nodiscard]] Regime current_regime() const noexcept {
        return static_cast<Regime>(regime_.load(std::memory_order_acquire));
    }

    /**
     * @brief Return the posterior probability of being in RiskOff.
     *
     * Thread-safe.
     */
    [[nodiscard]] double risk_off_posterior() const noexcept;

    /**
     * @brief Total number of regime transitions since construction.
     *
     * Thread-safe.
     */
    [[nodiscard]] uint64_t total_transitions() const noexcept {
        return transitions_.load(std::memory_order_relaxed);
    }

    /**
     * @brief Total update() calls since construction.
     *
     * Thread-safe.
     */
    [[nodiscard]] uint64_t total_updates() const noexcept {
        return updates_.load(std::memory_order_relaxed);
    }

    /**
     * @brief Human-readable name for a Regime value.
     */
    [[nodiscard]] static const char* regime_name(Regime r) noexcept;

    /**
     * @brief Register a callback invoked on every regime change.
     *
     * Thread-safe.
     */
    void set_regime_callback(RegimeCallback cb);

    /**
     * @brief Update configuration; resets posterior to uniform prior.
     *
     * Thread-safe.
     */
    void update_config(const Config& cfg);

    /**
     * @brief Reset posterior to uniform (0.5 / 0.5).
     *
     * Thread-safe.
     */
    void reset();

    /**
     * @brief Serialise state to a JSON object string.
     *
     * Fields: regime, risk_off_posterior, total_updates, total_transitions.
     */
    [[nodiscard]] std::string to_stats_json() const;

private:
    void update_regime_locked();

    mutable std::mutex mtx_;
    Config cfg_;

    // HMM state posteriors: prob[0] = P(RiskOn | observations),
    //                        prob[1] = P(RiskOff | observations).
    double prob_risk_on_{0.5};
    double prob_risk_off_{0.5};

    std::atomic<int>      regime_{static_cast<int>(Regime::RiskOn)};
    std::atomic<uint64_t> transitions_{0};
    std::atomic<uint64_t> updates_{0};

    RegimeCallback regime_cb_;
};

} // namespace llmquant
