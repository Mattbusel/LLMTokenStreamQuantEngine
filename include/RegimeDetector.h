#pragma once

#include <atomic>
#include <cstdint>
#include <functional>
#include <mutex>
#include <string>

namespace llmquant {

/**
 * @brief Rolling market-regime classifier for the signal pipeline.
 *
 * Combines three real-time indicators into a discrete regime label:
 *
 *   1. **Bias momentum** — EMA of delta_bias_shift; positive → bullish, negative → bearish.
 *   2. **Volatility level** — EMA of |volatility_adjustment|; high → volatile.
 *   3. **Block rate** — Fraction of signals blocked by risk gates; high → risk-off.
 *
 * ## Regime labels
 * | Regime    | Condition                                                         |
 * |-----------|-------------------------------------------------------------------|
 * | Bull      | Bias momentum > momentum_threshold AND block_rate < risk_off_threshold |
 * | Bear      | Bias momentum < -momentum_threshold AND block_rate < risk_off_threshold |
 * | Volatile  | Volatility EMA > volatility_threshold (takes priority over momentum) |
 * | RiskOff   | Block rate >= risk_off_threshold (highest priority)                |
 * | Neutral   | None of the above conditions met                                  |
 *
 * Regime changes fire an optional callback.
 *
 * ## Thread safety
 * `update()` and all accessors are safe to call from any thread.
 *
 * ## Feature flag
 * Controlled by `LLMQUANT_ENABLE_REGIME_DETECTOR` (default ON).
 */
class RegimeDetector {
public:
    /**
     * @brief Detected market regime.
     */
    enum class Regime {
        Neutral,  ///< No directional signal; waiting for clear trend.
        Bull,     ///< Positive bias momentum with acceptable risk.
        Bear,     ///< Negative bias momentum with acceptable risk.
        Volatile, ///< High volatility; directional edge unreliable.
        RiskOff,  ///< High block rate; risk gates restricting signal flow.
    };

    /**
     * @brief Regime classifier tuning parameters.
     */
    struct Config {
        /**
         * @brief EMA alpha for bias momentum smoothing.
         *
         * 0.1 = heavy smoothing; 0.5 = moderate.
         * Default 0.15.
         */
        double momentum_alpha{0.15};

        /**
         * @brief EMA alpha for volatility level smoothing.
         *
         * Default 0.2.
         */
        double volatility_alpha{0.2};

        /**
         * @brief EMA alpha for block rate smoothing.
         *
         * Default 0.1 (slower to avoid flapping on transient spikes).
         */
        double block_rate_alpha{0.1};

        /**
         * @brief |bias momentum| above which the regime is Bull or Bear.
         *
         * Default 0.05 (5% of max bias).
         */
        double momentum_threshold{0.05};

        /**
         * @brief Volatility EMA above which the regime is Volatile.
         *
         * Default 0.10.
         */
        double volatility_threshold{0.10};

        /**
         * @brief Block rate EMA above which the regime is RiskOff.
         *
         * Default 0.70 (70% of signals blocked → risk-off mode).
         */
        double risk_off_threshold{0.70};
    };

    /**
     * @brief Callback invoked when the detected regime changes.
     */
    using RegimeChangeCallback =
        std::function<void(Regime new_regime, Regime old_regime)>;

    /**
     * @brief Construct with the given configuration.
     *
     * @param config Thresholds and EMA parameters.
     */
    explicit RegimeDetector(Config config = Config{});

    /**
     * @brief Feed a new signal observation into the detector.
     *
     * Updates all three EMAs and reclassifies the current regime.
     * Fires the regime-change callback if the regime changed.
     *
     * Thread-safe.
     *
     * @param bias_shift      delta_bias_shift from the emitted TradeSignal.
     * @param volatility      volatility_adjustment from the signal.
     * @param signal_blocked  true if the signal was blocked by a risk gate.
     */
    void update(double bias_shift, double volatility, bool signal_blocked);

    /**
     * @brief Return the current regime.
     *
     * Thread-safe (atomic read).
     */
    [[nodiscard]] Regime current_regime() const noexcept {
        return static_cast<Regime>(regime_.load(std::memory_order_acquire));
    }

    /**
     * @brief Return a human-readable name for a regime value.
     *
     * @param r Regime enum value.
     * @return "Neutral", "Bull", "Bear", "Volatile", "RiskOff".
     */
    [[nodiscard]] static const char* regime_name(Regime r) noexcept;

    /**
     * @brief Return the current regime name string.
     *
     * Thread-safe.
     */
    [[nodiscard]] const char* current_regime_name() const noexcept {
        return regime_name(current_regime());
    }

    /**
     * @brief Return the current EMA-smoothed bias momentum.
     *
     * Thread-safe (atomic read).
     */
    [[nodiscard]] double get_momentum() const noexcept {
        return momentum_ema_.load(std::memory_order_relaxed);
    }

    /**
     * @brief Return the current EMA-smoothed volatility level.
     *
     * Thread-safe (atomic read).
     */
    [[nodiscard]] double get_volatility_ema() const noexcept {
        return volatility_ema_.load(std::memory_order_relaxed);
    }

    /**
     * @brief Return the current EMA-smoothed block rate.
     *
     * Thread-safe (atomic read).
     */
    [[nodiscard]] double get_block_rate_ema() const noexcept {
        return block_rate_ema_.load(std::memory_order_relaxed);
    }

    /**
     * @brief Total number of regime transitions since construction.
     *
     * Thread-safe (atomic read).
     */
    [[nodiscard]] uint64_t total_transitions() const noexcept {
        return transitions_.load(std::memory_order_relaxed);
    }

    /**
     * @brief Total number of update() calls since construction.
     *
     * Thread-safe (atomic read).
     */
    [[nodiscard]] uint64_t total_updates() const noexcept {
        return updates_.load(std::memory_order_relaxed);
    }

    /**
     * @brief Register a callback invoked on every regime change.
     *
     * @param cb Callable (new_regime, old_regime); may be nullptr to unregister.
     */
    void set_regime_change_callback(RegimeChangeCallback cb);

    /**
     * @brief Update configuration at runtime.
     *
     * EMA state is preserved.  Thread-safe.
     *
     * @param config New configuration.
     */
    void update_config(const Config& config);

    /**
     * @brief Return a copy of the current configuration.
     *
     * Thread-safe.
     */
    [[nodiscard]] Config get_config() const;

    /**
     * @brief Serialise current state to a JSON object string.
     *
     * Fields: regime, momentum, volatility_ema, block_rate_ema,
     * total_updates, total_transitions.
     * Thread-safe.
     */
    [[nodiscard]] std::string to_stats_json() const;

    /**
     * @brief Reset all EMA state and transition counters.
     *
     * The regime resets to Neutral.  Thread-safe.
     */
    void reset();

private:
    [[nodiscard]] Regime classify_locked() const;

    mutable std::mutex mutex_;
    Config config_;

    std::atomic<int>    regime_{static_cast<int>(Regime::Neutral)};
    std::atomic<double> momentum_ema_{0.0};
    std::atomic<double> volatility_ema_{0.0};
    std::atomic<double> block_rate_ema_{0.0};

    std::atomic<uint64_t> transitions_{0};
    std::atomic<uint64_t> updates_{0};

    RegimeChangeCallback regime_cb_;
};

} // namespace llmquant
