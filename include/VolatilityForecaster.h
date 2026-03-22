#pragma once

#include <atomic>
#include <cstdint>
#include <functional>
#include <mutex>
#include <string>

namespace llmquant {

/**
 * @brief GARCH(1,1)-inspired online volatility forecaster from the sentiment stream.
 *
 * Classical GARCH(1,1): σ²_t = ω + α*r²_{t-1} + β*σ²_{t-1}
 *
 * Here `r_t` is the incremental delta_bias_shift (sentiment "return") and
 * σ²_t is the one-step-ahead conditional variance of the sentiment stream.
 * High forecast volatility indicates an uncertain, noisy signal environment;
 * the caller may use this to widen risk thresholds or reduce position size.
 *
 * ## Parameters
 * Default calibration (ω=0.00001, α=0.15, β=0.84) produces a persistence
 * of α+β=0.99, typical for financial return series.
 *
 * ## Annualised vol proxy
 * The raw conditional variance is in units of bias²/token.  The caller
 * can interpret `sqrt(conditional_variance())` as a per-token "vol" and
 * compare it against a user-defined threshold to scale risk parameters.
 *
 * ## Feature flag
 * Controlled by `LLMQUANT_ENABLE_VOL_FORECASTER` (default ON).
 */
class VolatilityForecaster {
public:
    /**
     * @brief Configuration.
     */
    struct Config {
        /**
         * @brief GARCH ω (long-run variance weight).
         *
         * Anchors the conditional variance toward a long-run mean.
         * Default 0.00001.
         */
        double omega{0.00001};

        /**
         * @brief GARCH α (shock coefficient).
         *
         * Weight on the squared most-recent innovation r²_{t-1}.
         * Default 0.15.
         */
        double alpha{0.15};

        /**
         * @brief GARCH β (persistence coefficient).
         *
         * Weight on the previous conditional variance σ²_{t-1}.
         * Default 0.84.  α + β must be < 1 for covariance stationarity.
         */
        double beta{0.84};

        /**
         * @brief Initial conditional variance σ²_0.
         *
         * Warm-starts the recursion.  Default 0.01.
         */
        double initial_variance{0.01};

        /**
         * @brief High-vol threshold.
         *
         * When sqrt(σ²_t) > this value, on_high_vol fires.
         * 0 = disabled.  Default 0.1.
         */
        double high_vol_threshold{0.1};

        /**
         * @brief Callback fired when conditional vol exceeds high_vol_threshold.
         *
         * Parameters: (current_vol, conditional_variance).
         * Called from the thread invoking record().
         */
        std::function<void(double, double)> on_high_vol;
    };

    // -----------------------------------------------------------------------
    // Construction
    // -----------------------------------------------------------------------

    explicit VolatilityForecaster(Config config = Config{});

    // -----------------------------------------------------------------------
    // Update
    // -----------------------------------------------------------------------

    /**
     * @brief Record a new bias value and update the GARCH recursion.
     *
     * `innovation = value - previous_value` on first call; thereafter
     * uses the running difference.  Thread-safe.
     *
     * @param bias_value  Current delta_bias_shift from TradeSignal.
     */
    void record(double bias_value);

    /**
     * @brief Reset state to the configured initial variance.
     *
     * Thread-safe.
     */
    void reset();

    // -----------------------------------------------------------------------
    // Query
    // -----------------------------------------------------------------------

    /**
     * @brief Current one-step-ahead conditional variance σ²_t.
     *
     * Thread-safe.
     */
    [[nodiscard]] double conditional_variance() const noexcept;

    /**
     * @brief Current conditional volatility √σ²_t (per-token standard deviation).
     *
     * Thread-safe.
     */
    [[nodiscard]] double conditional_vol() const noexcept;

    /**
     * @brief Long-run (unconditional) variance ω / (1 - α - β).
     *
     * Returns conditional_variance() if α + β ≥ 1 (non-stationary region).
     */
    [[nodiscard]] double long_run_variance() const noexcept;

    /**
     * @brief True if the most-recent record() call exceeded high_vol_threshold.
     */
    [[nodiscard]] bool is_high_vol() const noexcept {
        return high_vol_.load(std::memory_order_relaxed);
    }

    // -----------------------------------------------------------------------
    // Statistics
    // -----------------------------------------------------------------------

    /** @brief Total calls to record(). */
    [[nodiscard]] uint64_t total_records() const noexcept {
        return total_records_.load(std::memory_order_relaxed);
    }

    /** @brief Times on_high_vol has fired. */
    [[nodiscard]] uint64_t high_vol_events() const noexcept {
        return high_vol_events_.load(std::memory_order_relaxed);
    }

    /**
     * @brief Update configuration; resets state.
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

    double sigma2_{0.01};         // current conditional variance
    double prev_value_{0.0};      // previous bias value for innovation calculation
    bool   has_prev_{false};

    std::atomic<bool>     high_vol_{false};
    std::atomic<uint64_t> total_records_{0};
    std::atomic<uint64_t> high_vol_events_{0};
};

} // namespace llmquant
