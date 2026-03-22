#pragma once

#include <atomic>
#include <functional>
#include <mutex>
#include <string>

namespace llmquant {

/**
 * @brief Dynamically scales trade notional based on market regime and
 *        real-time volatility.
 *
 * Combines two orthogonal risk dimensions into a single size multiplier:
 *
 * 1. **Regime (Hurst exponent)**
 *    Provided by FractalDimensionEstimator.  A Hurst exponent H > 0.5
 *    implies persistent/trending behaviour — momentum strategies perform
 *    better.  H < 0.5 implies mean reversion — strategies should be smaller
 *    and more selective to avoid whipsaws.
 *
 * 2. **Volatility (GARCH forecast or realized vol)**
 *    Provided by VolatilityForecaster or any rolling estimator.  High
 *    volatility → smaller size (fat tails, wider spreads).
 *
 * ## Size formula
 *
 * ```
 * regime_factor = lerp(regime_min, regime_max,
 *                      clamp((H - 0.5) / (H_max - 0.5), 0, 1))
 *
 * vol_factor = clamp(target_vol / current_vol, vol_floor, vol_cap)
 *
 * size_multiplier = base_notional * regime_factor * vol_factor
 * ```
 *
 * In the default configuration, a trending market (H = 0.8) with benign
 * volatility produces a multiplier ≈ 1.5×; a mean-reverting market (H = 0.2)
 * with elevated volatility might reduce to 0.2×.
 *
 * ## Feature flag
 * Controlled by `LLMQUANT_ENABLE_REGIME_SIZER` (default ON).
 */
class RegimeAwareSizer {
public:
    struct Config {
        // ── Regime axis ──────────────────────────────────────────────────────

        /** @brief Hurst exponent at which maximum size is allowed. Default 0.8. */
        double hurst_max{0.8};

        /**
         * @brief Size multiplier when H == 0.5 (random walk, no regime).
         * Default 1.0 (no adjustment).
         */
        double regime_neutral{1.0};

        /**
         * @brief Size multiplier at the trending extreme (H == hurst_max).
         * Default 1.5.
         */
        double regime_max{1.5};

        /**
         * @brief Size multiplier at the mean-reverting extreme (H close to 0).
         * Default 0.4.
         */
        double regime_min{0.4};

        // ── Volatility axis ──────────────────────────────────────────────────

        /**
         * @brief Target annualised volatility (fractional, e.g. 0.15 = 15 %).
         * When realised vol equals this, the vol factor is 1.0.
         * Default 0.20.
         */
        double target_vol{0.20};

        /** @brief Minimum vol factor to prevent over-sizing. Default 0.1. */
        double vol_floor{0.1};

        /** @brief Maximum vol factor to prevent under-sizing. Default 3.0. */
        double vol_cap{3.0};

        // ── Output ───────────────────────────────────────────────────────────

        /**
         * @brief Callback fired whenever the recommended multiplier changes
         *        by more than change_threshold.
         *
         * Parameters: (new_multiplier, old_multiplier).
         */
        std::function<void(double new_mult, double old_mult)> on_size_change;

        /**
         * @brief Minimum fractional change required to fire on_size_change.
         * Default 0.05 (5 %).
         */
        double change_threshold{0.05};
    };

    // -----------------------------------------------------------------------
    // Construction
    // -----------------------------------------------------------------------

    explicit RegimeAwareSizer(Config config = Config{});

    // -----------------------------------------------------------------------
    // Update inputs
    // -----------------------------------------------------------------------

    /**
     * @brief Update the Hurst exponent.
     *
     * Pass the value returned by FractalDimensionEstimator::hurst().
     * Thread-safe.
     */
    void update_hurst(double h) noexcept;

    /**
     * @brief Update the current annualised volatility estimate.
     *
     * Pass the value returned by VolatilityForecaster::annualised_vol() or
     * any equivalent realized-vol estimator.  Must be > 0.
     * Thread-safe.
     */
    void update_vol(double vol) noexcept;

    // -----------------------------------------------------------------------
    // Query
    // -----------------------------------------------------------------------

    /**
     * @brief Combined size multiplier incorporating both regime and vol.
     *
     * Multiply your base notional by this value.  Thread-safe.
     */
    [[nodiscard]] double size_multiplier() const noexcept {
        return multiplier_.load(std::memory_order_relaxed);
    }

    /** @brief Current regime factor component.  Thread-safe. */
    [[nodiscard]] double regime_factor() const noexcept {
        return regime_factor_.load(std::memory_order_relaxed);
    }

    /** @brief Current volatility factor component.  Thread-safe. */
    [[nodiscard]] double vol_factor() const noexcept {
        return vol_factor_.load(std::memory_order_relaxed);
    }

    /** @brief Most-recently fed Hurst exponent.  Thread-safe. */
    [[nodiscard]] double current_hurst() const noexcept {
        return hurst_.load(std::memory_order_relaxed);
    }

    /** @brief Most-recently fed volatility.  Thread-safe. */
    [[nodiscard]] double current_vol() const noexcept {
        return vol_.load(std::memory_order_relaxed);
    }

    /** @brief Total times size_multiplier was recomputed.  Thread-safe. */
    [[nodiscard]] uint64_t update_count() const noexcept {
        return update_count_.load(std::memory_order_relaxed);
    }

    /** @brief Number of on_size_change callbacks fired.  Thread-safe. */
    [[nodiscard]] uint64_t change_events() const noexcept {
        return change_events_.load(std::memory_order_relaxed);
    }

    // -----------------------------------------------------------------------
    // Config / reset
    // -----------------------------------------------------------------------

    /** @brief Replace configuration; resets multiplier to 1.0. */
    void update_config(Config config);

    /** @brief Reset multiplier and counters. */
    void reset();

    /** @brief JSON snapshot of current state. */
    [[nodiscard]] std::string to_stats_json() const;

private:
    /// Recompute multiplier from current hurst_ / vol_.  Caller must hold mutex_.
    /// Returns new multiplier; fires callback outside lock.
    double recompute_locked();

    mutable std::mutex mutex_;
    Config config_;

    double hurst_val_{0.5};
    double vol_val_{0.0};

    std::atomic<double>   hurst_{0.5};
    std::atomic<double>   vol_{0.0};
    std::atomic<double>   regime_factor_{1.0};
    std::atomic<double>   vol_factor_{1.0};
    std::atomic<double>   multiplier_{1.0};
    std::atomic<uint64_t> update_count_{0};
    std::atomic<uint64_t> change_events_{0};
};

} // namespace llmquant
