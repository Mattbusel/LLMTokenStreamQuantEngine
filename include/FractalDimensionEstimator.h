#pragma once

#include <atomic>
#include <cstdint>
#include <deque>
#include <functional>
#include <mutex>
#include <string>

namespace llmquant {

/**
 * @brief Online Hurst-exponent estimator for the bias/sentiment time series.
 *
 * Uses the classical Rescaled Range (R/S) statistic computed on a rolling
 * window to estimate the Hurst exponent H ∈ (0, 1):
 *
 *   H > 0.5 → persistent / trending signal  (follow the bias)
 *   H = 0.5 → random walk (no exploitable structure)
 *   H < 0.5 → anti-persistent / mean-reverting (fade the bias)
 *
 * ## Algorithm
 * Over a window of n values {x_1, …, x_n}:
 *   1. μ  = mean(x)
 *   2. D_k = Σ_{i=1}^{k} (x_i − μ)  (cumulative deviation)
 *   3. R   = max(D_k) − min(D_k)
 *   4. S   = std_dev(x)
 *   5. RS  = R / S   (NaN-safe: returns 0 when S ≈ 0)
 *
 * Two nested windows (full window n and half-window n/2) provide a
 * two-point log–log regression estimate:
 *   H ≈ log(RS_n / RS_{n/2}) / log(2)
 *
 * ## Feature flag
 * Controlled by `LLMQUANT_ENABLE_FRACTAL_DIM` (default ON).
 */
class FractalDimensionEstimator {
public:
    /**
     * @brief Called whenever the Hurst estimate crosses the 0.5 boundary.
     *
     * Parameters: (previous_hurst, new_hurst).
     * Fired from the thread invoking record().
     */
    using RegimeCallback = std::function<void(double prev_h, double new_h)>;

    /**
     * @brief Configuration.
     */
    struct Config {
        /**
         * @brief Full window size for R/S computation.
         *
         * Must be ≥ 4.  Default 64.
         */
        int window_size{64};

        /**
         * @brief Minimum samples before hurst() returns a meaningful value.
         *
         * Default is window_size / 2.  Override to warm up faster.
         */
        int min_samples{32};

        /**
         * @brief Callback fired when the Hurst estimate crosses 0.5.
         *
         * Lets the caller switch between trend-following and mean-reversion
         * strategies without polling.
         */
        RegimeCallback on_regime_change;
    };

    // -----------------------------------------------------------------------
    // Construction
    // -----------------------------------------------------------------------

    explicit FractalDimensionEstimator(Config config = Config{});

    // -----------------------------------------------------------------------
    // Update
    // -----------------------------------------------------------------------

    /**
     * @brief Record a new observation (delta_bias_shift or raw bias).
     *
     * Updates the rolling window and recomputes the Hurst estimate.
     * Thread-safe.
     *
     * @param value  Scalar observation.
     */
    void record(double value);

    /**
     * @brief Reset all state.
     *
     * Thread-safe.
     */
    void reset();

    // -----------------------------------------------------------------------
    // Query
    // -----------------------------------------------------------------------

    /**
     * @brief Current Hurst exponent estimate.
     *
     * Returns 0.5 (random walk) when fewer than min_samples have been seen.
     * Thread-safe.
     */
    [[nodiscard]] double hurst() const noexcept;

    /**
     * @brief True when H > 0.5 (persistent / trending).
     *
     * Thread-safe.
     */
    [[nodiscard]] bool is_trending() const noexcept {
        return hurst() > 0.5;
    }

    /**
     * @brief True when H < 0.5 (anti-persistent / mean-reverting).
     *
     * Thread-safe.
     */
    [[nodiscard]] bool is_mean_reverting() const noexcept {
        return hurst() < 0.5;
    }

    /**
     * @brief Raw R/S value for the full window.
     *
     * Thread-safe.
     */
    [[nodiscard]] double rs_full() const noexcept;

    /**
     * @brief Number of samples currently in the window.
     *
     * Thread-safe.
     */
    [[nodiscard]] int sample_count() const noexcept;

    // -----------------------------------------------------------------------
    // Statistics
    // -----------------------------------------------------------------------

    /** @brief Total observations recorded. */
    [[nodiscard]] uint64_t total_records() const noexcept {
        return total_records_.load(std::memory_order_relaxed);
    }

    /** @brief Number of regime-change events fired. */
    [[nodiscard]] uint64_t regime_changes() const noexcept {
        return regime_changes_.load(std::memory_order_relaxed);
    }

    /**
     * @brief Update configuration; clears the window.
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
    Config             config_;
    std::deque<double> window_;
    double             last_hurst_{0.5};
    double             rs_full_{0.0};

    std::atomic<uint64_t> total_records_{0};
    std::atomic<uint64_t> regime_changes_{0};

    // Compute R/S for a sub-range of the window [begin, end).
    // Returns 0 if the range is too short or stddev == 0.
    static double compute_rs(const std::deque<double>& w, int begin, int end) noexcept;
};

} // namespace llmquant
