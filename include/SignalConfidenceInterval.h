#pragma once

#include <atomic>
#include <cstdint>
#include <functional>
#include <mutex>
#include <string>
#include <vector>

namespace llmquant {

/**
 * @brief Rolling jackknife confidence interval estimator for trade signals.
 *
 * Position sizing should reflect not just the point estimate of a signal's
 * strength but also how uncertain that estimate is.  A wide confidence
 * interval means the signal is noisy; a tight interval means it is reliable.
 *
 * This class maintains a rolling window of signal values and applies the
 * delete-1 jackknife to estimate the standard error of the mean.  The 95%
 * confidence interval is:
 *
 *   CI = [mean − 1.96 × se_jack, mean + 1.96 × se_jack]
 *
 * where se_jack = sqrt((n-1)/n × Σ(pseudo_value − mean_pseudo)²).
 *
 * This is O(N) per update but fast enough for the typical window sizes
 * (32–256 signals) used in practice.
 *
 * A narrow-interval callback fires when the half-width drops below
 * `narrow_threshold`, signalling a high-confidence signal environment.
 *
 * ## Feature flag
 * Controlled by `LLMQUANT_ENABLE_SIGNAL_CI` (default ON).
 *
 * Thread safety: all public methods are thread-safe.
 */
class SignalConfidenceInterval {
public:
    // -----------------------------------------------------------------------
    // Configuration
    // -----------------------------------------------------------------------

    struct Config {
        /**
         * @brief Number of most recent signals retained for jackknife.
         *
         * Larger = more stable estimates, higher computational cost.
         * Default 64.
         */
        std::size_t window_size{64};

        /**
         * @brief z-multiplier for the interval.
         *
         * 1.96 = 95% CI (default).  Use 2.576 for 99%.
         */
        double z{1.96};

        /**
         * @brief Half-width below which the interval is considered "narrow".
         *
         * Fires `on_narrow_interval` callback.  Default 0.05.
         */
        double narrow_threshold{0.05};

        /**
         * @brief Fired when the CI half-width drops below narrow_threshold.
         *
         * Parameters: (mean, half_width).  Called outside the lock.
         */
        std::function<void(double mean, double half_width)> on_narrow_interval;

        /**
         * @brief Fired when the CI half-width rises above wide_threshold.
         *
         * Parameters: (mean, half_width).  Called outside the lock.
         */
        std::function<void(double mean, double half_width)> on_wide_interval;

        /**
         * @brief Half-width above which the interval is considered "wide".
         *
         * Default 0.3.
         */
        double wide_threshold{0.3};
    };

    // -----------------------------------------------------------------------
    // Construction
    // -----------------------------------------------------------------------

    explicit SignalConfidenceInterval(Config cfg = Config{});

    // -----------------------------------------------------------------------
    // Signal intake
    // -----------------------------------------------------------------------

    /**
     * @brief Add a signal value to the rolling window and recompute the CI.
     *
     * @param value Signal value (e.g. delta_bias_shift or confidence score).
     */
    void record(double value);

    // -----------------------------------------------------------------------
    // Accessors
    // -----------------------------------------------------------------------

    /**
     * @brief Rolling mean of the current window.
     */
    [[nodiscard]] double mean() const noexcept {
        return mean_.load(std::memory_order_relaxed);
    }

    /**
     * @brief Current CI half-width (z × jackknife SE).
     */
    [[nodiscard]] double half_width() const noexcept {
        return half_width_.load(std::memory_order_relaxed);
    }

    /**
     * @brief Lower bound of the CI: mean − half_width.
     */
    [[nodiscard]] double lower() const noexcept {
        return mean_.load(std::memory_order_relaxed)
             - half_width_.load(std::memory_order_relaxed);
    }

    /**
     * @brief Upper bound of the CI: mean + half_width.
     */
    [[nodiscard]] double upper() const noexcept {
        return mean_.load(std::memory_order_relaxed)
             + half_width_.load(std::memory_order_relaxed);
    }

    /**
     * @brief True when half_width < narrow_threshold.
     */
    [[nodiscard]] bool is_narrow() const noexcept;

    /**
     * @brief True when half_width > wide_threshold.
     */
    [[nodiscard]] bool is_wide() const noexcept;

    /**
     * @brief Number of values in the current window (≤ window_size).
     */
    [[nodiscard]] std::size_t window_fill() const noexcept;

    /**
     * @brief Total values recorded.
     */
    [[nodiscard]] uint64_t total_records() const noexcept {
        return total_.load(std::memory_order_relaxed);
    }

    // -----------------------------------------------------------------------
    // Control
    // -----------------------------------------------------------------------

    /**
     * @brief Clear the rolling window; retain configuration.
     */
    void reset();

    /**
     * @brief Update configuration and reset.
     */
    void update_config(const Config& cfg);

    /**
     * @brief JSON diagnostics snapshot.
     */
    [[nodiscard]] std::string to_stats_json() const;

private:
    Config cfg_;

    std::vector<double> window_;
    std::size_t head_{0};
    std::size_t count_{0};

    std::atomic<double>   mean_{0.0};
    std::atomic<double>   half_width_{0.0};
    std::atomic<uint64_t> total_{0};

    bool prev_narrow_{false};
    bool prev_wide_{false};

    mutable std::mutex mutex_;

    void recompute_locked();
};

} // namespace llmquant
