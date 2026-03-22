#pragma once

#include <atomic>
#include <cstdint>
#include <deque>
#include <mutex>
#include <string>

namespace llmquant {

/**
 * @brief Rolling Sharpe ratio of the bias signal stream.
 *
 * Maintains a rolling window of bias observations and computes the
 * signal-to-noise ratio as:
 *
 *   Sharpe = μ / σ
 *
 * where μ and σ are the rolling mean and standard deviation of the
 * bias stream.  A high Sharpe indicates a directionally consistent,
 * low-noise signal; a low or negative Sharpe indicates a noisy or
 * mean-reverting stream where position sizing should be conservative.
 *
 * ## Intended use
 * - Scale Kelly fraction down when Sharpe < `min_sharpe`.
 * - Combine with VolatilityForecaster: high GARCH vol + low Sharpe →
 *   near-zero sizing.
 * - Feed into RegimeDetector as an additional regime signal.
 *
 * ## Feature flag
 * Controlled by `LLMQUANT_ENABLE_ROLLING_SHARPE` (default ON).
 */
class RollingSharpeBiasTracker {
public:
    /**
     * @brief Configuration.
     */
    struct Config {
        /**
         * @brief Rolling window size (number of observations).
         *
         * Default 100.
         */
        int window_size{100};

        /**
         * @brief Minimum samples before Sharpe is reported (returns 0 otherwise).
         *
         * Default 20.
         */
        int min_samples{20};

        /**
         * @brief Sharpe ratio below which the signal is considered poor quality.
         *
         * Default 0.5.
         */
        double min_sharpe{0.5};
    };

    // -----------------------------------------------------------------------
    // Construction
    // -----------------------------------------------------------------------

    explicit RollingSharpeBiasTracker(Config config = Config{});

    // -----------------------------------------------------------------------
    // Update
    // -----------------------------------------------------------------------

    /**
     * @brief Record a new bias observation.
     *
     * Thread-safe.
     *
     * @param bias_value  Current delta_bias_shift.
     * @return            Rolling Sharpe ratio (0 if fewer than min_samples seen).
     */
    double record(double bias_value);

    /**
     * @brief Reset the rolling window.
     *
     * Thread-safe.
     */
    void reset();

    // -----------------------------------------------------------------------
    // Query
    // -----------------------------------------------------------------------

    /**
     * @brief Most recent rolling Sharpe ratio.
     *
     * Thread-safe.
     */
    [[nodiscard]] double last_sharpe() const noexcept;

    /**
     * @brief Rolling mean of the bias stream over the current window.
     *
     * Thread-safe.
     */
    [[nodiscard]] double rolling_mean() const noexcept;

    /**
     * @brief Rolling standard deviation of the bias stream.
     *
     * Thread-safe.
     */
    [[nodiscard]] double rolling_stddev() const noexcept;

    /**
     * @brief Number of observations in the current window.
     *
     * Thread-safe.
     */
    [[nodiscard]] int sample_count() const noexcept;

    /**
     * @brief True when the most recent Sharpe < Config::min_sharpe.
     *
     * Thread-safe.
     */
    [[nodiscard]] bool is_poor_quality() const noexcept {
        return poor_quality_.load(std::memory_order_relaxed);
    }

    // -----------------------------------------------------------------------
    // Statistics
    // -----------------------------------------------------------------------

    /** @brief Total calls to record(). */
    [[nodiscard]] uint64_t total_records() const noexcept {
        return total_records_.load(std::memory_order_relaxed);
    }

    /**
     * @brief Update configuration; clears the rolling window.
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
    Config             config_;

    std::deque<double> window_;
    double             sum_{0.0};
    double             sum2_{0.0};
    double             last_sharpe_{0.0};

    std::atomic<bool>     poor_quality_{false};
    std::atomic<uint64_t> total_records_{0};
};

} // namespace llmquant
