#pragma once

#include <atomic>
#include <chrono>
#include <cstdint>
#include <functional>
#include <mutex>
#include <string>

namespace llmquant {

/**
 * @brief Dynamically adjusts the token-polling interval based on market state.
 *
 * Polling the LLM API at a fixed rate wastes resources during quiet periods
 * and may miss events during volatile ones.  This controller applies a
 * multiplicative adaptation rule:
 *
 *   interval_ms_new = interval_ms_current * decay_factor  (when active)
 *   interval_ms_new = interval_ms_current * growth_factor (when quiet)
 *
 * "Activity" is measured by the magnitude of recent bias deltas.  Above the
 * high-activity threshold the interval shrinks (faster polling); below the
 * low-activity threshold it grows (slower polling).
 *
 * ## Intended use
 * @code
 * AdaptiveSamplingController ctrl;
 * // After each token batch:
 * ctrl.record_activity(std::abs(signal.delta_bias_shift));
 * // When scheduling the next poll:
 * auto delay = ctrl.recommended_interval_ms();
 * std::this_thread::sleep_for(std::chrono::milliseconds(delay));
 * @endcode
 *
 * ## Feature flag
 * Controlled by `LLMQUANT_ENABLE_ADAPTIVE_SAMPLING` (default ON).
 */
class AdaptiveSamplingController {
public:
    /**
     * @brief Configuration.
     */
    struct Config {
        /**
         * @brief Minimum polling interval (fastest sampling). Default 10 ms.
         */
        int64_t min_interval_ms{10};

        /**
         * @brief Maximum polling interval (slowest sampling). Default 5000 ms.
         */
        int64_t max_interval_ms{5000};

        /**
         * @brief Initial / default polling interval. Default 100 ms.
         */
        int64_t initial_interval_ms{100};

        /**
         * @brief Activity magnitude above which polling accelerates.
         *
         * Default 0.01 (1% bias delta per token).
         */
        double high_activity_threshold{0.01};

        /**
         * @brief Activity magnitude below which polling decelerates.
         *
         * Default 0.001.
         */
        double low_activity_threshold{0.001};

        /**
         * @brief Multiplicative factor to shrink interval on high activity.
         *
         * Must be in (0, 1).  Default 0.7 (shrink by 30%).
         */
        double decay_factor{0.7};

        /**
         * @brief Multiplicative factor to grow interval on low activity.
         *
         * Must be > 1.  Default 1.3 (grow by 30%).
         */
        double growth_factor{1.3};

        /**
         * @brief Callback fired when the recommended interval changes tier
         *        (crosses min/max bounds or moves from fast→slow or slow→fast).
         *
         * Parameters: (new_interval_ms).
         */
        std::function<void(int64_t)> on_interval_change;
    };

    // -----------------------------------------------------------------------
    // Construction
    // -----------------------------------------------------------------------

    explicit AdaptiveSamplingController(Config config = Config{});

    // -----------------------------------------------------------------------
    // Update
    // -----------------------------------------------------------------------

    /**
     * @brief Record an activity observation and adapt the interval.
     *
     * @param activity  Non-negative activity magnitude (e.g. |delta_bias_shift|).
     * Thread-safe.
     */
    void record_activity(double activity);

    /**
     * @brief Reset the interval to initial_interval_ms.
     *
     * Thread-safe.
     */
    void reset();

    // -----------------------------------------------------------------------
    // Query
    // -----------------------------------------------------------------------

    /**
     * @brief Current recommended polling interval in milliseconds.
     *
     * Thread-safe.
     */
    [[nodiscard]] int64_t recommended_interval_ms() const noexcept;

    /**
     * @brief True when the interval is at the minimum (fastest polling).
     *
     * Thread-safe.
     */
    [[nodiscard]] bool is_at_min() const noexcept;

    /**
     * @brief True when the interval is at the maximum (slowest polling).
     *
     * Thread-safe.
     */
    [[nodiscard]] bool is_at_max() const noexcept;

    // -----------------------------------------------------------------------
    // Statistics
    // -----------------------------------------------------------------------

    /** @brief Total activity observations recorded. */
    [[nodiscard]] uint64_t total_records() const noexcept {
        return total_records_.load(std::memory_order_relaxed);
    }

    /** @brief Number of times the interval was decreased. */
    [[nodiscard]] uint64_t accelerations() const noexcept {
        return accelerations_.load(std::memory_order_relaxed);
    }

    /** @brief Number of times the interval was increased. */
    [[nodiscard]] uint64_t decelerations() const noexcept {
        return decelerations_.load(std::memory_order_relaxed);
    }

    /**
     * @brief Update configuration; resets interval to new initial_interval_ms.
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
    Config  config_;
    int64_t interval_ms_;

    std::atomic<uint64_t> total_records_{0};
    std::atomic<uint64_t> accelerations_{0};
    std::atomic<uint64_t> decelerations_{0};
};

} // namespace llmquant
