#pragma once

#include <atomic>
#include <functional>
#include <mutex>
#include <string>
#include <vector>

namespace llmquant {

/**
 * @brief Detects periodic cycles in the LLM sentiment bias stream via
 *        normalized autocorrelation.
 *
 * Macro events (Fed meetings, earnings seasons, FOMC cycles) imprint
 * periodic patterns into news flows and consequently into LLM sentiment.
 * This class maintains a rolling window of bias observations, computes the
 * normalized autocorrelation function (ACF) at lags [1, max_lag], and
 * identifies the dominant cycle period as the lag with the highest positive
 * autocorrelation.
 *
 * ## Interpretation
 * | Metric         | Meaning                                                  |
 * |----------------|----------------------------------------------------------|
 * | dominant_period| Most likely recurrence lag in number of signals          |
 * | cycle_strength | ACF at dominant_period; range [-1, 1]; >0.3 = meaningful|
 * | is_cyclic()    | cycle_strength >= cyclic_threshold (default 0.3)         |
 *
 * A `dominant_period` of 20 on a 1-signal-per-5-min feed ≈ 100-minute cycle.
 *
 * ## Feature flag
 * Controlled by `LLMQUANT_ENABLE_SENTIMENT_CYCLE` (default ON).
 */
class SentimentCycleDetector {
public:
    /**
     * @brief Configuration for the cycle detector.
     */
    struct Config {
        /** @brief Rolling window of bias samples used for ACF computation. */
        int window_size{128};

        /**
         * @brief Maximum lag (in samples) to test for periodicity.
         *
         * Practical upper bound: window_size / 4 to avoid high-variance estimates.
         */
        int max_lag{32};

        /**
         * @brief Minimum number of samples before cycle estimates are reported.
         *
         * Before this threshold dominant_period() returns 0 and cycle_strength()
         * returns 0.0.
         */
        int min_samples{32};

        /**
         * @brief ACF threshold above which is_cyclic() returns true.
         *
         * 0.3 is a conventional "significant autocorrelation" boundary.
         */
        double cyclic_threshold{0.30};

        /**
         * @brief Callback fired when the dominant period changes.
         *
         * Parameters: (new_period, old_period, cycle_strength).
         * Fired outside the internal lock.
         */
        std::function<void(int new_period, int old_period, double strength)> on_period_change;
    };

    /** @brief Construct with default configuration. */
    SentimentCycleDetector();

    /** @brief Construct with specified configuration. */
    explicit SentimentCycleDetector(Config cfg);

    /**
     * @brief Record a new bias observation.
     *
     * Appends to the rolling window and, once min_samples is reached,
     * recomputes the ACF and updates dominant_period / cycle_strength.
     * Fires on_period_change if the dominant period shifts.
     */
    void record(double bias) noexcept;

    /** @brief Update configuration; clears window state. */
    void update_config(Config cfg);

    /** @brief Return a copy of the current configuration. */
    [[nodiscard]] Config get_config() const;

    /**
     * @brief Lag (in samples) of the dominant autocorrelation peak.
     *
     * Returns 0 if insufficient samples have been collected.
     * Thread-safe.
     */
    [[nodiscard]] int dominant_period() const noexcept {
        return dominant_period_.load(std::memory_order_relaxed);
    }

    /**
     * @brief Normalized autocorrelation at the dominant period.
     *
     * Range [-1, 1].  Values >= cyclic_threshold indicate a meaningful cycle.
     * Thread-safe.
     */
    [[nodiscard]] double cycle_strength() const noexcept {
        return cycle_strength_.load(std::memory_order_relaxed);
    }

    /**
     * @brief True when cycle_strength() >= config.cyclic_threshold.
     * Thread-safe.
     */
    [[nodiscard]] bool is_cyclic() const noexcept;

    /**
     * @brief Full normalized ACF vector at all lags [1..max_lag].
     *
     * Acquires the mutex — not suitable for hot-path polling.
     */
    [[nodiscard]] std::vector<double> acf_snapshot() const;

    /** @brief Total observations recorded. Thread-safe. */
    [[nodiscard]] uint64_t total_records() const noexcept {
        return total_.load(std::memory_order_relaxed);
    }

    /** @brief Total times the dominant period changed. Thread-safe. */
    [[nodiscard]] uint64_t period_changes() const noexcept {
        return period_changes_.load(std::memory_order_relaxed);
    }

    /** @brief Reset window and counters. */
    void reset();

    /** @brief Serialise current state to a compact JSON string. */
    [[nodiscard]] std::string to_stats_json() const;

private:
    /// Recompute ACF from the current ring buffer; caller must hold mutex_.
    void recompute_locked();

    mutable std::mutex mutex_;
    Config             config_;

    // Rolling ring buffer of bias samples.
    std::vector<double> ring_;
    int                 ring_head_{0};
    int                 ring_fill_{0};

    // Most-recent ACF values (under mutex_)
    std::vector<double> acf_;
    int                 current_period_{0};

    // Lock-free readable outputs
    std::atomic<int>      dominant_period_{0};
    std::atomic<double>   cycle_strength_{0.0};
    std::atomic<double>   cyclic_threshold_cache_{0.30};
    std::atomic<uint64_t> total_{0};
    std::atomic<uint64_t> period_changes_{0};
};

} // namespace llmquant
