#pragma once

#include <atomic>
#include <cstdint>
#include <deque>
#include <functional>
#include <mutex>
#include <string>

namespace llmquant {

/**
 * @brief Online Z-score anomaly detector for the bias/sentiment stream.
 *
 * Maintains a rolling window of the most recent `window_size` values.
 * After each new observation the Z-score is computed:
 *
 *   z = (x - μ) / σ    where μ, σ are the rolling mean and standard deviation.
 *
 * When |z| exceeds configured thresholds the detector fires the appropriate
 * callback.  Two tiers allow callers to distinguish soft anomalies (worth
 * logging) from hard anomalies (worth gating signals or alerting operations).
 *
 * ## Intended use
 * - Feed `signal.delta_bias_shift` to detect sudden sentiment spikes.
 * - Feed `entropy` from TokenEntropyMeter to detect token-diversity bursts.
 * - Feed `conditional_vol` from VolatilityForecaster to detect vol-of-vol events.
 *
 * ## Feature flag
 * Controlled by `LLMQUANT_ENABLE_ANOMALY_DETECTOR` (default ON).
 */
class AnomalyDetector {
public:
    /**
     * @brief Anomaly severity tier.
     */
    enum class Severity { Soft, Hard };

    /**
     * @brief A detected anomaly event.
     */
    struct AnomalyEvent {
        double   value;      ///< The anomalous observation.
        double   z_score;    ///< Computed Z-score.
        double   mean;       ///< Rolling mean at time of detection.
        double   stddev;     ///< Rolling standard deviation.
        Severity severity;   ///< Tier (Soft or Hard).
    };

    /**
     * @brief Callback type for anomaly events.
     *
     * Called from the thread invoking record().  Must not block.
     */
    using AnomalyCallback = std::function<void(const AnomalyEvent&)>;

    /**
     * @brief Configuration.
     */
    struct Config {
        /**
         * @brief Size of the rolling window for mean/stddev estimation.
         *
         * Default 50.
         */
        int window_size{50};

        /**
         * @brief Minimum number of observations before anomaly detection begins.
         *
         * Prevents false positives during warm-up.  Default 10.
         */
        int min_samples{10};

        /**
         * @brief Z-score threshold for a "soft" anomaly (e.g. 2σ event).
         *
         * Default 2.0.
         */
        double soft_threshold{2.0};

        /**
         * @brief Z-score threshold for a "hard" anomaly (e.g. 3σ event).
         *
         * Default 3.0.
         */
        double hard_threshold{3.0};

        /**
         * @brief Callback invoked for soft anomalies (|z| ≥ soft_threshold).
         *
         * Not called when hard_threshold is also exceeded (hard_cb handles that).
         */
        AnomalyCallback soft_cb;

        /**
         * @brief Callback invoked for hard anomalies (|z| ≥ hard_threshold).
         */
        AnomalyCallback hard_cb;

        /**
         * @brief Human-readable name for this detector (used in log messages).
         *
         * Default "bias".
         */
        std::string name{"bias"};
    };

    // -----------------------------------------------------------------------
    // Construction
    // -----------------------------------------------------------------------

    explicit AnomalyDetector(Config config = Config{});

    // -----------------------------------------------------------------------
    // Recording
    // -----------------------------------------------------------------------

    /**
     * @brief Record a new observation and check for anomalies.
     *
     * Updates the rolling window, computes the Z-score, and fires the
     * appropriate callback if a threshold is exceeded.  Thread-safe.
     *
     * @param value  The scalar observation (bias, entropy, vol, etc.).
     * @return       The computed Z-score (NaN if fewer than min_samples seen).
     */
    double record(double value);

    /**
     * @brief Clear the rolling window and reset counters.
     *
     * Thread-safe.
     */
    void reset();

    // -----------------------------------------------------------------------
    // Query
    // -----------------------------------------------------------------------

    /**
     * @brief Most recent computed Z-score.
     *
     * Returns 0 if fewer than min_samples have been recorded.
     * Thread-safe.
     */
    [[nodiscard]] double last_z_score() const noexcept;

    /**
     * @brief Rolling mean over the current window.
     *
     * Thread-safe.
     */
    [[nodiscard]] double rolling_mean() const noexcept;

    /**
     * @brief Rolling standard deviation over the current window.
     *
     * Thread-safe.
     */
    [[nodiscard]] double rolling_stddev() const noexcept;

    /**
     * @brief Number of samples currently in the window.
     *
     * Thread-safe.
     */
    [[nodiscard]] int sample_count() const noexcept;

    // -----------------------------------------------------------------------
    // Statistics
    // -----------------------------------------------------------------------

    /** @brief Total values recorded. */
    [[nodiscard]] uint64_t total_records() const noexcept {
        return total_records_.load(std::memory_order_relaxed);
    }

    /** @brief Soft anomaly events fired. */
    [[nodiscard]] uint64_t soft_anomalies() const noexcept {
        return soft_anomalies_.load(std::memory_order_relaxed);
    }

    /** @brief Hard anomaly events fired. */
    [[nodiscard]] uint64_t hard_anomalies() const noexcept {
        return hard_anomalies_.load(std::memory_order_relaxed);
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
    std::deque<double> window_;   // rolling observations
    double             sum_{0.0}; // running sum (for mean)
    double             sum2_{0.0};// running sum of squares (for variance)
    double             last_z_{0.0};

    std::atomic<uint64_t> total_records_{0};
    std::atomic<uint64_t> soft_anomalies_{0};
    std::atomic<uint64_t> hard_anomalies_{0};

    // Compute z-score; window must have ≥ 2 entries.  Called under lock.
    [[nodiscard]] static double compute_z(double value, double mean,
                                           double stddev) noexcept;
};

} // namespace llmquant
