#pragma once

#include <atomic>
#include <cstdint>
#include <mutex>
#include <string>
#include <vector>

namespace llmquant {

/**
 * @brief Real-time sentiment trend detector using a fixed-size sliding window.
 *
 * Ingests per-token sentiment scores and fits an online least-squares linear
 * regression over the most recent N samples.  The regression slope is used to
 * classify the sentiment stream as Improving, Declining, Stable, or Volatile.
 *
 * ## Trend classification
 * | Trend     | Condition                                                         |
 * |-----------|-------------------------------------------------------------------|
 * | Volatile  | residual_std_dev > volatility_threshold                          |
 * | Improving | slope > trend_threshold (and not Volatile)                       |
 * | Declining | slope < -trend_threshold (and not Volatile)                     |
 * | Stable    | |slope| <= trend_threshold (and not Volatile)                   |
 *
 * The window is a fixed-size circular buffer; samples outside the window are
 * automatically dropped.  All accessors are thread-safe via a shared mutex.
 *
 * ## Feature flag
 * Controlled by `LLMQUANT_ENABLE_SENTIMENT_TRAJECTORY` (default ON).
 * When the flag is OFF this header is still present but the implementation
 * stubs do nothing and always return Stable / 0.0.
 *
 * ## Typical usage
 * @code
 * SentimentTrajectoryAnalyzer traj;
 * for (auto& weight : weights) {
 *     traj.add_sample(weight.sentiment_score);
 * }
 * if (traj.get_trend() == SentimentTrend::Improving) { ... }
 * @endcode
 *
 * Thread safety: add_sample() and all accessors are safe to call from any
 * thread concurrently.
 */

/**
 * @brief Sentiment momentum direction over the recent token window.
 */
enum class SentimentTrend {
    Improving, ///< Positive slope > trend_threshold; sentiment is rising.
    Declining, ///< Negative slope < -trend_threshold; sentiment is falling.
    Stable,    ///< Slope magnitude <= trend_threshold; no directional trend.
    Volatile,  ///< Residual standard deviation > volatility_threshold; noisy signal.
    Undefined, ///< Fewer samples than min_samples_for_trend; not yet classifiable.
};

class SentimentTrajectoryAnalyzer {
public:
    /**
     * @brief Configuration for the trajectory analyser.
     */
    struct Config {
        /**
         * @brief Number of most-recent token scores kept in the sliding window.
         *
         * Larger values give a smoother (but laggier) trend estimate.
         * Smaller values react faster but produce noisier classifications.
         * Must be >= 2.
         */
        int window_size{20};

        /**
         * @brief Minimum regression slope (per-sample units) to classify
         *        the stream as Improving or Declining rather than Stable.
         *
         * A value of 0.01 means the fitted line must rise or fall by at least
         * 0.01 sentiment units per token to be classified as directional.
         */
        double trend_threshold{0.01};

        /**
         * @brief Maximum residual standard deviation before the stream is
         *        classified as Volatile rather than applying slope classification.
         *
         * When the residuals of the regression are large, a slope value is
         * unreliable, so Volatile takes priority over Improving/Declining.
         */
        double volatility_threshold{0.15};

        /**
         * @brief Minimum number of samples before any trend classification is
         *        returned; below this count get_trend() returns Undefined.
         *
         * Must be >= 2 and <= window_size.
         */
        int min_samples_for_trend{5};
    };

    /**
     * @brief Construct the analyser with the given configuration.
     *
     * @param config Window size, trend threshold, and volatility threshold.
     */
    explicit SentimentTrajectoryAnalyzer(Config config = Config{});

    /**
     * @brief Add a new sentiment score sample to the sliding window.
     *
     * Old samples falling outside window_size are automatically evicted.
     * Thread-safe.
     *
     * @param sentiment_score Raw sentiment score, typically in [-1.0, 1.0].
     */
    void add_sample(double sentiment_score);

    /**
     * @brief Return the current trend classification.
     *
     * Returns SentimentTrend::Undefined if fewer than min_samples_for_trend
     * samples have been added since construction or last reset().
     * Thread-safe.
     *
     * @return Trend enum value.
     */
    [[nodiscard]] SentimentTrend get_trend() const;

    /**
     * @brief Return the least-squares regression slope across the current window.
     *
     * Positive slope means sentiment is improving over time; negative means
     * declining.  Returns 0.0 if fewer than 2 samples are in the window.
     * Thread-safe.
     *
     * @return Regression slope in sentiment-units-per-sample.
     */
    [[nodiscard]] double get_slope() const;

    /**
     * @brief Return the coefficient of determination (R²) for the current window.
     *
     * R² in [0.0, 1.0]: values close to 1.0 mean the linear model fits well;
     * values close to 0.0 mean the signal is noisy or non-linear.
     * Returns 0.0 if fewer than 2 samples are in the window.
     * Thread-safe.
     *
     * @return R² in [0.0, 1.0].
     */
    [[nodiscard]] double get_r_squared() const;

    /**
     * @brief Return the residual standard deviation of the current regression fit.
     *
     * A low value indicates the linear model fits well; a high value indicates
     * volatility or non-linear behaviour.  Used internally to classify Volatile.
     * Returns 0.0 if fewer than 2 samples are in the window.
     * Thread-safe.
     *
     * @return Residual standard deviation in the same units as sentiment_score.
     */
    [[nodiscard]] double get_residual_std() const;

    /**
     * @brief Return the simple arithmetic mean of the current window.
     *
     * Returns 0.0 if no samples are in the window.
     * Thread-safe.
     *
     * @return Mean sentiment score of the current window.
     */
    [[nodiscard]] double get_mean() const;

    /**
     * @brief Return the number of samples currently in the sliding window.
     *
     * In [0, window_size].  Thread-safe.
     *
     * @return Current sample count.
     */
    [[nodiscard]] int get_sample_count() const;

    /**
     * @brief Return the total number of samples added since construction or reset().
     *
     * Monotonically increasing; always >= get_sample_count().
     * Thread-safe (atomic read).
     *
     * @return Cumulative sample count.
     */
    [[nodiscard]] uint64_t get_total_samples() const noexcept {
        return total_samples_.load(std::memory_order_relaxed);
    }

    /**
     * @brief Return a human-readable label for a SentimentTrend value.
     *
     * @param trend Trend enum value.
     * @return "Improving", "Declining", "Stable", "Volatile", or "Undefined".
     */
    [[nodiscard]] static const char* trend_name(SentimentTrend trend) noexcept;

    /**
     * @brief Serialise the current state to a compact JSON string.
     *
     * Fields: trend (string), slope, r_squared, residual_std, mean,
     * sample_count, total_samples.
     * Thread-safe.
     *
     * @return JSON object string.
     */
    [[nodiscard]] std::string to_stats_json() const;

    /**
     * @brief Reset the sliding window and all statistics.
     *
     * Thread-safe.
     */
    void reset();

private:
    /**
     * @brief Compute regression statistics from the current window.
     *
     * Fills slope, r_squared, and residual_std.
     * Caller must hold window_mutex_.
     */
    struct RegressionResult {
        double slope{0.0};
        double r_squared{0.0};
        double residual_std{0.0};
        double mean{0.0};
    };
    [[nodiscard]] RegressionResult compute_regression_locked() const;

    Config config_;
    mutable std::mutex window_mutex_;
    std::vector<double> window_;   ///< Circular buffer (newest at back).
    std::atomic<uint64_t> total_samples_{0};
};

} // namespace llmquant
