#pragma once

#include <atomic>
#include <cstdint>
#include <deque>
#include <mutex>
#include <string>

namespace llmquant {

/**
 * @brief Platt-scaling confidence calibrator with online outcome feedback.
 *
 * Most confidence estimators are over- or under-confident.  The
 * ConfidenceCalibrator learns a monotonic mapping from raw confidence
 * to empirical win probability by recording (predicted, outcome) pairs
 * and fitting a simple logistic regression:
 *
 *   calibrated(x) = sigmoid(a * x + b)
 *
 * Parameters a and b are updated after each outcome via stochastic gradient
 * descent (learning_rate), providing an online calibration that adapts as
 * market regimes shift.
 *
 * ## Feedback loop
 * Wire PositionTracker's trade-close callback:
 * ```cpp
 * tracker.set_trade_close_callback([&](uint64_t, double, bool won) {
 *     calibrator.record_outcome(last_confidence, won);
 * });
 * ```
 * Then in the signal path:
 * ```cpp
 * signal.confidence = calibrator.calibrate(signal.confidence);
 * ```
 *
 * ## Convergence
 * Requires at least `min_samples` outcomes before calibration is active.
 * Before that, raw confidence is passed through unchanged.
 *
 * ## Thread safety
 * All methods are thread-safe.
 *
 * ## Feature flag
 * Controlled by `LLMQUANT_ENABLE_CONFIDENCE_CALIBRATOR` (default ON).
 */
class ConfidenceCalibrator {
public:
    /**
     * @brief Configuration.
     */
    struct Config {
        /**
         * @brief SGD learning rate for Platt scaling parameters.
         * Default 0.01.
         */
        double learning_rate{0.01};

        /**
         * @brief Minimum number of outcomes before calibration activates.
         * Below this threshold, raw confidence is returned unchanged.
         * Default 20.
         */
        int min_samples{20};

        /**
         * @brief Sliding window size for outcome history.
         * Older outcomes are discarded to allow regime adaptation.
         * Default 200.
         */
        int window_size{200};
    };

    explicit ConfidenceCalibrator(Config cfg = Config{});

    // Non-copyable.
    ConfidenceCalibrator(const ConfidenceCalibrator&)            = delete;
    ConfidenceCalibrator& operator=(const ConfidenceCalibrator&) = delete;

    /**
     * @brief Apply the learned calibration to a raw confidence score.
     *
     * Returns the raw value unchanged if fewer than `min_samples` outcomes
     * have been recorded.  Thread-safe.
     *
     * @param raw_confidence Raw confidence in [0.0, 1.0].
     * @return Calibrated confidence in [0.0, 1.0].
     */
    [[nodiscard]] double calibrate(double raw_confidence) const;

    /**
     * @brief Record a trade outcome and update calibration parameters.
     *
     * Thread-safe.
     *
     * @param predicted_confidence The confidence that was used for this trade.
     * @param won                  true if the trade was a win.
     */
    void record_outcome(double predicted_confidence, bool won);

    /**
     * @brief Total outcomes recorded.
     */
    [[nodiscard]] uint64_t total_outcomes() const noexcept {
        return total_outcomes_.load(std::memory_order_relaxed);
    }

    /**
     * @brief True once at least `min_samples` outcomes have been seen.
     */
    [[nodiscard]] bool is_calibrated() const noexcept;

    /**
     * @brief Empirical win rate across the current outcome window.
     */
    [[nodiscard]] double empirical_win_rate() const;

    /**
     * @brief Platt scaling parameters (a, b) as currently fitted.
     *
     * Returns {1.0, 0.0} before calibration is active (identity transform).
     */
    [[nodiscard]] std::pair<double, double> platt_params() const;

    /**
     * @brief Reset all outcome history and return to pass-through mode.
     *
     * Thread-safe.
     */
    void reset();

    /**
     * @brief Update configuration (resets calibration state).
     *
     * Thread-safe.
     */
    void update_config(const Config& cfg);

    /**
     * @brief Serialise current state to a JSON object string.
     */
    [[nodiscard]] std::string to_stats_json() const;

private:
    [[nodiscard]] static double sigmoid(double x) noexcept;
    void sgd_step_locked(double predicted, bool won);

    Config cfg_;
    mutable std::mutex mtx_;

    double platt_a_{1.0};  // slope
    double platt_b_{0.0};  // intercept

    struct OutcomeRecord {
        double confidence;
        bool   won;
    };
    std::deque<OutcomeRecord> history_;

    std::atomic<uint64_t> total_outcomes_{0};
};

} // namespace llmquant
