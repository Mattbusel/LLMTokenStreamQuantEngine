#pragma once

#include "TradeSignalEngine.h"

#include <atomic>
#include <cstdint>
#include <functional>
#include <mutex>
#include <string>
#include <vector>

namespace llmquant {

/**
 * @brief Online Platt-scaling calibration for raw signal confidence scores.
 *
 * Raw TradeSignal confidence values are model outputs that may be poorly
 * calibrated (e.g., systematically high during trending regimes, low during
 * chop).  This engine maintains a rolling history of (raw_score, outcome)
 * pairs and fits a logistic (sigmoid) calibration: P(win) = σ(A·x + B),
 * where A and B are updated incrementally via gradient descent.
 *
 * ## What it provides
 * - `calibrate(raw_score)` → P(win | raw_score) in [0, 1]
 * - `expected_calibration_error()` → ECE over the current window
 * - `reliability_diagram_bins()` → mean predicted P vs mean actual outcome
 *   per confidence bucket, for plotting reliability diagrams
 *
 * ## Feature flag
 * Controlled by `LLMQUANT_ENABLE_SIGNAL_CALIBRATION` (default ON).
 *
 * ## Typical usage
 * @code
 * SignalCalibrationEngine cal;
 * // After each trade resolves:
 * cal.record_outcome(signal.confidence, trade_was_profitable);
 * // Before sizing a new position:
 * double p_win = cal.calibrate(signal.confidence);
 * double kelly_f = 2.0 * p_win - 1.0;  // half-Kelly fraction
 * @endcode
 *
 * Thread safety: all public methods are thread-safe.
 */
class SignalCalibrationEngine {
public:
    // -----------------------------------------------------------------------
    // Configuration
    // -----------------------------------------------------------------------

    struct Config {
        /**
         * @brief Rolling window size (number of (score, outcome) samples to retain).
         *
         * Older samples are evicted when the window is full.
         * Default 256.
         */
        int window_size{256};

        /**
         * @brief Gradient-descent learning rate for Platt parameter updates.
         *
         * Smaller values make calibration stable but slower to adapt.
         * Default 0.01.
         */
        double learning_rate{0.01};

        /**
         * @brief Minimum number of samples before calibration is considered valid.
         *
         * Below this count, `calibrate()` returns the raw score unchanged.
         * Default 20.
         */
        int min_samples{20};

        /**
         * @brief Number of bins for the reliability diagram and ECE computation.
         *
         * Default 10.
         */
        int n_bins{10};

        /**
         * @brief L2 regularisation strength for Platt A and B parameters.
         *
         * Prevents A from growing arbitrarily large on small datasets.
         * Default 0.01.
         */
        double l2_reg{0.01};

        /**
         * @brief Callback invoked after each `record_outcome()` with the
         *        current calibrated ECE.
         */
        std::function<void(double ece)> on_recalibrated;
    };

    // -----------------------------------------------------------------------
    // Reliability-diagram bin (one per confidence bucket)
    // -----------------------------------------------------------------------

    struct ReliabilityBin {
        double bin_low{0.0};    ///< Lower bound of raw-score bucket.
        double bin_high{0.0};   ///< Upper bound.
        double mean_predicted{0.0}; ///< Mean calibrate(score) for samples in bin.
        double mean_actual{0.0};    ///< Fraction of samples in bin with outcome=true.
        int    count{0};            ///< Number of samples falling in bin.
    };

    // -----------------------------------------------------------------------
    // Construction
    // -----------------------------------------------------------------------

    explicit SignalCalibrationEngine(Config config = Config{});

    // -----------------------------------------------------------------------
    // Online learning
    // -----------------------------------------------------------------------

    /**
     * @brief Record a (raw_score, outcome) observation and update calibration.
     *
     * Performs one step of gradient descent on the logistic loss:
     *   L = -[y·log(σ) + (1-y)·log(1-σ)]   where σ = σ(A·score + B).
     *
     * @param raw_score  Raw signal confidence value (any range; typically [0,1]).
     * @param positive_outcome true if the signal led to a profitable outcome.
     */
    void record_outcome(double raw_score, bool positive_outcome);

    // -----------------------------------------------------------------------
    // Inference
    // -----------------------------------------------------------------------

    /**
     * @brief Map a raw score to a calibrated probability P(win | score).
     *
     * Returns the raw score when fewer than min_samples have been recorded
     * (identity calibration).
     *
     * @param raw_score Raw signal confidence (typically in [0,1]).
     * @return Calibrated probability in (0, 1).
     */
    [[nodiscard]] double calibrate(double raw_score) const noexcept;

    // -----------------------------------------------------------------------
    // Diagnostics
    // -----------------------------------------------------------------------

    /**
     * @brief Expected calibration error (ECE) over the current window.
     *
     * ECE = Σ (|count_b| / N) · |mean_predicted_b − mean_actual_b|
     *
     * Range [0, 1]; 0 = perfectly calibrated.
     * Returns 0 when fewer than min_samples have been seen.
     */
    [[nodiscard]] double expected_calibration_error() const;

    /**
     * @brief Return per-bin reliability data for plotting a reliability diagram.
     *
     * Bins samples by their RAW score (before calibration) into n_bins equal
     * buckets over [0, 1].  The mean calibrated prediction and mean actual
     * outcome per bucket reveal systematic over/under confidence.
     */
    [[nodiscard]] std::vector<ReliabilityBin> reliability_diagram_bins() const;

    /**
     * @brief Current Platt scaling parameter A (scale).
     */
    [[nodiscard]] double platt_a() const noexcept;

    /**
     * @brief Current Platt scaling parameter B (bias/shift).
     */
    [[nodiscard]] double platt_b() const noexcept;

    /**
     * @brief Number of (score, outcome) samples in the current window.
     */
    [[nodiscard]] uint64_t sample_count() const noexcept {
        return sample_count_.load(std::memory_order_relaxed);
    }

    /**
     * @brief Total observations recorded (including those evicted from window).
     */
    [[nodiscard]] uint64_t total_recorded() const noexcept {
        return total_recorded_.load(std::memory_order_relaxed);
    }

    // -----------------------------------------------------------------------
    // Config management
    // -----------------------------------------------------------------------

    /**
     * @brief Update configuration and reset calibration state.
     */
    void update_config(const Config& config);

    /**
     * @brief Get current configuration.
     */
    [[nodiscard]] Config get_config() const;

    /**
     * @brief Reset calibration parameters and clear history.
     */
    void reset();

    /**
     * @brief JSON diagnostics snapshot.
     */
    [[nodiscard]] std::string to_stats_json() const;

private:
    Config config_;

    // Rolling window of observations.
    struct Sample { double score; bool outcome; };
    std::vector<Sample> window_;
    int  ring_head_{0};
    int  ring_fill_{0};

    // Platt parameters A (scale) and B (intercept).
    double platt_a_{1.0};
    double platt_b_{0.0};

    std::atomic<uint64_t> sample_count_{0};
    std::atomic<uint64_t> total_recorded_{0};

    mutable std::mutex mutex_;

    static double sigmoid(double x) noexcept;
    void gradient_step_locked(double score, double target);
};

} // namespace llmquant
