#pragma once

#include <atomic>
#include <cstdint>
#include <mutex>
#include <string>

namespace llmquant {

/**
 * @brief 1-D Kalman filter for smoothing noisy signal outputs.
 *
 * Models the signal as a latent scalar state that evolves according to a
 * random walk (process noise Q) and is observed with additive Gaussian
 * measurement noise (observation noise R).
 *
 * ## State equations
 * ```
 * x_k|k-1 = x_k-1|k-1                  (prediction — constant model)
 * P_k|k-1 = P_k-1|k-1 + Q              (predict covariance)
 *
 * K_k = P_k|k-1 / (P_k|k-1 + R)        (Kalman gain)
 * x_k|k = x_k|k-1 + K_k * (z_k - x_k|k-1)  (update state)
 * P_k|k = (1 - K_k) * P_k|k-1          (update covariance)
 * ```
 *
 * ## Why this matters for trade signals
 * Raw LLM confidence scores are noisy: consecutive tokens for the same
 * underlying narrative can swing ±0.2 purely from sampling variance.
 * Smoothing via Kalman reduces false triggers without adding a fixed lag
 * (unlike a simple moving average), because the filter automatically
 * tightens when recent observations are consistent.
 *
 * ## Usage
 * ```cpp
 * KalmanFilter kf;
 * double smoothed = kf.update(raw_signal);
 * ```
 *
 * ## Thread safety
 * All methods are thread-safe.
 *
 * ## Feature flag
 * Controlled by `LLMQUANT_ENABLE_KALMAN_FILTER` (default ON).
 */
class KalmanFilter {
public:
    /**
     * @brief Configuration parameters.
     */
    struct Config {
        /**
         * @brief Process noise covariance Q.
         *
         * Higher values make the filter track rapid state changes faster
         * (less smoothing); lower values give a smoother but laggier output.
         * Default: 1e-4.
         */
        double process_noise{1e-4};

        /**
         * @brief Observation noise covariance R.
         *
         * Higher values indicate noisier measurements and produce stronger
         * smoothing. Default: 1e-2.
         */
        double observation_noise{1e-2};

        /**
         * @brief Initial state estimate x_0.
         *
         * Set to the first expected signal value if known; otherwise 0.0.
         * Default: 0.0.
         */
        double initial_state{0.0};

        /**
         * @brief Initial error covariance P_0.
         *
         * Large value signals high uncertainty about the initial state.
         * Default: 1.0.
         */
        double initial_covariance{1.0};
    };

    explicit KalmanFilter(Config cfg = Config{});

    // Non-copyable.
    KalmanFilter(const KalmanFilter&)            = delete;
    KalmanFilter& operator=(const KalmanFilter&) = delete;

    /**
     * @brief Feed a new noisy observation and return the smoothed estimate.
     *
     * Thread-safe.
     *
     * @param observation  Raw (noisy) signal value.
     * @return Kalman-filtered state estimate.
     */
    [[nodiscard]] double update(double observation);

    /**
     * @brief Return the current state estimate without updating.
     *
     * Thread-safe.
     */
    [[nodiscard]] double state() const noexcept;

    /**
     * @brief Return the current error covariance (uncertainty estimate).
     *
     * Thread-safe.
     */
    [[nodiscard]] double covariance() const noexcept;

    /**
     * @brief Return the most recent Kalman gain (0.0–1.0).
     *
     * A gain near 1.0 means the filter is trusting observations heavily;
     * near 0.0 means it is relying on the prior prediction.
     *
     * Thread-safe.
     */
    [[nodiscard]] double last_gain() const noexcept;

    /**
     * @brief Total number of update() calls since construction or reset().
     *
     * Thread-safe.
     */
    [[nodiscard]] uint64_t total_updates() const noexcept {
        return updates_.load(std::memory_order_relaxed);
    }

    /**
     * @brief Reset state to the initial conditions specified in Config.
     *
     * Thread-safe.
     */
    void reset();

    /**
     * @brief Update configuration and reset state.
     *
     * Thread-safe.
     */
    void update_config(const Config& cfg);

    /**
     * @brief Serialise current state to a JSON object string.
     *
     * Fields: state, covariance, last_gain, process_noise, observation_noise,
     * total_updates.
     */
    [[nodiscard]] std::string to_stats_json() const;

private:
    mutable std::mutex mtx_;
    Config cfg_;

    double state_{0.0};
    double covariance_{1.0};
    double last_gain_{0.0};

    std::atomic<uint64_t> updates_{0};
};

} // namespace llmquant
