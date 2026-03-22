#pragma once

/// @file BiasKalmanFilter.h
/// @brief 1-D Kalman filter for smoothing and predicting the LLM bias stream,
///        with online noise covariance estimation via innovation monitoring.
///
/// ## Motivation
/// The raw bias stream contains observation noise.  A Kalman filter separates
/// the true latent bias (state) from noise, providing:
///   - Smoothed bias estimate x̂_t (optimal linear estimate of true bias).
///   - Innovation sequence ν_t = z_t - x̂_{t|t-1} (unexpected information).
///   - Innovation variance S_t (expected surprise magnitude).
///   - Normalised innovation squared (NIS) = ν²/S — a χ²(1) statistic under
///     correct filter tuning.  NIS > chi2_threshold signals model mismatch
///     (regime change, fat tails, structural break).
///
/// ## Model
/// State transition:  x_t = x_{t-1} + w_t,  w_t ~ N(0, Q)  (random walk)
/// Observation:       z_t = x_t + v_t,       v_t ~ N(0, R)
/// Q and R are tunable; R can be estimated online from innovation variance.
///
/// ## Feature flag
/// `LLMQUANT_ENABLE_KALMAN_FILTER` (default ON).
///
/// Thread-safe.

#include <atomic>
#include <cstdint>
#include <functional>
#include <mutex>
#include <string>

namespace llmquant {

class BiasKalmanFilter {
public:
    struct Config {
        /// Process noise variance Q (state diffusion). Default 0.001.
        double Q{0.001};

        /// Initial observation noise variance R. Default 0.1.
        double R{0.1};

        /// Estimate R online from rolling innovation variance if true. Default true.
        bool   estimate_R{true};

        /// EMA period for online R estimation. Default 20.
        int    R_estimation_period{20};

        /// Initial state estimate. Default 0.0.
        double x0{0.0};

        /// Initial state variance P0. Default 1.0.
        double P0{1.0};

        /// NIS threshold (χ²(1) 95% critical value ≈ 3.84). Default 3.84.
        double nis_threshold{3.84};

        /// Fired when NIS > nis_threshold (outside lock). Param: NIS value.
        std::function<void(double nis)> on_model_mismatch;
    };

    explicit BiasKalmanFilter(Config cfg = Config{});

    // -----------------------------------------------------------------------
    // Data intake
    // -----------------------------------------------------------------------

    void record(double bias);

    // -----------------------------------------------------------------------
    // Accessors (lock-free)
    // -----------------------------------------------------------------------

    /// Kalman-smoothed state estimate x̂_t.
    [[nodiscard]] double filtered_bias() const noexcept {
        return x_hat_.load(std::memory_order_relaxed);
    }

    /// Innovation ν_t = z_t − x̂_{t|t-1}.
    [[nodiscard]] double innovation() const noexcept {
        return innovation_.load(std::memory_order_relaxed);
    }

    /// Normalised Innovation Squared ν²/S.
    [[nodiscard]] double nis() const noexcept {
        return nis_.load(std::memory_order_relaxed);
    }

    /// Current Kalman gain K_t.
    [[nodiscard]] double kalman_gain() const noexcept {
        return K_.load(std::memory_order_relaxed);
    }

    /// Total model-mismatch events fired.
    [[nodiscard]] uint64_t mismatch_events() const noexcept {
        return mismatch_events_.load(std::memory_order_relaxed);
    }

    /// Total observations recorded.
    [[nodiscard]] uint64_t total_records() const noexcept {
        return total_.load(std::memory_order_relaxed);
    }

    [[nodiscard]] std::string to_stats_json() const;

    void reset();
    void update_config(const Config& cfg);

private:
    mutable std::mutex mutex_;
    Config cfg_;

    // Filter state
    double x_hat_{0.0};   // state estimate
    double P_{1.0};       // state variance estimate
    double R_est_{0.1};   // online R estimate
    double innov_ema_{0.0}; // EMA of ν² for R estimation

    std::atomic<double>   x_hat_a_{0.0};
    std::atomic<double>   innovation_{0.0};
    std::atomic<double>   nis_{0.0};
    std::atomic<double>   K_{0.0};
    std::atomic<uint64_t> mismatch_events_{0};
    std::atomic<uint64_t> total_{0};
};

} // namespace llmquant
