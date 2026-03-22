#pragma once

/// @file TokenStreamDifferencer.h
/// @brief Computes first-, second-, and third-order differences of the token
///        weight series, exposing sentiment velocity, acceleration, and jerk.
///
/// ## Motivation
/// Financial time-series analysis routinely differences a series to achieve
/// stationarity.  Applying the same principle to token weights reveals:
/// - **Velocity**     : Δw_t = w_t − w_{t-1}      (rate of bias change)
/// - **Acceleration** : Δ²w_t = Δw_t − Δw_{t-1}   (curvature of bias trend)
/// - **Jerk**         : Δ³w_t = Δ²w_t − Δ²w_{t-1} (rate of acceleration change)
///
/// Jerk is particularly useful: a sudden spike in |jerk| signals an abrupt
/// narrative reversal, which is a stronger early-warning signal than velocity
/// alone because it fires one derivative earlier than the momentum peak.
///
/// Each metric is also EMA-smoothed for noise reduction.
///
/// Fires `on_jerk_spike(raw_jerk, ema_jerk)` when |jerk| exceeds `jerk_threshold`.
///
/// ## Feature flag
/// `LLMQUANT_ENABLE_STREAM_DIFFERENCER` (default ON).
///
/// Thread-safe.

#include <atomic>
#include <cstdint>
#include <functional>
#include <mutex>
#include <string>

namespace llmquant {

class TokenStreamDifferencer {
public:
    struct Config {
        /// EMA alpha for smoothing differences. Default 0.2.
        double ema_alpha{0.2};

        /// |Jerk| threshold to fire the spike callback. Default 0.1.
        double jerk_threshold{0.1};

        /// Hysteresis factor: re-arm after |jerk_ema| < jerk_threshold * (1 - h). Default 0.3.
        double hysteresis{0.3};

        /// Fired when |jerk| exceeds the threshold.
        /// Parameters: (raw_jerk, ema_jerk).
        std::function<void(double raw_jerk, double ema_jerk)> on_jerk_spike;
    };

    explicit TokenStreamDifferencer(Config cfg = Config{});

    // -----------------------------------------------------------------------
    // Data intake
    // -----------------------------------------------------------------------

    /// Record a new token weight and update all difference series.
    void record(double weight);

    // -----------------------------------------------------------------------
    // Accessors (lock-free)
    // -----------------------------------------------------------------------

    /// Raw first difference (velocity) of the last observation.
    [[nodiscard]] double velocity() const noexcept {
        return velocity_.load(std::memory_order_relaxed);
    }

    /// EMA-smoothed first difference.
    [[nodiscard]] double velocity_ema() const noexcept {
        return velocity_ema_.load(std::memory_order_relaxed);
    }

    /// Raw second difference (acceleration).
    [[nodiscard]] double acceleration() const noexcept {
        return acceleration_.load(std::memory_order_relaxed);
    }

    /// EMA-smoothed second difference.
    [[nodiscard]] double acceleration_ema() const noexcept {
        return acceleration_ema_.load(std::memory_order_relaxed);
    }

    /// Raw third difference (jerk).
    [[nodiscard]] double jerk() const noexcept {
        return jerk_.load(std::memory_order_relaxed);
    }

    /// EMA-smoothed third difference.
    [[nodiscard]] double jerk_ema() const noexcept {
        return jerk_ema_.load(std::memory_order_relaxed);
    }

    /// Total observations recorded.
    [[nodiscard]] uint64_t observation_count() const noexcept {
        return obs_count_.load(std::memory_order_relaxed);
    }

    /// Total jerk spikes fired.
    [[nodiscard]] uint64_t jerk_spike_count() const noexcept {
        return spike_count_.load(std::memory_order_relaxed);
    }

    [[nodiscard]] std::string to_stats_json() const;

    void reset();
    void update_config(const Config& cfg);

private:
    mutable std::mutex mutex_;
    Config cfg_;

    double w_prev_{0.0};
    double d1_prev_{0.0};
    double d2_prev_{0.0};

    double vel_ema_val_{0.0};
    double acc_ema_val_{0.0};
    double jerk_ema_val_{0.0};

    int  fill_{0};     // 0→1→2→3 (need 3 steps for jerk)
    bool armed_{true};

    std::atomic<double>   velocity_{0.0};
    std::atomic<double>   velocity_ema_{0.0};
    std::atomic<double>   acceleration_{0.0};
    std::atomic<double>   acceleration_ema_{0.0};
    std::atomic<double>   jerk_{0.0};
    std::atomic<double>   jerk_ema_{0.0};
    std::atomic<uint64_t> obs_count_{0};
    std::atomic<uint64_t> spike_count_{0};
};

} // namespace llmquant
