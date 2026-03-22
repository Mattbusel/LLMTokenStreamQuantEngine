#pragma once

/// @file SignalSlopeMeter.h
/// @brief OLS linear regression slope over a rolling window of bias values.
///
/// ## Motivation
/// The raw bias value tells you *where* sentiment is, not *how fast it is
/// moving*.  The derivative (slope) reveals acceleration: a rising slope on
/// a bullish regime confirms conviction; a falling slope warns of exhaustion.
///
/// ## Algorithm
/// Maintains a circular buffer of the last `window` bias observations.
/// On each `record(bias)`, recomputes the OLS slope:
///
///   slope = (n·Σxy - Σx·Σy) / (n·Σx² - (Σx)²)
///
/// where x = 0,1,...,n-1 (time index) and y = bias values.
///
/// slope_score ∈ [-1, 1]:
///   slope_score = clamp(slope / saturation_slope, -1, 1)
///
/// Fires `on_trend_acceleration(slope)` when |slope| first crosses
/// `acceleration_threshold` (rising edge only).
///
/// ## Feature flag
/// `LLMQUANT_ENABLE_SIGNAL_SLOPE` (default ON).
///
/// Thread-safe.

#include <atomic>
#include <cstdint>
#include <functional>
#include <mutex>
#include <string>
#include <vector>

namespace llmquant {

class SignalSlopeMeter {
public:
    struct Config {
        /// Rolling window size. Default 20.
        int window{20};

        /// |slope| above which on_trend_acceleration fires. Default 0.02.
        double acceleration_threshold{0.02};

        /// slope value that maps to slope_score = ±1.0. Default 0.05.
        double saturation_slope{0.05};

        /// Fired when |slope| first crosses acceleration_threshold (outside lock).
        std::function<void(double slope)> on_trend_acceleration;
    };

    explicit SignalSlopeMeter(Config cfg = Config{});

    // -----------------------------------------------------------------------
    // Data intake
    // -----------------------------------------------------------------------

    /// Record a new bias observation.
    void record(double bias);

    // -----------------------------------------------------------------------
    // Accessors (lock-free)
    // -----------------------------------------------------------------------

    /// Raw OLS slope (units: bias per observation step).
    [[nodiscard]] double slope() const noexcept {
        return slope_.load(std::memory_order_relaxed);
    }

    /// Normalized slope score ∈ [-1, 1].
    [[nodiscard]] double slope_score() const noexcept {
        return slope_score_.load(std::memory_order_relaxed);
    }

    /// True when |slope| > acceleration_threshold.
    [[nodiscard]] bool is_accelerating() const noexcept {
        return accelerating_.load(std::memory_order_relaxed);
    }

    /// Total observations recorded.
    [[nodiscard]] uint64_t observation_count() const noexcept {
        return obs_count_.load(std::memory_order_relaxed);
    }

    [[nodiscard]] std::string to_stats_json() const;

    void reset();
    void update_config(const Config& cfg);

private:
    mutable std::mutex mutex_;
    Config cfg_;

    std::vector<double> buf_;
    int head_{0};
    int fill_{0};

    bool prev_accelerating_{false};

    std::atomic<double>   slope_{0.0};
    std::atomic<double>   slope_score_{0.0};
    std::atomic<bool>     accelerating_{false};
    std::atomic<uint64_t> obs_count_{0};
};

} // namespace llmquant
