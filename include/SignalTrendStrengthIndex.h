#pragma once

/// @file SignalTrendStrengthIndex.h
/// @brief Computes a trend-strength index (TSI) for the bias stream using
///        double-smoothed momentum, analogous to the True Strength Index
///        used in technical analysis.
///
/// ## Motivation
/// The True Strength Index (TSI) applies two EMA smoothing passes to the
/// first difference of the signal:
///
///   momentum_t = x_t − x_{t-1}
///   TSI = 100 * EMA(EMA(momentum, r), s) / EMA(EMA(|momentum|, r), s)
///
/// where r is the "first smoothing" period and s is the "second smoothing"
/// period.  TSI ∈ (−100, +100):
///   - TSI > 0 → net upward (bullish) momentum is dominant.
///   - TSI < 0 → net downward (bearish) momentum dominant.
///   - TSI crossing zero → momentum regime flip.
///   - |TSI| > strength_threshold → strong directional regime.
///
/// Fires `on_zero_cross(old_tsi, new_tsi)` on sign flip.
/// Fires `on_strong(tsi)` when |TSI| > strength_threshold.
///
/// ## Feature flag
/// `LLMQUANT_ENABLE_TREND_STRENGTH_INDEX` (default ON).
///
/// Thread-safe.

#include <atomic>
#include <cstdint>
#include <functional>
#include <mutex>
#include <string>

namespace llmquant {

class SignalTrendStrengthIndex {
public:
    struct Config {
        /// First EMA smoothing period (fast). Default 13.
        int r{13};

        /// Second EMA smoothing period (slow). Default 7.
        int s{7};

        /// Minimum samples before TSI is meaningful. Default 20.
        int min_samples{20};

        /// |TSI| above this fires on_strong. Default 30.0.
        double strength_threshold{30.0};

        /// Fired when TSI changes sign (outside lock).
        std::function<void(double old_tsi, double new_tsi)> on_zero_cross;

        /// Fired when |TSI| > strength_threshold (outside lock).
        std::function<void(double tsi)> on_strong;
    };

    explicit SignalTrendStrengthIndex(Config cfg = Config{});

    // -----------------------------------------------------------------------
    // Data intake
    // -----------------------------------------------------------------------

    void record(double bias);

    // -----------------------------------------------------------------------
    // Accessors (lock-free)
    // -----------------------------------------------------------------------

    /// Current TSI value ∈ (−100, +100).
    [[nodiscard]] double tsi() const noexcept {
        return tsi_.load(std::memory_order_relaxed);
    }

    /// Total zero-cross events.
    [[nodiscard]] uint64_t zero_cross_events() const noexcept {
        return cross_events_.load(std::memory_order_relaxed);
    }

    /// Total strong-momentum events.
    [[nodiscard]] uint64_t strong_events() const noexcept {
        return strong_events_.load(std::memory_order_relaxed);
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

    // Double-smoothed momentum EMA state
    double ema1_m_{0.0};   // EMA_r(momentum)
    double ema2_m_{0.0};   // EMA_s(EMA_r(momentum))
    double ema1_am_{0.0};  // EMA_r(|momentum|)
    double ema2_am_{0.0};  // EMA_s(EMA_r(|momentum|))
    double alpha_r_{0.0};
    double alpha_s_{0.0};
    double prev_bias_{0.0};
    double prev_tsi_{0.0};
    bool   seeded_{false};

    std::atomic<double>   tsi_{0.0};
    std::atomic<uint64_t> cross_events_{0};
    std::atomic<uint64_t> strong_events_{0};
    std::atomic<uint64_t> total_{0};
};

} // namespace llmquant
