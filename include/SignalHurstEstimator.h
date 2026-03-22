#pragma once

/// @file SignalHurstEstimator.h
/// @brief Estimates the Hurst exponent of the LLM bias signal stream.
///
/// ## Motivation
/// The Hurst exponent H classifies time-series persistence:
///   H > 0.5  — trending / persistent (momentum, follow the signal)
///   H = 0.5  — random walk (no memory)
///   H < 0.5  — mean-reverting / anti-persistent (fade the signal)
///
/// Applied to LLM bias streams: a high-H narrative is building momentum and
/// should be amplified; a low-H narrative is oscillating and signals should
/// be faded or position-sized down.
///
/// ## Algorithm
/// Uses the rescaled range (R/S) statistic over the rolling window:
///   1. Compute mean μ of the window.
///   2. Z_t = cumulative sum of (x_i − μ).
///   3. R = max(Z) − min(Z)  (range of cumulative deviations).
///   4. S = stddev of (x_i − μ).
///   5. H ≈ log(R/S) / log(n)   (single-scale estimate).
///
/// Degenerate case (S ≈ 0, constant series) → H = 0.5 (neutral).
///
/// ## Feature flag
/// `LLMQUANT_ENABLE_HURST_ESTIMATOR` (default ON).
///
/// Thread-safe.

#include <atomic>
#include <cstdint>
#include <functional>
#include <mutex>
#include <string>
#include <vector>

namespace llmquant {

class SignalHurstEstimator {
public:
    struct Config {
        /// Rolling window size. Must be >= 8. Default 64.
        int window{64};

        /// Minimum samples before emitting estimates. Default 16.
        int min_samples{16};

        /// H above this fires on_trending (persistent signal). Default 0.60.
        double trending_threshold{0.60};

        /// H below this fires on_mean_reverting (anti-persistent). Default 0.40.
        double reverting_threshold{0.40};

        /// Hysteresis multiplier for clearing states. Default 0.85.
        double clear_hysteresis{0.85};

        /// Fired on rising edge when H crosses above trending_threshold.
        std::function<void(double hurst)> on_trending;

        /// Fired on rising edge when H crosses below reverting_threshold.
        std::function<void(double hurst)> on_mean_reverting;
    };

    explicit SignalHurstEstimator(Config cfg = Config{});

    // -----------------------------------------------------------------------
    // Data intake
    // -----------------------------------------------------------------------

    /// Record a new bias observation. Recomputes Hurst estimate after each call.
    void record(double bias);

    // -----------------------------------------------------------------------
    // Accessors (lock-free)
    // -----------------------------------------------------------------------

    /// Current Hurst exponent estimate ∈ [0, 1].
    [[nodiscard]] double hurst() const noexcept {
        return hurst_.load(std::memory_order_relaxed);
    }

    /// True when H > trending_threshold (persistent / momentum regime).
    [[nodiscard]] bool is_trending() const noexcept {
        return trending_.load(std::memory_order_relaxed);
    }

    /// True when H < reverting_threshold (anti-persistent / mean-reversion regime).
    [[nodiscard]] bool is_mean_reverting() const noexcept {
        return reverting_.load(std::memory_order_relaxed);
    }

    /// Number of times trending state was entered (rising edge events).
    [[nodiscard]] uint64_t trend_events() const noexcept {
        return trend_events_.load(std::memory_order_relaxed);
    }

    /// Number of times mean-reverting state was entered (rising edge events).
    [[nodiscard]] uint64_t revert_events() const noexcept {
        return revert_events_.load(std::memory_order_relaxed);
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

    std::vector<double> buf_;
    int head_{0};
    int fill_{0};

    bool prev_trending_{false};
    bool prev_reverting_{false};

    std::atomic<double>   hurst_{0.5};
    std::atomic<bool>     trending_{false};
    std::atomic<bool>     reverting_{false};
    std::atomic<uint64_t> total_{0};
    std::atomic<uint64_t> trend_events_{0};
    std::atomic<uint64_t> revert_events_{0};

    void recompute_locked();
};

} // namespace llmquant
