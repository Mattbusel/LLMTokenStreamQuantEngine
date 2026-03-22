#pragma once

/// @file NarrativeMassIndex.h
/// @brief Computes the Mass Index of the bias stream — a volatility oscillator
///        that detects range expansions signalling potential trend reversals.
///
/// ## Motivation
/// The Mass Index (Dorsey, 1992) originally uses high-low price ranges.
/// Applied to a bias stream we use the rolling absolute deviation instead:
///
///   single_EMA_t = EMA(|x_t|, α_fast)
///   double_EMA_t = EMA(single_EMA_t, α_slow)
///   ratio_t      = single_EMA_t / (double_EMA_t + ε)
///   MI_t         = rolling sum of ratio over `period` samples
///
/// When MI_t rises above `reversal_bulge` (typically 27) and then falls below
/// `reversal_trigger` (typically 26.5), a "reversal bulge" has formed —
/// a high-probability trend-reversal signal regardless of direction.
///
/// Fires `on_bulge_start(mi)` when MI > reversal_bulge.
/// Fires `on_reversal_signal(mi)` when MI drops back below reversal_trigger
/// (after a bulge).
///
/// ## Feature flag
/// `LLMQUANT_ENABLE_MASS_INDEX` (default ON).
///
/// Thread-safe.

#include <atomic>
#include <cstdint>
#include <functional>
#include <mutex>
#include <string>
#include <vector>

namespace llmquant {

class NarrativeMassIndex {
public:
    struct Config {
        /// Fast EMA period for single EMA. Default 9 (α = 2/(9+1)).
        int fast_period{9};

        /// Slow EMA period for double EMA. Default 9 (same period).
        int slow_period{9};

        /// Rolling sum window (usually 25). Default 25.
        int period{25};

        /// MI above this → bulge forming. Default 27.0.
        double reversal_bulge{27.0};

        /// MI below this (after bulge) → reversal signal. Default 26.5.
        double reversal_trigger{26.5};

        /// Minimum fill before signalling. Default period.
        int min_samples{-1}; // <0 → use period

        /// Fired when MI > reversal_bulge (outside lock).
        std::function<void(double mi)> on_bulge_start;

        /// Fired when MI drops below reversal_trigger after a bulge (outside lock).
        std::function<void(double mi)> on_reversal_signal;
    };

    explicit NarrativeMassIndex(Config cfg = Config{});

    // -----------------------------------------------------------------------
    // Data intake
    // -----------------------------------------------------------------------

    void record(double bias);

    // -----------------------------------------------------------------------
    // Accessors (lock-free)
    // -----------------------------------------------------------------------

    /// Current Mass Index value (rolling sum of EMA ratio).
    [[nodiscard]] double mass_index() const noexcept {
        return mi_.load(std::memory_order_relaxed);
    }

    /// True when MI is currently above reversal_bulge.
    [[nodiscard]] bool in_bulge() const noexcept {
        return in_bulge_.load(std::memory_order_relaxed);
    }

    /// Total bulge events fired.
    [[nodiscard]] uint64_t bulge_events() const noexcept {
        return bulge_events_.load(std::memory_order_relaxed);
    }

    /// Total reversal signals fired.
    [[nodiscard]] uint64_t reversal_events() const noexcept {
        return reversal_events_.load(std::memory_order_relaxed);
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

    double ema1_{0.0};     // fast EMA of |bias|
    double ema2_{0.0};     // slow EMA of ema1
    double alpha1_{0.0};
    double alpha2_{0.0};

    std::vector<double> ratio_buf_; // ring buffer of EMA ratios
    int ratio_head_{0};
    int ratio_fill_{0};

    bool was_in_bulge_{false};

    std::atomic<double>   mi_{0.0};
    std::atomic<bool>     in_bulge_{false};
    std::atomic<uint64_t> bulge_events_{0};
    std::atomic<uint64_t> reversal_events_{0};
    std::atomic<uint64_t> total_{0};
};

} // namespace llmquant
