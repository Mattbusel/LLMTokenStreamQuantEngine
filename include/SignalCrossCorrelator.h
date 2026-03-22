#pragma once

/// @file SignalCrossCorrelator.h
/// @brief Computes cross-correlation between the raw bias stream and a smoothed
///        (EMA) version of itself to detect lead-lag relationships at lag τ.
///
/// ## Motivation
/// The cross-correlation between a raw signal and its own EMA at lag τ
/// measures how much of the current signal was "predictable" from the smoothed
/// past.  A high cross-correlation at short lags → EMA is a good predictor
/// (momentum regime).  A negative cross-correlation at lag 1 → signal reverts
/// faster than the EMA tracks (mean-reversion regime).
///
/// ## Algorithm
/// Maintains a rolling window and computes:
///   R(τ) = Cov(x_t, ema_{t-τ}) / (std_x * std_ema + ε)
/// for τ in [0, max_lag].  The lag with maximum |R(τ)| is the "dominant lag".
///
/// Fires `on_dominant_lag_change(old_lag, new_lag, r_val)` when dominant lag
/// changes, signalling a shift in lead-lag structure.
///
/// ## Feature flag
/// `LLMQUANT_ENABLE_CROSS_CORRELATOR` (default ON).
///
/// Thread-safe.

#include <atomic>
#include <cstdint>
#include <functional>
#include <mutex>
#include <string>
#include <vector>

namespace llmquant {

class SignalCrossCorrelator {
public:
    struct Config {
        /// EMA decay for the smoothed channel. Default 0.10.
        double ema_alpha{0.10};

        /// Rolling window for correlation computation. Default 60.
        int window{60};

        /// Maximum lag τ to compute R(τ). Default 5.
        int max_lag{5};

        /// Minimum fill before reporting. Default 20.
        int min_samples{20};

        /// Fired when dominant lag changes (outside lock).
        std::function<void(int old_lag, int new_lag, double r_val)> on_dominant_lag_change;
    };

    explicit SignalCrossCorrelator(Config cfg = Config{});

    // -----------------------------------------------------------------------
    // Data intake
    // -----------------------------------------------------------------------

    void record(double bias);

    // -----------------------------------------------------------------------
    // Accessors (lock-free)
    // -----------------------------------------------------------------------

    /// Cross-correlation at lag 0 (contemporaneous).
    [[nodiscard]] double r_lag0() const noexcept {
        return r0_.load(std::memory_order_relaxed);
    }

    /// Cross-correlation at lag 1.
    [[nodiscard]] double r_lag1() const noexcept {
        return r1_.load(std::memory_order_relaxed);
    }

    /// Dominant lag (τ with maximum |R(τ)|).
    [[nodiscard]] int dominant_lag() const noexcept {
        return dom_lag_.load(std::memory_order_relaxed);
    }

    /// Total dominant-lag change events.
    [[nodiscard]] uint64_t lag_change_events() const noexcept {
        return change_events_.load(std::memory_order_relaxed);
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

    std::vector<double> raw_buf_;   // rolling raw bias
    std::vector<double> ema_buf_;   // rolling EMA values
    int buf_head_{0};
    int buf_fill_{0};

    double ema_{0.0};
    int prev_dom_lag_{-1};

    std::atomic<double>   r0_{0.0};
    std::atomic<double>   r1_{0.0};
    std::atomic<int>      dom_lag_{0};
    std::atomic<uint64_t> change_events_{0};
    std::atomic<uint64_t> total_{0};
};

} // namespace llmquant
