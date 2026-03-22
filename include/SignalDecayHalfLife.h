#pragma once

/// @file SignalDecayHalfLife.h
/// @brief Exponential half-life estimator for signal momentum decay.
///
/// ## Motivation
/// After a sentiment event (earnings, news), signal magnitude decays roughly
/// exponentially.  Knowing the half-life tells the trading system HOW LONG the
/// signal will remain tradeable — and whether momentum is fading faster than usual.
///
/// ## Algorithm
/// Maintains a rolling window of |bias| values.  Fits a least-squares line to
/// the log-transformed magnitudes (log|bias_t|) over time t:
///   log|bias_t| = a - λt + ε
///
/// The decay rate λ gives the half-life:
///   t½ = ln(2) / λ   (in samples)
///
/// OLS on log-space: using the rolling sums Σt, Σlog|x|, Σt², Σt·log|x| over
/// the window. O(1) per update using running Welford-style accumulation.
///
/// Fires `on_fast_decay` when t½ < fast_threshold (signal fading quickly).
/// Fires `on_slow_decay` when t½ > slow_threshold (signal persisting unusually long).
///
/// Returns NaN when insufficient data or |bias| = 0 entries dominate the window.
///
/// ## Feature flag
/// `LLMQUANT_ENABLE_SIGNAL_DECAY_HALFLIFE` (default ON).
///
/// Thread-safe.

#include <atomic>
#include <cstdint>
#include <functional>
#include <mutex>
#include <string>
#include <vector>

namespace llmquant {

class SignalDecayHalfLife {
public:
    struct Config {
        /// Sliding window size for OLS regression. Default 32.
        int window{32};

        /// Minimum valid samples before half-life is computable. Default 8.
        int min_samples{8};

        /// Minimum |bias| to include in log regression. Default 1e-6.
        double min_abs{1e-6};

        /// Half-life below this threshold fires on_fast_decay (samples). Default 5.0.
        double fast_threshold{5.0};

        /// Half-life above this threshold fires on_slow_decay (samples). Default 50.0.
        double slow_threshold{50.0};

        /// Fired (outside lock) when estimated half-life < fast_threshold.
        std::function<void(double half_life)> on_fast_decay;

        /// Fired (outside lock) when estimated half-life > slow_threshold.
        std::function<void(double half_life)> on_slow_decay;
    };

    explicit SignalDecayHalfLife(Config cfg = Config{});

    // -----------------------------------------------------------------------
    // Data intake
    // -----------------------------------------------------------------------

    void record(double bias);

    // -----------------------------------------------------------------------
    // Accessors (lock-free)
    // -----------------------------------------------------------------------

    /// Estimated half-life in samples. Returns NaN if not enough valid data.
    [[nodiscard]] double half_life() const noexcept {
        return half_life_.load(std::memory_order_relaxed);
    }

    /// Estimated decay rate λ (nats per sample). NaN if not computable.
    [[nodiscard]] double decay_rate() const noexcept {
        return decay_rate_.load(std::memory_order_relaxed);
    }

    /// R² of the log-linear fit (0-1). NaN if not computable.
    [[nodiscard]] double r_squared() const noexcept {
        return r_squared_.load(std::memory_order_relaxed);
    }

    [[nodiscard]] uint64_t total_records() const noexcept {
        return total_.load(std::memory_order_relaxed);
    }

    [[nodiscard]] uint64_t fast_decay_events() const noexcept {
        return fast_events_.load(std::memory_order_relaxed);
    }

    [[nodiscard]] uint64_t slow_decay_events() const noexcept {
        return slow_events_.load(std::memory_order_relaxed);
    }

    [[nodiscard]] std::string to_stats_json() const;

    void reset();
    void update_config(const Config& cfg);

private:
    mutable std::mutex mutex_;
    Config             cfg_;

    // Circular buffer of (time_index, log|bias|) pairs
    struct Sample { double t; double log_abs; };
    std::vector<Sample> buf_;
    int head_{0};
    int fill_{0};
    uint64_t tick_{0};  ///< monotonic time index

    bool prev_fast_{false};
    bool prev_slow_{false};

    std::atomic<double>   half_life_{std::numeric_limits<double>::quiet_NaN()};
    std::atomic<double>   decay_rate_{std::numeric_limits<double>::quiet_NaN()};
    std::atomic<double>   r_squared_{std::numeric_limits<double>::quiet_NaN()};
    std::atomic<uint64_t> total_{0};
    std::atomic<uint64_t> fast_events_{0};
    std::atomic<uint64_t> slow_events_{0};

    void recompute_locked();
};

} // namespace llmquant
