#pragma once

/// @file TokenWeightQuantiser.h
/// @brief Maps continuous token weights to a fixed quantization grid using
///        stochastic rounding, maintaining exact expected value.
///
/// ## Motivation
/// Downstream signal processing at integer precision (e.g., SIMD integer
/// accumulation, fixed-point arithmetic) is 2-8× faster than floating-point.
/// Naive truncation or round-to-nearest introduces systematic bias.  Stochastic
/// rounding eliminates bias: the probability of rounding up equals the fractional
/// part, so E[quantised(w)] = w exactly.  This enables exact expected-value
/// preservation across many tokens while permitting integer-precision math.
///
/// ## Algorithm
/// Given N quantization levels spanning [range_min, range_max]:
/// - `step = (range_max - range_min) / (N - 1)`
/// - `frac = (w - range_min) / step`
/// - `level = floor(frac)`, rounded up with probability `frac - floor(frac)`.
/// - Quantised float = `range_min + level * step`.
///
/// Also tracks:
/// - `quantisation_error` = rolling mean of |w - quantised(w)|
/// - `clamp_count` : times a weight was outside [range_min, range_max]
///
/// ## Feature flag
/// `LLMQUANT_ENABLE_TOKEN_QUANTISER` (default ON).
///
/// Thread-safe.

#include <atomic>
#include <cstdint>
#include <functional>
#include <mutex>
#include <random>
#include <string>

namespace llmquant {

class TokenWeightQuantiser {
public:
    struct Config {
        /// Number of quantization levels. Default 256.
        int    levels{256};

        /// Range minimum (weights below this are clamped). Default -1.0.
        double range_min{-1.0};

        /// Range maximum (weights above this are clamped). Default 1.0.
        double range_max{1.0};

        /// EMA alpha for quantisation error tracking. Default 0.05.
        double error_alpha{0.05};

        /// Fired when clamp_rate (clamped / total) exceeds this fraction. Default 0.05.
        double clamp_alert_fraction{0.05};

        /// Fired when clamp rate is too high.
        /// Parameters: (clamp_rate, last_weight).
        std::function<void(double clamp_rate, double weight)> on_high_clamp_rate;
    };

    explicit TokenWeightQuantiser(Config cfg = Config{});

    // -----------------------------------------------------------------------
    // Quantization
    // -----------------------------------------------------------------------

    /// Quantise a weight using stochastic rounding.
    /// @return Quantised float value on the grid.
    [[nodiscard]] double quantise(double weight);

    /// Quantise and return the integer level index ∈ [0, levels-1].
    [[nodiscard]] int quantise_level(double weight);

    // -----------------------------------------------------------------------
    // Accessors (lock-free)
    // -----------------------------------------------------------------------

    /// EMA of absolute quantisation error.
    [[nodiscard]] double quantisation_error() const noexcept {
        return quant_error_.load(std::memory_order_relaxed);
    }

    /// Total values clamped to [range_min, range_max].
    [[nodiscard]] uint64_t clamp_count() const noexcept {
        return clamp_count_.load(std::memory_order_relaxed);
    }

    /// Total values quantised.
    [[nodiscard]] uint64_t total_count() const noexcept {
        return total_count_.load(std::memory_order_relaxed);
    }

    /// Clamp rate = clamp_count / total_count.
    [[nodiscard]] double clamp_rate() const noexcept;

    [[nodiscard]] std::string to_stats_json() const;

    void reset();
    void update_config(const Config& cfg);

private:
    mutable std::mutex mutex_;
    Config cfg_;
    std::mt19937_64 rng_;
    std::uniform_real_distribution<double> uniform_{0.0, 1.0};

    double quant_error_val_{0.0};

    std::atomic<double>   quant_error_{0.0};
    std::atomic<uint64_t> clamp_count_{0};
    std::atomic<uint64_t> total_count_{0};
};

} // namespace llmquant
