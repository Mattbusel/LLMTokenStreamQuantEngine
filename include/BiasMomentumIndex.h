#pragma once

/// @file BiasMomentumIndex.h
/// @brief MACD-style dual-EMA momentum index for LLM bias streams.
///
/// ## Motivation
/// The Moving Average Convergence Divergence (MACD) is the canonical momentum
/// oscillator in quantitative trading.  Applying it to the LLM bias stream
/// gives a fast, interpretable momentum signal:
///
///   fast_ema  = EMA(bias, fast_alpha)
///   slow_ema  = EMA(bias, slow_alpha)
///   macd_line = fast_ema - slow_ema       (signed momentum)
///   signal_line = EMA(macd_line, signal_alpha)  (smoothed MACD)
///   histogram = macd_line - signal_line    (acceleration)
///
/// Bullish crossover: macd_line crosses above signal_line → momentum turning up.
/// Bearish crossover: macd_line crosses below signal_line → momentum turning down.
///
/// ## Feature flag
/// `LLMQUANT_ENABLE_MOMENTUM_INDEX` (default ON).
///
/// Thread-safe.

#include <atomic>
#include <cstdint>
#include <functional>
#include <mutex>
#include <string>

namespace llmquant {

class BiasMomentumIndex {
public:
    struct Config {
        /// Fast EMA smoothing factor. Default 2/(8+1) ≈ 0.222.
        double fast_alpha{0.222};

        /// Slow EMA smoothing factor. Default 2/(21+1) ≈ 0.0909.
        double slow_alpha{0.0909};

        /// Signal line EMA smoothing factor. Default 2/(9+1) = 0.2.
        double signal_alpha{0.2};

        /// Fired on bullish crossover (macd crosses above signal_line). Receives macd value.
        std::function<void(double macd)> on_bullish_crossover;

        /// Fired on bearish crossover (macd crosses below signal_line). Receives macd value.
        std::function<void(double macd)> on_bearish_crossover;
    };

    explicit BiasMomentumIndex(Config cfg = Config{});

    // -----------------------------------------------------------------------
    // Data intake
    // -----------------------------------------------------------------------

    /// Record a new bias value and update all three EMAs. Returns current histogram.
    double record(double bias);

    // -----------------------------------------------------------------------
    // Accessors (lock-free)
    // -----------------------------------------------------------------------

    /// Fast EMA of bias.
    [[nodiscard]] double fast_ema() const noexcept {
        return fast_ema_.load(std::memory_order_relaxed);
    }

    /// Slow EMA of bias.
    [[nodiscard]] double slow_ema() const noexcept {
        return slow_ema_.load(std::memory_order_relaxed);
    }

    /// MACD line = fast_ema − slow_ema.
    [[nodiscard]] double macd_line() const noexcept {
        return macd_line_.load(std::memory_order_relaxed);
    }

    /// Signal line = EMA(macd_line).
    [[nodiscard]] double signal_line() const noexcept {
        return signal_line_.load(std::memory_order_relaxed);
    }

    /// Histogram = macd_line − signal_line (momentum acceleration).
    [[nodiscard]] double histogram() const noexcept {
        return histogram_.load(std::memory_order_relaxed);
    }

    /// True when MACD is above signal line (bullish momentum).
    [[nodiscard]] bool is_bullish() const noexcept {
        return bullish_.load(std::memory_order_relaxed);
    }

    /// Cumulative bullish crossover events.
    [[nodiscard]] uint64_t bullish_crossovers() const noexcept {
        return bullish_cross_.load(std::memory_order_relaxed);
    }

    /// Cumulative bearish crossover events.
    [[nodiscard]] uint64_t bearish_crossovers() const noexcept {
        return bearish_cross_.load(std::memory_order_relaxed);
    }

    [[nodiscard]] uint64_t total_records() const noexcept {
        return total_.load(std::memory_order_relaxed);
    }

    [[nodiscard]] std::string to_stats_json() const;

    void reset();
    void update_config(const Config& cfg);

private:
    mutable std::mutex mutex_;
    Config cfg_;

    double fast_ema_v_{0.0};
    double slow_ema_v_{0.0};
    double signal_v_{0.0};
    bool   seeded_{false};
    bool   prev_bullish_{false};

    std::atomic<double>   fast_ema_{0.0};
    std::atomic<double>   slow_ema_{0.0};
    std::atomic<double>   macd_line_{0.0};
    std::atomic<double>   signal_line_{0.0};
    std::atomic<double>   histogram_{0.0};
    std::atomic<bool>     bullish_{false};
    std::atomic<uint64_t> bullish_cross_{0};
    std::atomic<uint64_t> bearish_cross_{0};
    std::atomic<uint64_t> total_{0};
};

} // namespace llmquant
