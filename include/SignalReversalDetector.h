#pragma once

/// @file SignalReversalDetector.h
/// @brief Detects price-action-style reversal patterns in the LLM bias series.
///
/// ## Motivation
/// In price-action analysis, a reversal pattern is characterised by:
///   1. A directional move (the "thrust").
///   2. A counter-move that exceeds `reversal_fraction` of the thrust size.
///   3. The counter-move completing within `max_reversal_ticks` observations.
///
/// Applied to LLM bias: when a strong sentiment trend reverses decisively,
/// it signals a narrative pivot that warrants immediate position adjustment.
///
/// ## Algorithm
/// - Tracks the most recent local extremum (peak or trough) and the "thrust"
///   size from that extremum to the current value.
/// - When the counter-move from the most recent extremum exceeds
///   `reversal_fraction * |thrust|`, a reversal is declared.
/// - Fires `on_bullish_reversal(bias)` or `on_bearish_reversal(bias)`.
/// - After a reversal, resets and waits for a new thrust to form.
///
/// ## Feature flag
/// `LLMQUANT_ENABLE_REVERSAL_DETECTOR` (default ON).
///
/// Thread-safe.

#include <atomic>
#include <cstdint>
#include <functional>
#include <mutex>
#include <string>

namespace llmquant {

class SignalReversalDetector {
public:
    struct Config {
        /// Counter-move fraction of thrust required to declare a reversal. Default 0.5.
        double reversal_fraction{0.5};

        /// Minimum thrust size to track. Smaller moves are noise. Default 0.05.
        double min_thrust{0.05};

        /// Fired when a bearish→bullish reversal occurs (outside lock).
        std::function<void(double bias)> on_bullish_reversal;

        /// Fired when a bullish→bearish reversal occurs (outside lock).
        std::function<void(double bias)> on_bearish_reversal;
    };

    explicit SignalReversalDetector(Config cfg = Config{});

    // -----------------------------------------------------------------------
    // Data intake
    // -----------------------------------------------------------------------

    void record(double bias);

    // -----------------------------------------------------------------------
    // Accessors (lock-free)
    // -----------------------------------------------------------------------

    /// Direction of the most recently declared reversal: +1=bullish, -1=bearish, 0=none.
    [[nodiscard]] int last_reversal_direction() const noexcept {
        return last_dir_.load(std::memory_order_relaxed);
    }

    /// Number of bullish reversal events fired.
    [[nodiscard]] uint64_t bullish_reversals() const noexcept {
        return bull_events_.load(std::memory_order_relaxed);
    }

    /// Number of bearish reversal events fired.
    [[nodiscard]] uint64_t bearish_reversals() const noexcept {
        return bear_events_.load(std::memory_order_relaxed);
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

    // State
    bool   seeded_{false};
    double extremum_{0.0};      // current local high/low being tracked
    int    thrust_dir_{0};      // +1 if we're above extremum, -1 if below
    double thrust_size_{0.0};

    std::atomic<int>      last_dir_{0};
    std::atomic<uint64_t> bull_events_{0};
    std::atomic<uint64_t> bear_events_{0};
    std::atomic<uint64_t> total_{0};
};

} // namespace llmquant
