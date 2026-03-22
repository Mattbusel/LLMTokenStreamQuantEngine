#pragma once

/// @file BiasHysteresisGate.h
/// @brief Debounce / time-persistence gate for directional bias signals.
///
/// ## Motivation
/// Raw LLM bias signals can spike briefly and retract — emitting a trade
/// signal on a single-tick exceedance produces whipsaws.  This gate requires
/// the bias magnitude to exceed `enter_threshold` for `required_ticks`
/// *consecutive* evaluations before the gate opens (allowing signal emission).
/// Once open, the gate stays open until bias falls below `exit_threshold`
/// (hysteresis prevents chattering around the boundary).
///
/// ## Algorithm
/// - `evaluate(bias)`: returns true when gate is open.
///   - If gate is **closed**: increment `consecutive_` when |bias| > enter_threshold;
///     reset to 0 otherwise.  Gate opens when consecutive count >= required_ticks.
///   - If gate is **open**: gate closes when |bias| < exit_threshold.
/// - On open: fires `on_gate_open(bias)` outside the lock (once per open event).
/// - On close: fires `on_gate_close(bias)` outside the lock (once per close event).
///
/// ## Feature flag
/// `LLMQUANT_ENABLE_BIAS_HYSTERESIS_GATE` (default ON).
///
/// Thread-safe: all public methods are thread-safe.

#include <atomic>
#include <cstdint>
#include <functional>
#include <mutex>
#include <string>

namespace llmquant {

class BiasHysteresisGate {
public:
    struct Config {
        /// Absolute bias magnitude that must be sustained to open the gate. Default 0.10.
        double enter_threshold{0.10};

        /// Absolute bias magnitude below which an open gate closes. Default 0.05.
        /// Must be ≤ enter_threshold for sensible hysteresis.
        double exit_threshold{0.05};

        /// Number of consecutive evaluations bias must exceed enter_threshold
        /// before the gate opens. Default 3.
        int required_ticks{3};

        /// Fired when the gate transitions from closed → open.
        /// Parameter: bias value at the opening tick.
        std::function<void(double)> on_gate_open;

        /// Fired when the gate transitions from open → closed.
        /// Parameter: bias value at the closing tick.
        std::function<void(double)> on_gate_close;
    };

    explicit BiasHysteresisGate(Config cfg = Config{});

    // -----------------------------------------------------------------------
    // Core
    // -----------------------------------------------------------------------

    /// Evaluate a new bias sample.
    /// @return true when the gate is currently open (signal should pass through).
    bool evaluate(double bias);

    // -----------------------------------------------------------------------
    // Accessors (lock-free)
    // -----------------------------------------------------------------------

    /// True when the gate is currently open.
    [[nodiscard]] bool is_open() const noexcept {
        return open_flag_.load(std::memory_order_relaxed);
    }

    /// Running count of consecutive ticks above enter_threshold (resets to 0 once gate opens).
    [[nodiscard]] int consecutive_ticks() const noexcept {
        return consec_flag_.load(std::memory_order_relaxed);
    }

    /// Total calls to evaluate().
    [[nodiscard]] uint64_t total_evaluated() const noexcept {
        return total_.load(std::memory_order_relaxed);
    }

    /// Number of gate-open events since construction or last reset().
    [[nodiscard]] uint64_t gate_open_events() const noexcept {
        return open_events_.load(std::memory_order_relaxed);
    }

    /// Number of gate-close events since construction or last reset().
    [[nodiscard]] uint64_t gate_close_events() const noexcept {
        return close_events_.load(std::memory_order_relaxed);
    }

    // -----------------------------------------------------------------------
    // Mutation
    // -----------------------------------------------------------------------

    void reset();
    void update_config(const Config& cfg);

    // -----------------------------------------------------------------------
    // Diagnostics
    // -----------------------------------------------------------------------

    [[nodiscard]] std::string to_stats_json() const;

private:
    mutable std::mutex mutex_;
    Config cfg_;

    // Internal state (modified under mutex_)
    bool gate_open_{false};
    int  consecutive_{0};

    // Atomic outputs for lock-free hot-path reads
    std::atomic<bool>     open_flag_{false};
    std::atomic<int>      consec_flag_{0};
    std::atomic<uint64_t> total_{0};
    std::atomic<uint64_t> open_events_{0};
    std::atomic<uint64_t> close_events_{0};
};

} // namespace llmquant
