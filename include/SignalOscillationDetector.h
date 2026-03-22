#pragma once

/// @file SignalOscillationDetector.h
/// @brief Detects oscillating (alternating direction) LLM bias patterns.
///
/// ## Motivation
/// When an LLM flip-flops between bullish and bearish tokens, the resulting
/// signals alternate in sign — pure noise with no exploitable momentum.
/// The zero-crossing rate (ZCR) of the bias change series measures this:
/// ZCR = fraction of consecutive pairs where sign(Δ_t) ≠ sign(Δ_{t-1}).
///
/// High ZCR (> oscillation_threshold) → model is oscillating → suppress or
/// fade signals.  Fires on_oscillating and on_stabilized callbacks.
///
/// ## Algorithm
/// Maintain a rolling ring of sign(bias_t − bias_{t-1}).
/// ZCR = (count of sign changes in window) / (window − 1).
///
/// ## Feature flag
/// `LLMQUANT_ENABLE_OSCILLATION_DETECTOR` (default ON).
///
/// Thread-safe.

#include <atomic>
#include <cstdint>
#include <functional>
#include <mutex>
#include <string>
#include <vector>

namespace llmquant {

class SignalOscillationDetector {
public:
    struct Config {
        /// Rolling window for ZCR computation. Default 16.
        int window{16};

        /// Minimum samples before reporting. Default 4.
        int min_samples{4};

        /// ZCR above this fires on_oscillating. Default 0.65.
        double oscillation_threshold{0.65};

        /// Hysteresis for clearing. Default 0.75.
        double clear_hysteresis{0.75};

        /// Fired when ZCR rises above threshold (rising edge).
        std::function<void(double zcr)> on_oscillating;

        /// Fired when ZCR drops back below threshold.
        std::function<void(double zcr)> on_stabilized;
    };

    explicit SignalOscillationDetector(Config cfg = Config{});

    // -----------------------------------------------------------------------
    // Data intake
    // -----------------------------------------------------------------------

    void record(double bias);

    // -----------------------------------------------------------------------
    // Accessors (lock-free)
    // -----------------------------------------------------------------------

    /// Zero-crossing rate ∈ [0, 1].
    [[nodiscard]] double zcr() const noexcept {
        return zcr_.load(std::memory_order_relaxed);
    }

    /// True when ZCR > oscillation_threshold.
    [[nodiscard]] bool is_oscillating() const noexcept {
        return oscillating_.load(std::memory_order_relaxed);
    }

    /// Rising-edge oscillation events.
    [[nodiscard]] uint64_t oscillation_events() const noexcept {
        return events_.load(std::memory_order_relaxed);
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

    // Ring of +1/-1 bias direction signs.
    std::vector<int8_t> sign_ring_;
    int head_{0};
    int fill_{0};

    double prev_bias_{0.0};
    bool   seeded_{false};
    bool   prev_oscillating_{false};

    std::atomic<double>   zcr_{0.0};
    std::atomic<bool>     oscillating_{false};
    std::atomic<uint64_t> events_{0};
    std::atomic<uint64_t> total_{0};
};

} // namespace llmquant
