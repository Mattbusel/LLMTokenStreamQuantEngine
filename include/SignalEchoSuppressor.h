#pragma once

/// @file SignalEchoSuppressor.h
/// @brief Detects and suppresses near-duplicate consecutive signals.
///
/// ## Motivation
/// LLM token streams can enter "echo" states where the model repeatedly
/// outputs nearly identical sentiment tokens (narrative stutter).  Each
/// such token triggers a signal that is informationally redundant but
/// still consumes OMS bandwidth and risk capacity.
///
/// The SignalEchoSuppressor measures the fraction of recent signals that
/// are "echoes" (|bias - prev_bias| < echo_threshold) and fires a
/// callback when the echo rate exceeds a configurable level.
///
/// ## Algorithm
/// Maintains a rolling boolean ring: entry = 1 if echo, 0 if novel.
/// echo_rate = (sum of last `window` echo flags) / window.
///
/// Fires `on_echo_detected(rate)` on rising edge above `echo_rate_threshold`.
/// Fires `on_echo_cleared(rate)` on falling edge below threshold × hysteresis.
///
/// ## Feature flag
/// `LLMQUANT_ENABLE_ECHO_SUPPRESSOR` (default ON).
///
/// Thread-safe.

#include <atomic>
#include <cstdint>
#include <functional>
#include <mutex>
#include <string>
#include <vector>

namespace llmquant {

class SignalEchoSuppressor {
public:
    struct Config {
        /// Bias difference below which a signal is classified as an echo. Default 1e-4.
        double echo_threshold{1e-4};

        /// Rolling window size for echo rate computation. Default 32.
        int window{32};

        /// Echo rate above which on_echo_detected fires. Default 0.6.
        double echo_rate_threshold{0.6};

        /// Hysteresis for clearing: fires when rate < threshold × this. Default 0.75.
        double clear_hysteresis{0.75};

        /// Fired when echo rate crosses above echo_rate_threshold (outside lock).
        std::function<void(double rate)> on_echo_detected;

        /// Fired when echo rate drops back below threshold after being in echo state.
        std::function<void(double rate)> on_echo_cleared;
    };

    explicit SignalEchoSuppressor(Config cfg = Config{});

    // -----------------------------------------------------------------------
    // Data intake
    // -----------------------------------------------------------------------

    /// Record a new signal bias value. Returns true if this signal is an echo.
    bool record(double bias);

    // -----------------------------------------------------------------------
    // Accessors (lock-free)
    // -----------------------------------------------------------------------

    /// Current rolling echo rate ∈ [0, 1].
    [[nodiscard]] double echo_rate() const noexcept {
        return echo_rate_.load(std::memory_order_relaxed);
    }

    /// True when echo rate > echo_rate_threshold.
    [[nodiscard]] bool is_echoing() const noexcept {
        return echoing_.load(std::memory_order_relaxed);
    }

    /// Total signals recorded (echoes + novel).
    [[nodiscard]] uint64_t total_signals() const noexcept {
        return total_.load(std::memory_order_relaxed);
    }

    /// Total echo signals counted.
    [[nodiscard]] uint64_t echo_count() const noexcept {
        return echo_count_.load(std::memory_order_relaxed);
    }

    /// Number of times echo state was entered (rising edge events).
    [[nodiscard]] uint64_t echo_events() const noexcept {
        return echo_events_.load(std::memory_order_relaxed);
    }

    [[nodiscard]] std::string to_stats_json() const;

    void reset();
    void update_config(const Config& cfg);

private:
    mutable std::mutex mutex_;
    Config cfg_;

    std::vector<int8_t> ring_;  // 1 = echo, 0 = novel
    int     head_{0};
    int     fill_{0};
    int     echo_sum_{0};

    double  prev_bias_{0.0};
    bool    seeded_{false};
    bool    prev_echoing_{false};

    std::atomic<double>   echo_rate_{0.0};
    std::atomic<bool>     echoing_{false};
    std::atomic<uint64_t> total_{0};
    std::atomic<uint64_t> echo_count_{0};
    std::atomic<uint64_t> echo_events_{0};
};

} // namespace llmquant
