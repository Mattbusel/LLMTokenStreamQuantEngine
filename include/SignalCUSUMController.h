#pragma once

/// @file SignalCUSUMController.h
/// @brief CUSUM control chart for online step-change detection in signal streams.
///
/// ## Motivation
/// Z-score detectors (AnomalyDetector) are tuned for point spikes.  CUSUM
/// (Cumulative Sum Control Chart) is fundamentally different: it accumulates
/// small deviations over time and fires when the cumulative sum exceeds a
/// threshold.  This makes it ideal for detecting STEP CHANGES — a regime
/// shift where the signal mean has permanently moved, even if no single point
/// is individually extreme.
///
/// Example: sentiment gradually drifting from +0.05 to +0.40 over 20 tokens
/// looks completely normal sample-by-sample but CUSUM catches it because it
/// accumulates the +0.01 excess at every step.
///
/// ## Algorithm (Two-sided CUSUM)
///   S+_n = max(0, S+_{n-1} + (x_n - μ_0 - k))   positive accumulator
///   S-_n = max(0, S-_{n-1} + (μ_0 - x_n - k))   negative accumulator
///
/// where:
///   μ_0 = target mean (default 0.0)
///   k   = allowance / slack; tune to ½ of the shift magnitude you want to
///         detect.  Default 0.10.
///   h   = decision threshold; fire when S+ > h or S- > h.  Default 5.0.
///
/// Fires `on_upward_shift(S+)` when S+ first crosses h.
/// Fires `on_downward_shift(S-)` when S- first crosses h.
/// CUSUM accumulators are reset after each alarm.
///
/// ## Feature flag
/// `LLMQUANT_ENABLE_SIGNAL_CUSUM` (default ON).
///
/// Thread-safe.

#include <atomic>
#include <cstdint>
#include <functional>
#include <mutex>
#include <string>

namespace llmquant {

class SignalCUSUMController {
public:
    struct Config {
        /// Target mean (null-hypothesis mean). Default 0.0.
        double target_mean{0.0};

        /// Allowance: half the detectable shift.  Default 0.10.
        double allowance{0.10};

        /// Decision threshold: fire when accumulator exceeds this. Default 5.0.
        double threshold{5.0};

        /// If true, reset accumulators to 0 after an alarm. Default true.
        bool reset_on_alarm{true};

        /// Fired (outside lock) when S+ exceeds threshold (upward mean shift).
        /// Parameter: current S+ value.
        std::function<void(double cusum_pos)> on_upward_shift;

        /// Fired (outside lock) when S- exceeds threshold (downward mean shift).
        /// Parameter: current S- value.
        std::function<void(double cusum_neg)> on_downward_shift;
    };

    explicit SignalCUSUMController(Config cfg = Config{});

    // -----------------------------------------------------------------------
    // Data intake
    // -----------------------------------------------------------------------

    /// Update CUSUM accumulators with a new signal value.
    void update(double x);

    // -----------------------------------------------------------------------
    // Accessors (lock-free)
    // -----------------------------------------------------------------------

    /// Current positive accumulator S+.
    [[nodiscard]] double cusum_pos() const noexcept {
        return sp_.load(std::memory_order_relaxed);
    }

    /// Current negative accumulator S-.
    [[nodiscard]] double cusum_neg() const noexcept {
        return sn_.load(std::memory_order_relaxed);
    }

    /// True when either accumulator currently exceeds threshold (alarm active).
    [[nodiscard]] bool is_alarmed() const noexcept {
        return alarmed_.load(std::memory_order_relaxed);
    }

    /// Total number of upward alarms fired.
    [[nodiscard]] uint64_t upward_alarms() const noexcept {
        return up_alarms_.load(std::memory_order_relaxed);
    }

    /// Total number of downward alarms fired.
    [[nodiscard]] uint64_t downward_alarms() const noexcept {
        return down_alarms_.load(std::memory_order_relaxed);
    }

    /// Total observations processed.
    [[nodiscard]] uint64_t total_updates() const noexcept {
        return total_.load(std::memory_order_relaxed);
    }

    [[nodiscard]] std::string to_stats_json() const;

    void reset();
    void update_config(const Config& cfg);

private:
    mutable std::mutex mutex_;
    Config cfg_;

    double cur_sp_{0.0};
    double cur_sn_{0.0};

    std::atomic<double>   sp_{0.0};
    std::atomic<double>   sn_{0.0};
    std::atomic<bool>     alarmed_{false};
    std::atomic<uint64_t> up_alarms_{0};
    std::atomic<uint64_t> down_alarms_{0};
    std::atomic<uint64_t> total_{0};
};

} // namespace llmquant
