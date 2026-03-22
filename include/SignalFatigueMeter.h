#pragma once

/// @file SignalFatigueMeter.h
/// @brief Consecutive same-direction signal streak detector.
///
/// ## Motivation
/// When the signal engine emits many consecutive bullish (or bearish) signals,
/// the underlying sentiment source may be exhausted or self-reinforcing.
/// Tracking the streak length quantifies "signal fatigue": the longer a one-
/// sided streak, the higher the probability of a near-term reversal.
///
/// ## Algorithm
/// Each call to `record(bias)` increments an internal streak counter when the
/// sign of `bias` matches the previous direction, or resets it to 1 on a flip.
///
/// Fatigue score ∈ [0, 1]:
///   fatigue = clamp(streak / saturation_streak, 0.0, 1.0)
///
/// where `saturation_streak` is the streak length at which fatigue saturates
/// to 1.0 (default 10).
///
/// Fires `on_fatigue(streak, score)` when the streak first reaches or exceeds
/// `fatigue_threshold` (default 5).
///
/// ## Feature flag
/// `LLMQUANT_ENABLE_SIGNAL_FATIGUE` (default ON).
///
/// Thread-safe.

#include <atomic>
#include <cstdint>
#include <functional>
#include <mutex>
#include <string>

namespace llmquant {

class SignalFatigueMeter {
public:
    struct Config {
        /// Streak length at which the on_fatigue callback fires. Default 5.
        int fatigue_threshold{5};

        /// Streak length at which fatigue score saturates to 1.0. Default 10.
        int saturation_streak{10};

        /// Fired when streak first reaches fatigue_threshold (outside lock).
        std::function<void(int streak, double score)> on_fatigue;
    };

    explicit SignalFatigueMeter(Config cfg = Config{});

    // -----------------------------------------------------------------------
    // Data intake
    // -----------------------------------------------------------------------

    /// Record a new bias value. Sign determines direction.
    /// Zero bias is treated as neutral and breaks any active streak.
    void record(double bias);

    // -----------------------------------------------------------------------
    // Accessors (lock-free)
    // -----------------------------------------------------------------------

    /// Fatigue score ∈ [0, 1].
    [[nodiscard]] double fatigue_score() const noexcept {
        return fatigue_score_.load(std::memory_order_relaxed);
    }

    /// Current streak length (same-direction consecutive records).
    [[nodiscard]] int streak() const noexcept {
        return streak_.load(std::memory_order_relaxed);
    }

    /// True when streak >= fatigue_threshold.
    [[nodiscard]] bool is_fatigued() const noexcept {
        return fatigued_.load(std::memory_order_relaxed);
    }

    /// Direction of current streak: +1 (bullish), -1 (bearish), 0 (neutral).
    [[nodiscard]] int direction() const noexcept {
        return direction_.load(std::memory_order_relaxed);
    }

    /// Total observations recorded.
    [[nodiscard]] uint64_t observation_count() const noexcept {
        return obs_count_.load(std::memory_order_relaxed);
    }

    [[nodiscard]] std::string to_stats_json() const;

    void reset();
    void update_config(const Config& cfg);

private:
    mutable std::mutex mutex_;
    Config cfg_;

    int  cur_streak_{0};
    int  cur_direction_{0};
    bool prev_fatigued_{false};

    std::atomic<double>   fatigue_score_{0.0};
    std::atomic<int>      streak_{0};
    std::atomic<bool>     fatigued_{false};
    std::atomic<int>      direction_{0};
    std::atomic<uint64_t> obs_count_{0};
};

} // namespace llmquant
