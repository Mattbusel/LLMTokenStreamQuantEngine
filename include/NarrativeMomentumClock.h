#pragma once

/// @file NarrativeMomentumClock.h
/// @brief Four-quadrant momentum clock for narrative sentiment rotation.
///
/// Maps the smoothed (bias_ema, velocity_ema) pair to one of four clock
/// quadrants inspired by the macroeconomic Investment Clock:
///
///   Q1 Rising     — bias ≥ 0, velocity ≥ 0  (bullish trend accelerating)
///   Q2 Fading     — bias ≥ 0, velocity <  0  (bullish trend decelerating)
///   Q3 Falling    — bias <  0, velocity <  0  (bearish trend accelerating)
///   Q4 Recovering — bias <  0, velocity ≥ 0  (bearish trend losing steam)
///
/// Fires on_quadrant_change when rotation moves to a new quadrant.
/// Thread-safe: hot-path reads use atomics; EMA state uses a mutex.

#include <atomic>
#include <cstdint>
#include <functional>
#include <mutex>
#include <string>

namespace llmquant {

class NarrativeMomentumClock {
public:
    /// Clock quadrant labels.
    enum class Quadrant : int {
        Rising    = 0, ///< bias >= 0, velocity >= 0
        Fading    = 1, ///< bias >= 0, velocity <  0
        Falling   = 2, ///< bias <  0, velocity <  0
        Recovering = 3 ///< bias <  0, velocity >= 0
    };

    struct Config {
        double bias_alpha     = 0.10; ///< EMA smoothing factor for bias
        double velocity_alpha = 0.20; ///< EMA smoothing factor for velocity
        /// Fired when the clock rotates to a new quadrant.
        std::function<void(Quadrant from, Quadrant to,
                           double bias_ema, double velocity_ema)>
            on_quadrant_change;
    };

    NarrativeMomentumClock();
    explicit NarrativeMomentumClock(const Config& cfg);

    /// Ingest one raw bias sample and advance the clock.
    void record(double bias);

    /// Current clock quadrant (lock-free).
    Quadrant quadrant() const noexcept;

    /// Smoothed bias EMA (lock-free).
    double bias_ema() const noexcept;

    /// Smoothed velocity EMA — rate of change of bias_ema (lock-free).
    double velocity_ema() const noexcept;

    /// Total samples recorded.
    uint64_t total_records() const noexcept;

    /// Number of quadrant transitions observed.
    uint64_t quadrant_transitions() const noexcept;

    /// True when in Q1 (Rising) — bullish trend accelerating.
    bool is_rising() const noexcept;

    /// True when in Q3 (Falling) — bearish trend accelerating.
    bool is_falling() const noexcept;

    /// Reset all EMA state and counters.
    void reset();

    /// Replace config and reset state so new alphas take effect immediately.
    void update_config(const Config& cfg);

    /// JSON summary for diagnostics and session output.
    std::string to_stats_json() const;

private:
    static Quadrant compute_quadrant(double b, double v) noexcept;

    mutable std::mutex mutex_;
    Config   cfg_;
    double   bias_ema_val_     = 0.0;
    double   velocity_ema_val_ = 0.0;
    bool     seeded_           = false;

    std::atomic<int>      quadrant_{0};      ///< Quadrant as int for lock-free read
    std::atomic<double>   bias_out_{0.0};
    std::atomic<double>   velocity_out_{0.0};
    std::atomic<uint64_t> total_{0};
    std::atomic<uint64_t> transitions_{0};
};

} // namespace llmquant
