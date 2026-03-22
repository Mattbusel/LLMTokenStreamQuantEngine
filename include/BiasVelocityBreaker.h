#pragma once

/// @file BiasVelocityBreaker.h
/// @brief Rate-of-change circuit breaker for LLM bias signals.
///
/// ## Motivation
/// When LLM sentiment flips violently in consecutive tokens the bias
/// velocity (first-order discrete derivative) can spike.  Trading on such
/// whipsaw signals amplifies noise rather than capturing genuine narrative
/// momentum.  The velocity breaker trips when the smoothed rate-of-change
/// exceeds a configurable threshold and holds until the velocity subsides.
///
/// ## Algorithm
/// velocity_t = EMA(|bias_t − bias_{t-1}|, alpha)
/// Trip when velocity_t > max_velocity.  Reset when velocity_t drops
/// below max_velocity × clear_hysteresis.
///
/// ## Feature flag
/// `LLMQUANT_ENABLE_VELOCITY_BREAKER` (default ON).
///
/// Thread-safe.

#include <atomic>
#include <cstdint>
#include <functional>
#include <mutex>
#include <string>

namespace llmquant {

class BiasVelocityBreaker {
public:
    struct Config {
        /// EMA smoothing factor for velocity. Default 0.2.
        double ema_alpha{0.2};

        /// Velocity threshold above which the breaker trips. Default 0.15.
        double max_velocity{0.15};

        /// Hysteresis for clearing: clear when velocity < max_velocity × this. Default 0.75.
        double clear_hysteresis{0.75};

        /// Fired when breaker trips (outside lock). Receives smoothed velocity.
        std::function<void(double velocity)> on_trip;

        /// Fired when breaker clears. Receives smoothed velocity.
        std::function<void(double velocity)> on_clear;
    };

    explicit BiasVelocityBreaker(Config cfg = Config{});

    // -----------------------------------------------------------------------
    // Data intake
    // -----------------------------------------------------------------------

    /// Record a new bias value. Returns true if breaker is currently tripped.
    bool record(double bias);

    // -----------------------------------------------------------------------
    // Accessors (lock-free)
    // -----------------------------------------------------------------------

    /// Smoothed velocity (EMA of |Δbias|). Zero until second observation.
    [[nodiscard]] double velocity() const noexcept {
        return velocity_.load(std::memory_order_relaxed);
    }

    /// True when velocity > max_velocity.
    [[nodiscard]] bool is_tripped() const noexcept {
        return tripped_.load(std::memory_order_relaxed);
    }

    /// Number of times breaker has tripped (rising-edge count).
    [[nodiscard]] uint64_t trip_count() const noexcept {
        return trip_count_.load(std::memory_order_relaxed);
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

    double prev_bias_{0.0};
    double velocity_ema_{0.0};
    bool   seeded_{false};
    bool   prev_tripped_{false};

    std::atomic<double>   velocity_{0.0};
    std::atomic<bool>     tripped_{false};
    std::atomic<uint64_t> trip_count_{0};
    std::atomic<uint64_t> total_{0};
};

} // namespace llmquant
