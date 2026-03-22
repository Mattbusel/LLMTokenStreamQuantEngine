#pragma once

#include <atomic>
#include <cstdint>
#include <functional>
#include <mutex>
#include <string>
#include <vector>

namespace llmquant {

/**
 * @brief Computes velocity and acceleration of the LLM sentiment bias stream.
 *
 * Sentiment bias is a continuous signal.  Its first and second time
 * derivatives carry distinct trading information:
 *
 *  - **Velocity (dB/dt)**: rate of change of bias.  High velocity signals a
 *    rapid sentiment shift — typically driven by breaking news entering the
 *    token stream.  Large positive velocity = bullish momentum building.
 *
 *  - **Acceleration (d²B/dt²)**: rate of change of velocity.  High positive
 *    acceleration means velocity itself is increasing — a regime change that
 *    may justify larger position entry.  Sudden negative acceleration after
 *    a spike may indicate the narrative is reversing.
 *
 * ## Method
 * Maintains a rolling window of (timestamp_ms, bias) observations.  Velocity
 * is estimated via a centered finite difference on the window endpoints:
 *   v ≈ (B_last - B_first) / Δt_ms
 * Acceleration is the delta between the most-recent and prior velocity
 * estimates, also time-normalized.
 *
 * ## Feature flag
 * Controlled by `LLMQUANT_ENABLE_VELOCITY_TRACKER` (default ON).
 */
class TokenVelocityTracker {
public:
    struct Config {
        /**
         * @brief Rolling window of observations used for derivative estimation.
         * Default 16.
         */
        int window_size{16};

        /**
         * @brief Minimum samples before velocity/acceleration are reported.
         * Default 4.
         */
        int min_samples{4};

        /**
         * @brief Velocity threshold above which is_fast_move() returns true.
         * Units: bias-units per second.  Default 0.5.
         */
        double fast_move_threshold{0.5};

        /**
         * @brief Callback fired when |velocity| > fast_move_threshold.
         * Parameters: (velocity, acceleration).
         */
        std::function<void(double vel, double accel)> on_fast_move;
    };

    // -----------------------------------------------------------------------
    // Construction
    // -----------------------------------------------------------------------

    explicit TokenVelocityTracker(Config config = Config{});

    // -----------------------------------------------------------------------
    // Update
    // -----------------------------------------------------------------------

    /**
     * @brief Record a new bias observation.
     *
     * Uses steady_clock internally for timestamps.
     * Thread-safe.
     */
    void record(double bias) noexcept;

    // -----------------------------------------------------------------------
    // Query
    // -----------------------------------------------------------------------

    /**
     * @brief Current bias velocity in bias-units per second.  Thread-safe. */
    [[nodiscard]] double velocity() const noexcept {
        return velocity_.load(std::memory_order_relaxed);
    }

    /**
     * @brief Current bias acceleration in bias-units per second².  Thread-safe. */
    [[nodiscard]] double acceleration() const noexcept {
        return acceleration_.load(std::memory_order_relaxed);
    }

    /**
     * @brief True when |velocity()| >= fast_move_threshold.  Thread-safe. */
    [[nodiscard]] bool is_fast_move() const noexcept;

    /** @brief Total observations recorded.  Thread-safe. */
    [[nodiscard]] uint64_t total_records() const noexcept {
        return total_.load(std::memory_order_relaxed);
    }

    // -----------------------------------------------------------------------
    // Config / reset
    // -----------------------------------------------------------------------

    /** @brief Replace configuration; clears window. */
    void update_config(Config config);

    /** @brief Reset window and derivative estimates. */
    void reset();

    /** @brief JSON snapshot of current estimates. */
    [[nodiscard]] std::string to_stats_json() const;

private:
    using Clock     = std::chrono::steady_clock;
    using TimePoint = Clock::time_point;

    /// Recompute velocity and acceleration from the rolling window.
    /// Caller must hold mutex_.
    void recompute_locked() noexcept;

    mutable std::mutex mutex_;
    Config config_;

    struct Sample {
        double time_s{0.0};
        double bias{0.0};
    };
    std::vector<Sample> ring_;
    int    ring_head_{0};
    int    ring_fill_{0};
    TimePoint epoch_;

    double prev_velocity_{0.0};

    std::atomic<double>   velocity_{0.0};
    std::atomic<double>   acceleration_{0.0};
    std::atomic<double>   threshold_{0.5};
    std::atomic<uint64_t> total_{0};
};

} // namespace llmquant
