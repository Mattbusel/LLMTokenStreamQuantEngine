#pragma once

#include <atomic>
#include <chrono>
#include <cstdint>
#include <functional>
#include <mutex>
#include <string>

namespace llmquant {

/**
 * @brief Circuit-breaker that trips on excessive signal velocity (Δbias/Δtime).
 *
 * The circuit breaker (`PipelineCircuitBreaker`) counts raw signal rejections.
 * This module watches a different threat: a *runaway feedback loop* where
 * accepted signals generate follow-on signals faster than risk gating can
 * dampen them.  Classic symptoms are high-frequency bias oscillations driven
 * by reflexive LLM responses.
 *
 * ## Mechanism
 * On each `record(bias)` call the module computes instantaneous velocity
 * (Δbias / Δt_seconds).  This is smoothed with an EMA.  When the smoothed
 * velocity exceeds `trip_threshold`, the breaker opens and fires
 * `on_trip`.  The breaker resets automatically once velocity falls below
 * `trip_threshold * recovery_factor` for at least one check, or can be
 * reset manually.
 *
 * ## Feature flag
 * Controlled by `LLMQUANT_ENABLE_VELOCITY_BREAKER` (default ON).
 */
class AdaptiveVelocityBreaker {
public:
    using TripCallback     = std::function<void(double velocity)>;
    using RecoveryCallback = std::function<void(double velocity)>;

    struct Config {
        /**
         * @brief Smoothed |Δbias/s| that trips the breaker.
         * Default 10.0 (10 units of bias per second).
         */
        double trip_threshold{10.0};

        /**
         * @brief Fraction of trip_threshold below which recovery occurs.
         * Default 0.5 (velocity must halve before recovery).
         */
        double recovery_factor{0.5};

        /**
         * @brief EMA smoothing factor for velocity (higher = more responsive).
         * Range (0,1].  Default 0.3.
         */
        double velocity_alpha{0.3};

        /**
         * @brief Minimum Δt (seconds) between records to compute velocity.
         * Very small intervals produce noisy measurements.  Default 1e-4.
         */
        double min_dt_s{1e-4};

        /**
         * @brief Called once when the breaker opens (velocity too high).
         */
        TripCallback on_trip;

        /**
         * @brief Called once when the breaker closes (velocity recovered).
         */
        RecoveryCallback on_recovery;
    };

    explicit AdaptiveVelocityBreaker(Config cfg = Config{}) : cfg_(std::move(cfg)) {}

    // Non-copyable.
    AdaptiveVelocityBreaker(const AdaptiveVelocityBreaker&)            = delete;
    AdaptiveVelocityBreaker& operator=(const AdaptiveVelocityBreaker&) = delete;

    /**
     * @brief Record a new bias value and update velocity estimate.
     *
     * @return true if the breaker is OPEN (velocity too high) after this record.
     */
    bool record(double bias) noexcept;

    /**
     * @brief Returns true when the breaker is open (velocity exceeded threshold).
     */
    [[nodiscard]] bool is_open() const noexcept {
        return open_.load(std::memory_order_acquire);
    }

    /**
     * @brief Current smoothed velocity (|Δbias/s|).
     */
    [[nodiscard]] double smoothed_velocity() const noexcept {
        return velocity_ema_.load(std::memory_order_relaxed);
    }

    /**
     * @brief Total number of times the breaker has tripped.
     */
    [[nodiscard]] uint64_t trip_count() const noexcept {
        return trip_count_.load(std::memory_order_relaxed);
    }

    /**
     * @brief Total number of recoveries.
     */
    [[nodiscard]] uint64_t recovery_count() const noexcept {
        return recovery_count_.load(std::memory_order_relaxed);
    }

    /**
     * @brief Force the breaker closed and reset velocity state.
     */
    void reset() noexcept;

    /**
     * @brief Update configuration at runtime.  Resets state.
     */
    void update_config(const Config& cfg);

    /**
     * @brief Serialise current state to a JSON object string.
     */
    [[nodiscard]] std::string to_stats_json() const;

private:
    using Clock     = std::chrono::steady_clock;
    using TimePoint = Clock::time_point;

    mutable std::mutex mtx_;
    Config cfg_;

    double    prev_bias_{0.0};
    TimePoint prev_time_{};
    bool      seeded_{false};

    std::atomic<double>   velocity_ema_{0.0};
    std::atomic<bool>     open_{false};
    std::atomic<uint64_t> trip_count_{0};
    std::atomic<uint64_t> recovery_count_{0};
};

} // namespace llmquant
