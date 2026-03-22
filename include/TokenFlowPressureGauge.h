#pragma once

/// @file TokenFlowPressureGauge.h
/// @brief Real-time token arrival pressure estimator.
///
/// ## Motivation
/// The rate at which LLM tokens arrive carries signal about stream health and
/// upstream load.  When tokens arrive faster than the target rate, downstream
/// components (LatencyController, risk engine) experience elevated pressure.
/// When the rate drops far below target, the stream may be stalling.
///
/// ## Algorithm
/// Maintains a Wilder-style EMA of successive inter-token arrival intervals:
///   ema_interval = alpha * interval + (1 - alpha) * ema_interval
///
/// Pressure score ∈ [0, 1]:
///   pressure = clamp(target_interval_ns / ema_interval_ns, 0.0, 1.0)
///
/// A score of 1.0 means tokens are arriving AT or ABOVE the target rate.
/// A score near 0 means the stream has slowed almost to a halt.
///
/// Fires `on_pressure_spike(pressure)` (outside the lock) whenever the score
/// first crosses above `spike_threshold` and remains there.
///
/// ## Feature flag
/// `LLMQUANT_ENABLE_FLOW_PRESSURE` (default ON).
///
/// Thread-safe.

#include <atomic>
#include <cstdint>
#include <functional>
#include <mutex>
#include <string>

namespace llmquant {

class TokenFlowPressureGauge {
public:
    struct Config {
        /// Target inter-token interval in nanoseconds (default 33 ms ≈ 30 tok/s).
        uint64_t target_interval_ns{33'000'000ULL};

        /// EMA smoothing factor (Wilder: 1/period). Default 0.10.
        double ema_alpha{0.10};

        /// Pressure score above which on_pressure_spike fires. Default 0.85.
        double spike_threshold{0.85};

        /// Fired when pressure crosses above spike_threshold.
        std::function<void(double pressure)> on_pressure_spike;
    };

    explicit TokenFlowPressureGauge(Config cfg = Config{});

    // -----------------------------------------------------------------------
    // Data intake
    // -----------------------------------------------------------------------

    /// Record the arrival of a new token at time `now_ns` (nanoseconds since epoch).
    void record_token(uint64_t now_ns);

    // -----------------------------------------------------------------------
    // Accessors (lock-free)
    // -----------------------------------------------------------------------

    /// Pressure score ∈ [0, 1]. 1 = at or above target rate.
    [[nodiscard]] double pressure() const noexcept {
        return pressure_.load(std::memory_order_relaxed);
    }

    /// True when pressure score exceeds spike_threshold.
    [[nodiscard]] bool is_spiking() const noexcept {
        return spiking_.load(std::memory_order_relaxed);
    }

    /// Number of tokens recorded.
    [[nodiscard]] uint64_t token_count() const noexcept {
        return token_count_.load(std::memory_order_relaxed);
    }

    [[nodiscard]] std::string to_stats_json() const;

    void reset();
    void update_config(const Config& cfg);

private:
    mutable std::mutex mutex_;
    Config cfg_;

    uint64_t last_ns_{0};
    double   ema_interval_ns_{0.0};
    bool     seeded_{false};

    bool prev_spiking_{false};

    std::atomic<double>   pressure_{0.0};
    std::atomic<bool>     spiking_{false};
    std::atomic<uint64_t> token_count_{0};
};

} // namespace llmquant
