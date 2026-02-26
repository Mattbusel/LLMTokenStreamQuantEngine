#pragma once

#include <atomic>
#include <chrono>
#include <cstdint>
#include <cmath>
#include <mutex>
#include <vector>

namespace llmquant {

/// Measures, aggregates and exposes latency statistics for the token-processing pipeline.
///
/// Hot-path methods (start_measurement / end_measurement / record_latency) use
/// lock-free atomics.  Percentile calculation (get_stats) acquires a mutex to
/// copy the sample window; this is acceptable since it is called on the
/// reporting path, not the hot path.
///
/// Thread safety: all methods are safe to call from multiple threads
/// simultaneously.
class LatencyController {
public:
    /// Construction-time parameters for the controller.
    struct Config {
        /// Desired p99 latency target; used for alerting and profiling.
        std::chrono::microseconds target_latency{std::chrono::microseconds{10}};
        /// Maximum number of recent samples retained for percentile computation.
        size_t sample_window{1000};
        /// When true, raw samples are stored so p95/p99 can be calculated.
        bool enable_profiling{true};
    };

    /// Snapshot of aggregated latency statistics.
    struct LatencyStats {
        std::chrono::microseconds avg_latency{0};
        std::chrono::microseconds min_latency{0};
        std::chrono::microseconds max_latency{0};
        std::chrono::microseconds p95_latency{0};
        std::chrono::microseconds p99_latency{0};
        /// Standard deviation of the sample window in milliseconds.
        double jitter_ms{0.0};
        /// Total number of measurements recorded since construction or last reset.
        uint64_t measurements{0};
    };

    /// Construct a controller with the given configuration.
    explicit LatencyController(const Config& config);

    /// Record the current high-resolution timestamp as the start of a measurement.
    ///
    /// Must be paired with a subsequent call to end_measurement() on the same thread.
    void start_measurement();

    /// Compute the elapsed time since start_measurement() and record it.
    void end_measurement();

    /// Record a pre-computed latency value directly (useful for external timers).
    ///
    /// # Arguments
    /// * `latency` — Duration to record in microseconds.
    void record_latency(std::chrono::microseconds latency);

    /// Return a consistent snapshot of all aggregated statistics.
    ///
    /// If no measurements have been recorded, all fields in the returned struct
    /// will be zero.
    LatencyStats get_stats() const;

    /// Reset all counters and the sample window to their initial states.
    void reset_stats();

    /// Profile hook: marks the beginning of token-processing for the next
    /// start_measurement() call.  No-op if profiling is disabled.
    void profile_token_processing();

    /// Profile hook: marks the beginning of signal-generation timing.
    /// No-op if profiling is disabled.
    void profile_signal_generation();

    /// Profile hook: measures pipeline queue lag.
    /// No-op if profiling is disabled.
    void profile_queue_lag();

    /// Composite pressure signal derived from token ingestion rate,
    /// semantic weight variance, and signal queue depth.
    ///
    /// All pressure values are normalised to [0.0, 1.0] where 1.0 is
    /// maximum pressure (shed load / backoff upstream).
    struct PressureState {
        double ingestion_pressure{0.0};   ///< Derived from token arrival rate vs capacity.
        double semantic_pressure{0.0};    ///< Derived from variance in semantic weight scores.
        double queue_pressure{0.0};       ///< Derived from signal queue depth vs max depth.
        double composite{0.0};            ///< max(ingestion, semantic, queue) — always worst signal.
    };

    /// Update the ingestion pressure component.
    ///
    /// Call this once per token with the current observed arrival rate
    /// (tokens/second) and the maximum sustainable rate.
    ///
    /// # Arguments
    /// * `arrival_rate_tps`  — Observed tokens per second.
    /// * `max_rate_tps`      — Capacity ceiling (tokens per second).
    void update_ingestion_pressure(double arrival_rate_tps, double max_rate_tps);

    /// Update the semantic pressure component from the variance of recent weights.
    ///
    /// High variance in semantic scores signals that the market is receiving
    /// conflicting signals — a condition that warrants increased caution.
    ///
    /// # Arguments
    /// * `weight_variance` — Variance of recent SemanticWeight.sentiment_score values.
    void update_semantic_pressure(double weight_variance);

    /// Update the queue-depth pressure component.
    ///
    /// # Arguments
    /// * `queue_depth`    — Current number of pending signals in the output queue.
    /// * `queue_capacity` — Maximum queue depth before signals are shed.
    void update_queue_pressure(size_t queue_depth, size_t queue_capacity);

    /// Return the most recently computed composite pressure state.
    PressureState get_pressure() const;

    /// Return the current exponential-backoff multiplier for source polling.
    ///
    /// Starts at 1.0x and increases up to 5.0x as composite pressure rises
    /// above 0.8.  Resets to 1.0x when pressure drops below 0.5.
    double get_backoff_multiplier() const;

private:
    Config config_;
    std::chrono::high_resolution_clock::time_point measurement_start_;

    std::atomic<uint64_t> total_measurements_{0};
    std::atomic<uint64_t> total_latency_us_{0};
    std::atomic<uint64_t> min_latency_us_{UINT64_MAX};
    std::atomic<uint64_t> max_latency_us_{0};

    std::vector<std::chrono::microseconds> latency_samples_;
    mutable std::mutex samples_mutex_;

    mutable std::mutex pressure_mutex_;
    PressureState pressure_;
    std::atomic<double> backoff_multiplier_{1.0};

    /// Recompute the composite pressure and update the backoff multiplier.
    /// Must be called with pressure_mutex_ held.
    void recompute_composite();
};

} // namespace llmquant
