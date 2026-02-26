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

private:
    Config config_;
    std::chrono::high_resolution_clock::time_point measurement_start_;

    std::atomic<uint64_t> total_measurements_{0};
    std::atomic<uint64_t> total_latency_us_{0};
    std::atomic<uint64_t> min_latency_us_{UINT64_MAX};
    std::atomic<uint64_t> max_latency_us_{0};

    std::vector<std::chrono::microseconds> latency_samples_;
    mutable std::mutex samples_mutex_;
};

} // namespace llmquant
