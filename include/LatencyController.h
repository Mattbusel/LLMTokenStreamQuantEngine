#pragma once

#include <atomic>
#include <chrono>
#include <cstdint>
#include <cmath>
#include <limits>
#include <mutex>
#include <vector>

namespace llmquant {

/**
 * @brief Measures, aggregates and exposes latency statistics for the token-processing pipeline.
 *
 * Hot-path methods (start_measurement / end_measurement / record_latency) use
 * lock-free atomics. Percentile calculation (get_stats) acquires a mutex to
 * copy the sample window; this is acceptable since it is called on the
 * reporting path, not the hot path.
 *
 * Thread safety: all methods are safe to call from multiple threads simultaneously.
 */
class LatencyController {
public:
    /**
     * @brief Construction-time parameters for the controller.
     */
    struct Config {
        /** @brief Desired p99 latency target; used for alerting and profiling. */
        std::chrono::microseconds target_latency{std::chrono::microseconds{10}};
        /** @brief Maximum number of recent samples retained for percentile computation. */
        size_t sample_window{1000};
        /** @brief When true, raw samples are stored so p95/p99 can be calculated. */
        bool enable_profiling{true};
    };

    /**
     * @brief Snapshot of aggregated latency statistics.
     */
    struct LatencyStats {
        std::chrono::microseconds avg_latency{0};
        std::chrono::microseconds min_latency{0};
        std::chrono::microseconds max_latency{0};
        std::chrono::microseconds p5_latency{0};   ///< 5th-percentile latency (fast-path lower bound).
        std::chrono::microseconds p25_latency{0};  ///< 25th-percentile latency (first quartile).
        std::chrono::microseconds p50_latency{0};
        std::chrono::microseconds p75_latency{0};  ///< 75th-percentile latency (third quartile).
        std::chrono::microseconds p95_latency{0};
        std::chrono::microseconds p99_latency{0};
        /** @brief Standard deviation of the sample window in milliseconds. */
        double jitter_ms{0.0};
        /** @brief Total number of measurements recorded since construction or last reset. */
        uint64_t measurements{0};
        /** @brief Number of samples that exceeded the configured target_latency. */
        uint64_t target_breaches{0};
    };

    /**
     * @brief Construct a controller with the given configuration.
     *
     * @param config Construction-time parameters.
     */
    explicit LatencyController(const Config& config);

    /**
     * @brief Record the current high-resolution timestamp as the start of a measurement.
     *
     * Must be paired with a subsequent call to end_measurement() on the same thread.
     */
    void start_measurement();

    /**
     * @brief Compute the elapsed time since start_measurement() and record it.
     *
     * No-op if start_measurement() was not called first on this thread.
     */
    void end_measurement();

    /**
     * @brief Record a pre-computed latency value directly (useful for external timers).
     *
     * @param latency Duration to record in microseconds.
     */
    void record_latency(std::chrono::microseconds latency);

    /**
     * @brief Record a batch of pre-computed latency values.
     *
     * Equivalent to calling record_latency() for each element in order.
     * The sample ring buffer lock is acquired once per call (not once per
     * value), reducing mutex contention in backtest / replay scenarios.
     *
     * @param latencies Ordered sequence of durations to record.
     */
    void record_batch(const std::vector<std::chrono::microseconds>& latencies);

    /**
     * @brief Return a consistent snapshot of all aggregated statistics.
     *
     * If no measurements have been recorded, all fields in the returned struct
     * will be zero.
     *
     * @return LatencyStats snapshot.
     */
    LatencyStats get_stats() const;

    /**
     * @brief Reset all counters and the sample window to their initial states.
     */
    void reset_stats();

    /**
     * @brief Return the configured target latency.
     *
     * @return Target latency as set at construction time.
     */
    std::chrono::microseconds get_target_latency() const noexcept {
        return config_.target_latency;
    }

    /**
     * @brief Hot-reload the controller configuration at runtime.
     *
     * Updates target_latency immediately.  If sample_window changes the ring
     * buffer is resized and all existing samples are discarded.
     * Thread-safe: acquires samples_mutex_.
     *
     * @param config New configuration to apply.
     */
    void update_config(const Config& config);

    /**
     * @brief Profile hook: marks the beginning of token-processing for the next
     *        start_measurement() call. No-op if profiling is disabled.
     */
    void profile_token_processing();

    /**
     * @brief Profile hook: marks the beginning of signal-generation timing.
     *
     * No-op if profiling is disabled.
     */
    void profile_signal_generation();

    /**
     * @brief Profile hook: measures pipeline queue lag.
     *
     * No-op if profiling is disabled.
     */
    void profile_queue_lag();

    /**
     * @brief Composite pressure signal derived from token ingestion rate,
     *        semantic weight variance, and signal queue depth.
     *
     * All pressure values are normalised to [0.0, 1.0] where 1.0 is
     * maximum pressure (shed load / backoff upstream).
     */
    struct PressureState {
        double ingestion_pressure{0.0};   ///< Derived from token arrival rate vs capacity.
        double semantic_pressure{0.0};    ///< Derived from variance in semantic weight scores.
        double queue_pressure{0.0};       ///< Derived from signal queue depth vs max depth.
        double composite{0.0};            ///< max(ingestion, semantic, queue) -- always worst signal.
    };

    /**
     * @brief Consolidated health snapshot for the latency sub-system.
     *
     * Aggregates the most latency-relevant diagnostic fields into a single
     * struct so callers don't need to fan out across get_stats(), get_pressure(),
     * is_warmed_up(), and get_slo_breach_rate() separately.
     */
    struct HealthState {
        bool   slo_met{true};            ///< true if p99 <= target_latency_us.
        bool   warmed_up{false};         ///< true if sample window is >= 50% full.
        double slo_breach_rate{0.0};     ///< Fraction of samples exceeding the target.
        double backoff_multiplier{1.0};  ///< Current backoff multiplier (1.0 = no backoff).
        double budget_remaining_us{0.0}; ///< Signed budget: target_us - p99_us.
    };

    /**
     * @brief Return a consolidated health snapshot.
     *
     * Equivalent to calling get_stats(), get_pressure(), is_warmed_up(),
     * get_slo_breach_rate(), get_latency_budget_remaining_us(), and
     * get_backoff_multiplier() in one go.
     *
     * Thread-safe (delegates to existing thread-safe accessors).
     *
     * @return HealthState snapshot.
     */
    HealthState get_health_state() const;

    /**
     * @brief Update the ingestion pressure component.
     *
     * Call this once per token with the current observed arrival rate
     * (tokens/second) and the maximum sustainable rate.
     *
     * @param arrival_rate_tps Observed tokens per second.
     * @param max_rate_tps     Capacity ceiling (tokens per second); clamped to avoid
     *                         division by zero if zero is passed.
     */
    void update_ingestion_pressure(double arrival_rate_tps, double max_rate_tps);

    /**
     * @brief Update the semantic pressure component from the variance of recent weights.
     *
     * High variance in semantic scores signals conflicting market information.
     *
     * @param weight_variance Variance of recent SemanticWeight::sentiment_score values.
     */
    void update_semantic_pressure(double weight_variance);

    /**
     * @brief Update the queue-depth pressure component.
     *
     * @param queue_depth    Current number of pending signals in the output queue.
     * @param queue_capacity Maximum queue depth before signals are shed.
     */
    void update_queue_pressure(size_t queue_depth, size_t queue_capacity);

    /**
     * @brief Return the most recently computed composite pressure state.
     *
     * @return PressureState snapshot.
     */
    PressureState get_pressure() const;

    /**
     * @brief Return true once the sample window is at least 50% populated.
     *
     * Percentile estimates are unreliable with very few samples.  Callers can
     * call is_warmed_up() before acting on p95/p99 alerts to suppress false
     * positives during the first seconds of a trading session.
     *
     * Thread-safe (acquires samples_mutex_).
     *
     * @return true if get_window_fill_ratio() >= 0.5; false otherwise.
     */
    bool is_warmed_up() const { return get_window_fill_ratio() >= 0.5; }

    /**
     * @brief Return true once the sample window is at least @p fraction populated.
     *
     * @param fraction Minimum fill ratio required, clamped to [0.0, 1.0].
     * @return true if get_window_fill_ratio() >= fraction.
     */
    bool is_warmed_up(double fraction) const {
        double f = fraction < 0.0 ? 0.0 : (fraction > 1.0 ? 1.0 : fraction);
        return get_window_fill_ratio() >= f;
    }

    /**
     * @brief Return the fraction of the sample window currently populated, in [0.0, 1.0].
     *
     * Rises from 0.0 to 1.0 as measurements are recorded up to the configured
     * sample_window size.  Once full, returns 1.0.  Useful for suppressing noisy
     * percentile estimates before warm-up completes.
     *
     * @return Sample window fill ratio in [0.0, 1.0].
     */
    double get_window_fill_ratio() const;

    /**
     * @brief Return the fraction of recorded samples that exceeded the configured
     *        target latency, in [0.0, 1.0].
     *
     * Uses the atomic target_breaches_ counter divided by total_measurements_.
     * Returns 0.0 if no measurements have been recorded yet.
     *
     * @return SLO breach rate in [0.0, 1.0].
     */
    double get_slo_breach_rate() const noexcept;

    /**
     * @brief Compute an arbitrary percentile of the current sample window on demand.
     *
     * Uses the same nearest-rank formula as get_stats().  If profiling is
     * disabled or the sample window is empty, returns zero.
     *
     * @param p Percentile fraction in [0.0, 1.0]; clamped if out of range.
     * @return The p-th percentile latency from the current sample window.
     */
    std::chrono::microseconds get_percentile(double p) const;

    /**
     * @brief Return the current exponential-backoff multiplier for source polling.
     *
     * Starts at 1.0x and increases up to 5.0x as composite pressure rises
     * above 0.8. Resets to 1.0x when pressure drops below 0.5.
     *
     * @return Current backoff multiplier in [1.0, 5.0].
     */
    double get_backoff_multiplier() const;

    /**
     * @brief Reset all pressure components and the backoff multiplier to their
     *        initial values.
     *
     * Clears ingestion_pressure, semantic_pressure, queue_pressure, and
     * composite to 0.0, and resets backoff_multiplier_ to 1.0.  Does NOT
     * affect latency samples or counters.
     *
     * Thread-safe (acquires pressure_mutex_).
     */
    void reset_pressure();

    /**
     * @brief Return the signed latency budget remaining in microseconds.
     *
     * Computed as target_latency_us - p99_latency_us.  A positive value
     * means the p99 is within budget; a negative value means the SLO is
     * being breached.  Returns target_latency_us (full budget) when no
     * samples have been recorded yet.
     *
     * Thread-safe (reads via get_stats()).
     *
     * @return Signed remaining budget in microseconds.
     */
    double get_latency_budget_remaining_us() const;

    /**
     * @brief One bucket of the cumulative latency histogram.
     */
    struct HistogramBucket {
        double   upper_bound_us;  ///< Upper bound in microseconds (inf = +Inf bucket).
        uint64_t count{0};        ///< Cumulative count of samples <= upper_bound_us.
    };

    /**
     * @brief Return a Prometheus-compatible cumulative latency histogram.
     *
     * Buckets (μs): 0.5, 1, 5, 10, 25, 50, 100, 250, 500, 1000, +Inf.
     * Each bucket's count includes all samples at or below its upper bound.
     * If profiling is disabled or no samples have been recorded, all counts
     * will be zero.
     *
     * @return Vector of HistogramBucket with monotonically increasing bounds.
     */
    std::vector<HistogramBucket> histogram_buckets() const;

    /**
     * @brief Estimate measurement throughput since construction or last reset_stats().
     *
     * Computed as total_measurements_ / elapsed_seconds since construction.
     * Returns 0.0 if less than 1 ms has elapsed (avoids divide-by-near-zero).
     *
     * Thread-safe (atomic read + construction_time_ written only at construction).
     *
     * @return Measurements recorded per second.
     */
    double get_throughput_estimate() const noexcept;

    /**
     * @brief Return a human-readable summary of current latency statistics.
     *
     * Format: "n=<count> avg=<avg>µs p50=<p50>µs p95=<p95>µs p99=<p99>µs
     *          min=<min>µs max=<max>µs jitter=<jitter>ms breaches=<breaches>"
     * If no measurements have been recorded, returns "n=0 (no data)".
     *
     * Thread-safe (delegates to get_stats()).
     *
     * @return Single-line stats summary string.
     */
    std::string format_stats() const;

    /**
     * @brief Return the sample variance of the current window in microseconds².
     *
     * Uses Bessel's correction (divides by n-1).  Returns 0.0 if fewer than
     * two samples are in the window or if profiling is disabled.
     *
     * Thread-safe (acquires samples_mutex_).
     *
     * @return Sample variance in µs².
     */
    double get_sample_variance_us() const;

    /**
     * @brief Return the number of valid samples currently in the window.
     *
     * Saturates at sample_window once the ring buffer is full.
     * Thread-safe (acquires samples_mutex_).
     *
     * @return Number of valid samples in [0, sample_window].
     */
    size_t get_sample_count() const;

    /**
     * @brief Return the sample standard deviation of the current window in microseconds.
     *
     * Computed as sqrt(get_sample_variance_us()).  Returns 0.0 if fewer than
     * two samples are in the window or if profiling is disabled.
     *
     * Thread-safe (delegates to get_sample_variance_us()).
     *
     * @return Standard deviation in µs.
     */
    double get_stddev_us() const;

private:
    Config config_;
    std::chrono::high_resolution_clock::time_point construction_time_{
        std::chrono::high_resolution_clock::now()};

    std::atomic<uint64_t> total_measurements_{0};
    std::atomic<uint64_t> total_latency_us_{0};
    std::atomic<uint64_t> min_latency_us_{UINT64_MAX};
    std::atomic<uint64_t> max_latency_us_{0};
    std::atomic<uint64_t> target_breaches_{0};

    std::vector<std::chrono::microseconds> latency_samples_;    ///< Ring buffer, size == config_.sample_window.
    mutable std::vector<std::chrono::microseconds> percentile_scratch_;  ///< Reused buffer for percentile calc.
    size_t sample_head_{0};    ///< Next write index (wraps at sample_window).
    size_t sample_count_{0};   ///< Valid entries (saturates at sample_window).
    mutable std::mutex samples_mutex_;

    mutable std::mutex pressure_mutex_;
    PressureState pressure_;
    double backoff_multiplier_{1.0};  ///< Protected by pressure_mutex_.

    /**
     * @brief Recompute the composite pressure and update the backoff multiplier.
     *
     * Must be called with pressure_mutex_ held.
     */
    void recompute_composite();
};

/**
 * @brief Thread-local start timestamp used by start_measurement / end_measurement.
 *
 * Declared outside the class because thread_local is not permitted on
 * non-static data members in C++.
 */
inline thread_local std::chrono::high_resolution_clock::time_point
    latency_measurement_start_;

/**
 * @brief Guard flag: true only between a paired start_measurement / end_measurement.
 *
 * Prevents end_measurement() from recording a garbage value if called before
 * start_measurement() on a given thread.
 */
inline thread_local bool latency_measurement_active_ = false;

} // namespace llmquant
