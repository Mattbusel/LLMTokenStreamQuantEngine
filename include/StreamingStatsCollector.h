#pragma once

/// @file StreamingStatsCollector.h
/// @brief Real-time throughput and latency percentile statistics for the LLM
///        token stream pipeline.
///
/// ## Purpose
/// Observability is critical for production LLM trading systems.  This
/// collector tracks:
///
/// - **Throughput**: tokens/second using a sliding time window.
/// - **Latency percentiles**: P50, P75, P90, P95, P99 computed via an
///   online order-statistics reservoir (T-Digest approximation using
///   fixed-bucket histogram approach for zero-allocation hot path).
/// - **Inter-token gap**: mean and max interval between consecutive tokens.
/// - **Byte throughput**: optional bytes/second if token lengths are recorded.
///
/// ## Algorithm
/// Latency percentiles use a fixed 256-bucket HDR-style histogram spanning
/// [0 µs, `max_latency_us`] with logarithmic bucket boundaries.  The histogram
/// supports cumulative lookups for any percentile in O(256) time.
///
/// Throughput is measured by counting tokens in a sliding window of
/// `throughput_window_ms` milliseconds using a ring buffer of epoch counts.
///
/// ## Thread safety
/// All methods are thread-safe via a single mutex.
///
/// ## Feature flag
/// `LLMQUANT_ENABLE_STREAMING_STATS` (default ON).

#include <array>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <functional>
#include <mutex>
#include <string>

namespace llmquant {

class StreamingStatsCollector {
public:
    // -----------------------------------------------------------------------
    // Histogram parameters
    // -----------------------------------------------------------------------

    static constexpr int kHistogramBuckets = 256;
    static constexpr int kThroughputSlots  = 64;   ///< Ring slots for throughput window.

    // -----------------------------------------------------------------------
    // Configuration
    // -----------------------------------------------------------------------

    struct Config {
        /// Maximum latency tracked by histogram (µs).  Observations above this
        /// are counted in the overflow bucket.  Default 1 000 000 µs (1 s).
        int64_t max_latency_us{1'000'000};

        /// Sliding window for throughput computation (ms).  Default 5 000 ms.
        int throughput_window_ms{5000};

        /// Minimum expected inter-token gap (µs) before "high gap" callback fires.
        int64_t gap_alert_us{500'000};  // 500 ms gap = stream may be stalling

        /// Fired when p99 latency exceeds this threshold (µs).
        int64_t p99_alert_us{50'000};

        /// Callback: (p99_latency_us, throughput_tps) fired when p99 exceeds alert.
        std::function<void(int64_t p99_us, double tps)> on_p99_alert;

        /// Callback: (gap_us) fired when inter-token gap exceeds alert threshold.
        std::function<void(int64_t gap_us)> on_high_gap;
    };

    // -----------------------------------------------------------------------
    // Latency snapshot
    // -----------------------------------------------------------------------

    struct LatencyPercentiles {
        int64_t p50_us{0};
        int64_t p75_us{0};
        int64_t p90_us{0};
        int64_t p95_us{0};
        int64_t p99_us{0};
        int64_t min_us{0};
        int64_t max_us{0};
        uint64_t count{0};
    };

    // -----------------------------------------------------------------------
    // Construction
    // -----------------------------------------------------------------------

    explicit StreamingStatsCollector(Config cfg = Config{});

    // -----------------------------------------------------------------------
    // Data intake
    // -----------------------------------------------------------------------

    /// Record a latency observation (time from token arrival to processing).
    /// @param latency_us  Observed latency in microseconds.
    /// @param token_bytes Optional byte length of the token (for byte throughput).
    void record_latency(int64_t latency_us, int token_bytes = 0);

    /// Record the current wall-clock time as a "token arrived" event.
    /// Used to compute throughput and inter-token gap.
    void record_arrival();

    /// Combined: record arrival AND latency in one call.
    void record(int64_t latency_us, int token_bytes = 0);

    // -----------------------------------------------------------------------
    // Queries
    // -----------------------------------------------------------------------

    /// Compute latency percentiles from the histogram.
    [[nodiscard]] LatencyPercentiles latency_percentiles() const;

    /// Tokens per second over the throughput window.
    [[nodiscard]] double throughput_tps() const;

    /// Bytes per second (requires token_bytes to be recorded).
    [[nodiscard]] double throughput_bps() const;

    /// Mean inter-token gap (µs).
    [[nodiscard]] double mean_gap_us() const noexcept {
        return mean_gap_us_.load(std::memory_order_relaxed);
    }

    /// Maximum observed inter-token gap (µs).
    [[nodiscard]] int64_t max_gap_us() const noexcept {
        return max_gap_us_.load(std::memory_order_relaxed);
    }

    /// Total observations recorded.
    [[nodiscard]] uint64_t total_count() const noexcept {
        return total_count_.load(std::memory_order_relaxed);
    }

    /// Total byte volume recorded.
    [[nodiscard]] uint64_t total_bytes() const noexcept {
        return total_bytes_.load(std::memory_order_relaxed);
    }

    /// Histogram overflow count (latency > max_latency_us).
    [[nodiscard]] uint64_t overflow_count() const noexcept {
        return overflow_count_.load(std::memory_order_relaxed);
    }

    // -----------------------------------------------------------------------
    // Control
    // -----------------------------------------------------------------------

    void reset();
    void update_config(const Config& cfg);

    [[nodiscard]] std::string to_stats_json() const;

private:
    // Compute bucket index for a latency value.
    [[nodiscard]] int bucket_for(int64_t latency_us) const noexcept;

    // Recover latency from bucket index (lower bound of bucket).
    [[nodiscard]] int64_t bucket_lower(int bucket) const noexcept;

    // Advance the throughput ring to the current epoch slot.
    void advance_throughput_epoch_locked();

    mutable std::mutex mutex_;
    Config cfg_;

    // ----- Latency histogram -----
    // Fixed-bucket HDR-style: bucket i spans [lower_i, lower_{i+1}).
    // Boundaries: lower_i = round(max_us * i^2 / (N-1)^2) for quadratic spacing.
    std::array<uint64_t, kHistogramBuckets> hist_{};
    uint64_t overflow_count_val_{0};

    int64_t hist_min_{INT64_MAX};
    int64_t hist_max_{0};

    // ----- Throughput ring -----
    // Each slot covers throughput_window_ms / kThroughputSlots milliseconds.
    struct ThroughputSlot {
        int64_t  epoch_ms{0};   ///< Start epoch of this slot.
        uint64_t count{0};
        uint64_t bytes{0};
    };
    std::array<ThroughputSlot, kThroughputSlots> tp_ring_{};
    int     tp_head_{0};        ///< Current (newest) ring slot.
    int64_t tp_slot_ms_{0};     ///< Milliseconds per slot.

    // ----- Inter-token gap -----
    int64_t last_arrival_us_{0};   ///< Wall clock µs of last arrival.
    double  gap_ema_{0.0};         ///< EMA of inter-token gap.
    static constexpr double kGapAlpha = 0.05;

    // ----- Atomics (fast reads without lock) -----
    std::atomic<uint64_t> total_count_{0};
    std::atomic<uint64_t> total_bytes_{0};
    std::atomic<uint64_t> overflow_count_{0};
    std::atomic<double>   mean_gap_us_{0.0};
    std::atomic<int64_t>  max_gap_us_{0};
};

} // namespace llmquant
