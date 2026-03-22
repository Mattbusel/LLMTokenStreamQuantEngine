#pragma once

/// @file LatencyJitterMonitor.h
/// @brief MAD-based latency jitter tracker for token-processing consistency.
///
/// ## Motivation
/// LatencyController tracks mean and percentile latencies — but percentiles alone
/// miss consistency.  A p99 of 8 μs is fine; a p99 of 8 μs that randomly spikes
/// to 800 μs every 30 seconds is not.  Median Absolute Deviation (MAD) captures
/// this spread without being distorted by rare outliers.
///
/// High MAD → the engine is processing inconsistently → upstream backpressure
/// may be necessary before p99 breaches the SLA budget.
///
/// ## Algorithm
/// Maintains a rolling window of per-token latency samples (μs).
/// On every record:
///   1. Evict the oldest sample (ring buffer).
///   2. Compute median of the window.
///   3. MAD = median(|x_i − median|) across the window.
///   4. If MAD > jitter_threshold_us and was below → fire on_jitter_spike.
///   5. If MAD < jitter_threshold_us × clear_hysteresis and was above → fire on_jitter_clear.
///
/// MAD computation is O(N log N) inside the mutex; N is bounded by window_size.
///
/// ## Feature flag
/// `LLMQUANT_ENABLE_LATENCY_JITTER` (default ON).
///
/// Thread-safe.

#include <atomic>
#include <cstdint>
#include <functional>
#include <mutex>
#include <string>
#include <vector>

namespace llmquant {

class LatencyJitterMonitor {
public:
    struct Config {
        /// Rolling window size (number of latency samples). Default 64.
        int window_size{64};

        /// MAD threshold in microseconds above which jitter is flagged. Default 500 μs.
        double jitter_threshold_us{500.0};

        /// Hysteresis: on_jitter_clear fires when MAD drops below
        /// jitter_threshold_us × clear_hysteresis. Default 0.75.
        double clear_hysteresis{0.75};

        /// Minimum samples before computing MAD. Default 8.
        int min_samples{8};

        /// Fired (outside lock) when MAD crosses above jitter_threshold_us.
        /// Parameter: current MAD in microseconds.
        std::function<void(double mad_us)> on_jitter_spike;

        /// Fired (outside lock) when MAD drops back below the clear threshold.
        /// Parameter: current MAD in microseconds.
        std::function<void(double mad_us)> on_jitter_clear;
    };

    explicit LatencyJitterMonitor(Config cfg = Config{});

    // -----------------------------------------------------------------------
    // Data intake
    // -----------------------------------------------------------------------

    /// Record a latency sample in microseconds.
    void record(double latency_us);

    // -----------------------------------------------------------------------
    // Accessors (lock-free)
    // -----------------------------------------------------------------------

    /// Current MAD in microseconds. 0 if insufficient data.
    [[nodiscard]] double mad_us() const noexcept {
        return mad_.load(std::memory_order_relaxed);
    }

    /// Median latency of the current window in microseconds.
    [[nodiscard]] double median_us() const noexcept {
        return median_.load(std::memory_order_relaxed);
    }

    /// True when MAD > jitter_threshold_us.
    [[nodiscard]] bool is_jittery() const noexcept {
        return jittery_.load(std::memory_order_relaxed);
    }

    /// Total latency samples recorded (including evicted ones).
    [[nodiscard]] uint64_t total_samples() const noexcept {
        return total_.load(std::memory_order_relaxed);
    }

    /// Number of times jitter threshold was crossed (rising edges only).
    [[nodiscard]] uint64_t jitter_events() const noexcept {
        return events_.load(std::memory_order_relaxed);
    }

    [[nodiscard]] std::string to_stats_json() const;

    void reset();
    void update_config(const Config& cfg);

private:
    void recompute_locked();

    mutable std::mutex mutex_;
    Config cfg_;

    std::vector<double> buf_;
    int  head_{0};
    int  fill_{0};
    bool prev_jittery_{false};

    std::atomic<double>   mad_{0.0};
    std::atomic<double>   median_{0.0};
    std::atomic<bool>     jittery_{false};
    std::atomic<uint64_t> total_{0};
    std::atomic<uint64_t> events_{0};
};

} // namespace llmquant
