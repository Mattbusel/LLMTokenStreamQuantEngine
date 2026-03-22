#pragma once

/// @file TokenClockRecalibrator.h
/// @brief Adaptive token-emission-rate estimator that dynamically rescales
///        latency budgets when LLM generation speed fluctuates.
///
/// ## Motivation
/// LLM inference throughput is not constant: model load, batch coalescing, and
/// KV-cache pressure all cause token inter-arrival times to vary by 2-5×.
/// If the engine's latency budget assumes a fixed token rate, it will either
/// drop tokens (budget too tight) or accumulate stale signals (budget too loose).
///
/// ## Algorithm
/// - Records inter-arrival timestamps via `record_token(timestamp_ns)`.
/// - Maintains a sliding window of the last `window_size` inter-arrival gaps.
/// - Computes: `rate_hz` = 1 / mean_gap_s, `jitter_cv` = std_gap / mean_gap.
/// - Outputs a `budget_scale_factor` = `target_rate_hz / rate_hz` clamped to
///   `[min_scale, max_scale]`: callers multiply their base latency budget by
///   this factor to track the actual LLM speed.
/// - Fires `on_rate_change(new_rate_hz, budget_scale)` when rate shifts by
///   more than `rate_change_threshold` (fractional).
///
/// ## Feature flag
/// `LLMQUANT_ENABLE_TOKEN_CLOCK` (default ON).
///
/// Thread-safe.

#include <atomic>
#include <cstdint>
#include <functional>
#include <mutex>
#include <string>
#include <vector>

namespace llmquant {

class TokenClockRecalibrator {
public:
    struct Config {
        /// Sliding window of inter-arrival intervals (samples). Default 64.
        int window_size{64};

        /// Expected / target token rate (tokens per second). Default 30.
        double target_rate_hz{30.0};

        /// Minimum allowed budget scale factor. Default 0.25.
        double min_scale{0.25};

        /// Maximum allowed budget scale factor. Default 4.0.
        double max_scale{4.0};

        /// Fractional rate change that triggers the callback. Default 0.15.
        double rate_change_threshold{0.15};

        /// Fired when estimated token rate changes significantly.
        /// Parameters: (new_rate_hz, budget_scale_factor).
        std::function<void(double rate_hz, double budget_scale)> on_rate_change;
    };

    explicit TokenClockRecalibrator(Config cfg = Config{});

    // -----------------------------------------------------------------------
    // Data intake
    // -----------------------------------------------------------------------

    /// Record the arrival of one token at `timestamp_ns` nanoseconds (monotonic).
    void record_token(uint64_t timestamp_ns);

    // -----------------------------------------------------------------------
    // Accessors (lock-free)
    // -----------------------------------------------------------------------

    /// Estimated token emission rate (tokens/second).
    [[nodiscard]] double rate_hz() const noexcept {
        return rate_hz_.load(std::memory_order_relaxed);
    }

    /// Current budget scale factor (multiply your base budget by this).
    [[nodiscard]] double budget_scale() const noexcept {
        return budget_scale_.load(std::memory_order_relaxed);
    }

    /// Coefficient of variation of inter-arrival gaps (jitter measure).
    [[nodiscard]] double jitter_cv() const noexcept {
        return jitter_cv_.load(std::memory_order_relaxed);
    }

    /// Total tokens recorded.
    [[nodiscard]] uint64_t token_count() const noexcept {
        return token_count_.load(std::memory_order_relaxed);
    }

    /// Total rate-change events fired.
    [[nodiscard]] uint64_t rate_change_count() const noexcept {
        return rate_change_count_.load(std::memory_order_relaxed);
    }

    /// JSON diagnostics snapshot.
    [[nodiscard]] std::string to_stats_json() const;

    /// Reset all state.
    void reset();

    /// Update configuration (resets state).
    void update_config(const Config& cfg);

private:
    void recompute_locked();

    mutable std::mutex mutex_;
    Config cfg_;

    std::vector<double> gaps_;   // ring buffer of inter-arrival seconds
    int    head_{0};
    int    fill_{0};

    uint64_t last_ts_{0};
    bool     seeded_{false};

    double prev_rate_hz_{0.0};

    std::atomic<double>   rate_hz_{0.0};
    std::atomic<double>   budget_scale_{1.0};
    std::atomic<double>   jitter_cv_{0.0};
    std::atomic<uint64_t> token_count_{0};
    std::atomic<uint64_t> rate_change_count_{0};
};

} // namespace llmquant
