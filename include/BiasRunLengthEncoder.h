#pragma once

/// @file BiasRunLengthEncoder.h
/// @brief Run-length statistics on the bias sign sequence.
///
/// ## Motivation
/// A run-length encoding of the sign of bias reveals persistence.
/// Long bullish runs → strongly trending bullish regime.
/// Frequent alternation (short runs) → noisy, mean-reverting regime.
/// Maximum run length is a measure of "conviction length".
///
/// ## Algorithm
/// Each `record(bias)` determines the sign (+1, -1, 0) of the value.
/// If the sign matches the current run, the run is extended.
/// On a sign change, the finished run's length is recorded and statistics
/// are updated:
///   - current_run: length of the in-progress run
///   - max_run:     longest run observed
///   - avg_run:     EMA-smoothed average run length (alpha=0.2)
///   - run_count:   total number of completed runs
///
/// Fires `on_long_run(direction, length)` when a completed run exceeds
/// `long_run_threshold` (default 10).
///
/// ## Feature flag
/// `LLMQUANT_ENABLE_RUN_LENGTH` (default ON).
///
/// Thread-safe.

#include <atomic>
#include <cstdint>
#include <functional>
#include <mutex>
#include <string>

namespace llmquant {

class BiasRunLengthEncoder {
public:
    struct Config {
        /// Completed run length above which on_long_run fires. Default 10.
        int long_run_threshold{10};

        /// EMA alpha for smoothing avg_run. Default 0.2.
        double avg_alpha{0.2};

        /// Fired when a completed run length >= long_run_threshold (outside lock).
        std::function<void(int direction, int length)> on_long_run;
    };

    explicit BiasRunLengthEncoder(Config cfg = Config{});

    // -----------------------------------------------------------------------
    // Data intake
    // -----------------------------------------------------------------------

    /// Record a new bias value.
    void record(double bias);

    // -----------------------------------------------------------------------
    // Accessors (lock-free)
    // -----------------------------------------------------------------------

    /// Length of the currently in-progress run.
    [[nodiscard]] int current_run() const noexcept {
        return current_run_.load(std::memory_order_relaxed);
    }

    /// Direction of the current run: +1, -1, or 0.
    [[nodiscard]] int current_direction() const noexcept {
        return current_dir_.load(std::memory_order_relaxed);
    }

    /// Longest completed run observed.
    [[nodiscard]] int max_run() const noexcept {
        return max_run_.load(std::memory_order_relaxed);
    }

    /// EMA-smoothed average completed run length.
    [[nodiscard]] double avg_run() const noexcept {
        return avg_run_.load(std::memory_order_relaxed);
    }

    /// Number of completed runs.
    [[nodiscard]] uint64_t run_count() const noexcept {
        return run_count_.load(std::memory_order_relaxed);
    }

    /// Total observations recorded.
    [[nodiscard]] uint64_t observation_count() const noexcept {
        return obs_count_.load(std::memory_order_relaxed);
    }

    [[nodiscard]] std::string to_stats_json() const;

    void reset();
    void update_config(const Config& cfg);

private:
    mutable std::mutex mutex_;
    Config cfg_;

    int    cur_dir_{0};
    int    cur_len_{0};
    int    max_len_{0};
    double avg_len_{0.0};
    bool   avg_seeded_{false};

    std::atomic<int>      current_run_{0};
    std::atomic<int>      current_dir_{0};
    std::atomic<int>      max_run_{0};
    std::atomic<double>   avg_run_{0.0};
    std::atomic<uint64_t> run_count_{0};
    std::atomic<uint64_t> obs_count_{0};
};

} // namespace llmquant
