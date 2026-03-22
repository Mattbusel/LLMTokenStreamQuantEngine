#pragma once

/// @file BiasLevelCrossing.h
/// @brief Tracks zero-crossing and level-crossing rates of the bias stream,
///        providing a frequency-domain proxy without FFT.
///
/// ## Motivation
/// The rate at which a time series crosses a given level (e.g., zero) is
/// directly related to its dominant frequency content.  High zero-crossing
/// rate → noisy, mean-reverting, high-frequency signal.  Low rate → trending,
/// low-frequency signal.  Unlike spectral entropy this is O(1) per sample.
///
/// ## Algorithm
/// - Tracks crossings of `n_levels` equally-spaced thresholds in [-1, +1].
/// - Computes rolling crossing rate: crossings per sample in last `window`.
/// - Primary metric: zero-crossing rate (ZCR) as the most stable indicator.
/// - Fires `on_zcr_change(old_zcr, new_zcr)` when ZCR changes by more than
///   `zcr_change_threshold` — a regime shift in signal frequency content.
///
/// ## Feature flag
/// `LLMQUANT_ENABLE_LEVEL_CROSSING` (default ON).
///
/// Thread-safe.

#include <atomic>
#include <cstdint>
#include <functional>
#include <mutex>
#include <string>
#include <vector>

namespace llmquant {

class BiasLevelCrossing {
public:
    struct Config {
        /// Rolling window for crossing rate computation. Default 50.
        int window{50};

        /// Minimum fill before reporting. Default 10.
        int min_samples{10};

        /// ZCR change above this fires on_zcr_change. Default 0.1.
        double zcr_change_threshold{0.1};

        /// Fired when zero-crossing rate changes significantly (outside lock).
        std::function<void(double old_zcr, double new_zcr)> on_zcr_change;
    };

    explicit BiasLevelCrossing(Config cfg = Config{});

    // -----------------------------------------------------------------------
    // Data intake
    // -----------------------------------------------------------------------

    void record(double bias);

    // -----------------------------------------------------------------------
    // Accessors (lock-free)
    // -----------------------------------------------------------------------

    /// Zero-crossing rate (crossings per sample) in the rolling window.
    [[nodiscard]] double zero_crossing_rate() const noexcept {
        return zcr_.load(std::memory_order_relaxed);
    }

    /// Mean crossing rate across all tracked levels.
    [[nodiscard]] double mean_crossing_rate() const noexcept {
        return mcr_.load(std::memory_order_relaxed);
    }

    /// Total ZCR-change events fired.
    [[nodiscard]] uint64_t zcr_change_events() const noexcept {
        return change_events_.load(std::memory_order_relaxed);
    }

    /// Total observations recorded.
    [[nodiscard]] uint64_t total_records() const noexcept {
        return total_.load(std::memory_order_relaxed);
    }

    [[nodiscard]] std::string to_stats_json() const;

    void reset();
    void update_config(const Config& cfg);

private:
    mutable std::mutex mutex_;
    Config cfg_;

    std::vector<double> win_buf_;
    int win_head_{0};
    int win_fill_{0};

    double prev_bias_{0.0};
    bool   seeded_{false};
    double prev_zcr_{-1.0};

    std::atomic<double>   zcr_{0.0};
    std::atomic<double>   mcr_{0.0};
    std::atomic<uint64_t> change_events_{0};
    std::atomic<uint64_t> total_{0};
};

} // namespace llmquant
