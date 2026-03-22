#pragma once

/// @file BiasRollingQuantileTracker.h
/// @brief Tracks rolling quantile estimates (P10/P25/P50/P75/P90) of the
///        bias stream using a sliding window with O(N log N) nth_element.
///
/// ## Motivation
/// Mean and variance miss distributional shape.  In a trending regime the
/// bias distribution is right-skewed; in a reverting regime it is symmetric.
/// By tracking the five canonical quantiles we detect skew, kurtosis proxies,
/// and tail-fattening without assuming Gaussianity.
///
/// ## Algorithm
/// - Maintains a ring buffer of the last `window` bias values.
/// - On every `record()` call recomputes all five quantiles via
///   std::nth_element (linear expected time).
/// - Fires `on_median_shift(old_p50, new_p50)` when the median moves by
///   more than `median_change_threshold`.
/// - Fires `on_skew_change(skew_dir)` when (P75-P50) vs (P50-P25) flips
///   sign or changes by more than `skew_threshold`.
///
/// ## Feature flag
/// `LLMQUANT_ENABLE_ROLLING_QUANTILE` (default ON).
///
/// Thread-safe.

#include <atomic>
#include <cstdint>
#include <functional>
#include <mutex>
#include <string>
#include <vector>

namespace llmquant {

class BiasRollingQuantileTracker {
public:
    struct Config {
        /// Rolling window length. Default 64.
        int window{64};

        /// Minimum fill before computing. Default 8.
        int min_samples{8};

        /// |ΔP50| above which on_median_shift fires. Default 0.05.
        double median_change_threshold{0.05};

        /// Change in skew asymmetry ratio (upper_iqr/lower_iqr) to fire on_skew_change. Default 0.3.
        double skew_threshold{0.3};

        /// Fired when median shifts by more than median_change_threshold (outside lock).
        std::function<void(double old_p50, double new_p50)> on_median_shift;

        /// Fired when skew direction changes significantly (outside lock).
        /// Parameter: new skew ratio = (P75-P50)/(P50-P25+eps).
        std::function<void(double skew_ratio)> on_skew_change;
    };

    explicit BiasRollingQuantileTracker(Config cfg = Config{});

    // -----------------------------------------------------------------------
    // Data intake
    // -----------------------------------------------------------------------

    void record(double bias);

    // -----------------------------------------------------------------------
    // Accessors (lock-free)
    // -----------------------------------------------------------------------

    [[nodiscard]] double p10() const noexcept { return p10_.load(std::memory_order_relaxed); }
    [[nodiscard]] double p25() const noexcept { return p25_.load(std::memory_order_relaxed); }
    [[nodiscard]] double p50() const noexcept { return p50_.load(std::memory_order_relaxed); }
    [[nodiscard]] double p75() const noexcept { return p75_.load(std::memory_order_relaxed); }
    [[nodiscard]] double p90() const noexcept { return p90_.load(std::memory_order_relaxed); }

    /// Interquartile range P75 - P25.
    [[nodiscard]] double iqr() const noexcept {
        return p75_.load(std::memory_order_relaxed) -
               p25_.load(std::memory_order_relaxed);
    }

    /// Skew proxy: (P75-P50)/(P50-P25+ε). >1 → right-skewed, <1 → left-skewed.
    [[nodiscard]] double skew_ratio() const noexcept {
        return skew_.load(std::memory_order_relaxed);
    }

    /// Total median-shift events fired.
    [[nodiscard]] uint64_t median_shift_events() const noexcept {
        return median_events_.load(std::memory_order_relaxed);
    }

    /// Total observations recorded.
    [[nodiscard]] uint64_t total_records() const noexcept {
        return total_.load(std::memory_order_relaxed);
    }

    [[nodiscard]] std::string to_stats_json() const;

    void reset();
    void update_config(const Config& cfg);

private:
    void compute_quantiles_locked();

    mutable std::mutex mutex_;
    Config cfg_;

    std::vector<double> win_buf_;
    int win_head_{0};
    int win_fill_{0};

    double prev_p50_{0.0};
    double prev_skew_{1.0};
    bool   seeded_{false};

    std::atomic<double>   p10_{0.0};
    std::atomic<double>   p25_{0.0};
    std::atomic<double>   p50_{0.0};
    std::atomic<double>   p75_{0.0};
    std::atomic<double>   p90_{0.0};
    std::atomic<double>   skew_{1.0};
    std::atomic<uint64_t> median_events_{0};
    std::atomic<uint64_t> total_{0};
};

} // namespace llmquant
