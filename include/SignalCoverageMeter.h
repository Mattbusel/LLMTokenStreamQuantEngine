#pragma once

/// @file SignalCoverageMeter.h
/// @brief Measures the fraction of the bias range actually utilized by signals.
///
/// ## Motivation
/// If all signals cluster in [-0.1, 0.1] of a configured range [-1, 1],
/// only 10% of the dynamic range is being used.  Low coverage signals that
/// the signal engine is effectively saturating at near-zero, or that the
/// token stream is producing very weak sentiment.  High coverage means the
/// engine is exploiting the full dynamic range.
///
/// ## Algorithm
/// Maintains a rolling min and max of bias values over the last `window`
/// observations.  Coverage score:
///
///   coverage = (rolling_max - rolling_min) / (range_max - range_min)
///   coverage = clamp(coverage, 0, 1)
///
/// Fires `on_range_expansion(coverage)` when coverage first crosses above
/// `expansion_threshold`.
///
/// ## Feature flag
/// `LLMQUANT_ENABLE_COVERAGE_METER` (default ON).
///
/// Thread-safe.

#include <atomic>
#include <cstdint>
#include <functional>
#include <mutex>
#include <string>
#include <vector>

namespace llmquant {

class SignalCoverageMeter {
public:
    struct Config {
        /// Rolling window size. Default 50.
        int window{50};

        /// Full bias range minimum. Default -1.0.
        double range_min{-1.0};

        /// Full bias range maximum. Default 1.0.
        double range_max{1.0};

        /// Coverage fraction above which on_range_expansion fires. Default 0.5.
        double expansion_threshold{0.5};

        /// Fired when coverage first crosses above expansion_threshold (outside lock).
        std::function<void(double coverage)> on_range_expansion;
    };

    explicit SignalCoverageMeter(Config cfg = Config{});

    // -----------------------------------------------------------------------
    // Data intake
    // -----------------------------------------------------------------------

    /// Record a new bias value.
    void record(double bias);

    // -----------------------------------------------------------------------
    // Accessors (lock-free)
    // -----------------------------------------------------------------------

    /// Current coverage score ∈ [0, 1].
    [[nodiscard]] double coverage() const noexcept {
        return coverage_.load(std::memory_order_relaxed);
    }

    /// Rolling minimum bias value in the window.
    [[nodiscard]] double rolling_min() const noexcept {
        return rolling_min_.load(std::memory_order_relaxed);
    }

    /// Rolling maximum bias value in the window.
    [[nodiscard]] double rolling_max() const noexcept {
        return rolling_max_.load(std::memory_order_relaxed);
    }

    /// True when coverage > expansion_threshold.
    [[nodiscard]] bool is_expanded() const noexcept {
        return expanded_.load(std::memory_order_relaxed);
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

    std::vector<double> buf_;
    int    head_{0};
    int    fill_{0};
    bool   prev_expanded_{false};

    std::atomic<double>   coverage_{0.0};
    std::atomic<double>   rolling_min_{0.0};
    std::atomic<double>   rolling_max_{0.0};
    std::atomic<bool>     expanded_{false};
    std::atomic<uint64_t> obs_count_{0};
};

} // namespace llmquant
