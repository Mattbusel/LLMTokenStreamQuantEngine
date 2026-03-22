#pragma once

/// @file SignalChoppinessIndex.h
/// @brief Computes the Choppiness Index (CI) for the LLM bias stream, measuring
///        whether the stream is trending or ranging/noisy.
///
/// ## Motivation
/// The Choppiness Index (Hutchinson, 2007) quantifies "choppiness":
///
///   CI = 100 * log10(ATR_sum / (max - min + ε)) / log10(N)
///
/// where ATR_sum = sum of |x_t - x_{t-1}| over N periods (total path length),
///       max, min = range of the signal over N periods,
///       N = window length.
///
/// CI ∈ [0, 100]:
///   CI → 100 → maximum choppiness (pure noise, no trend).
///   CI → 0   → maximum trending (all moves in one direction).
///   CI ≈ 61.8 → random walk / neutral (golden ratio boundary).
///
/// Fires `on_trending(ci)` when CI < `trending_threshold` (typically 38.2).
/// Fires `on_choppy(ci)` when CI > `choppy_threshold` (typically 61.8).
///
/// ## Feature flag
/// `LLMQUANT_ENABLE_CHOPPINESS_INDEX` (default ON).
///
/// Thread-safe.

#include <atomic>
#include <cstdint>
#include <functional>
#include <mutex>
#include <string>
#include <vector>

namespace llmquant {

class SignalChoppinessIndex {
public:
    struct Config {
        /// Rolling window N. Default 14.
        int window{14};

        /// Minimum fill before computing. Default 4.
        int min_samples{4};

        /// CI < this fires on_trending. Default 38.2.
        double trending_threshold{38.2};

        /// CI > this fires on_choppy. Default 61.8.
        double choppy_threshold{61.8};

        /// Fired when CI < trending_threshold (outside lock).
        std::function<void(double ci)> on_trending;

        /// Fired when CI > choppy_threshold (outside lock).
        std::function<void(double ci)> on_choppy;
    };

    explicit SignalChoppinessIndex(Config cfg = Config{});

    // -----------------------------------------------------------------------
    // Data intake
    // -----------------------------------------------------------------------

    void record(double bias);

    // -----------------------------------------------------------------------
    // Accessors (lock-free)
    // -----------------------------------------------------------------------

    /// Current Choppiness Index CI ∈ [0, 100].
    [[nodiscard]] double choppiness() const noexcept {
        return ci_.load(std::memory_order_relaxed);
    }

    /// True when CI < trending_threshold.
    [[nodiscard]] bool is_trending() const noexcept {
        return ci_.load(std::memory_order_relaxed) < cfg_.trending_threshold;
    }

    /// True when CI > choppy_threshold.
    [[nodiscard]] bool is_choppy() const noexcept {
        return ci_.load(std::memory_order_relaxed) > cfg_.choppy_threshold;
    }

    /// Total trending events fired.
    [[nodiscard]] uint64_t trending_events() const noexcept {
        return trend_events_.load(std::memory_order_relaxed);
    }

    /// Total choppy events fired.
    [[nodiscard]] uint64_t choppy_events() const noexcept {
        return choppy_events_.load(std::memory_order_relaxed);
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

    bool prev_trending_{false};
    bool prev_choppy_{false};

    std::atomic<double>   ci_{61.8};  // start at neutral/choppy
    std::atomic<uint64_t> trend_events_{0};
    std::atomic<uint64_t> choppy_events_{0};
    std::atomic<uint64_t> total_{0};
};

} // namespace llmquant
