#pragma once

/// @file SignalConfluenceDetector.h
/// @brief Detects when multiple recent signals agree in direction (confluence).
///
/// ## Motivation
/// A single signal may be noise.  When the last N signals all point in the
/// same direction, that "confluence" is a much stronger indication of a
/// sustained regime.  Conversely, a mix of bullish and bearish signals over
/// the same window indicates indecision.
///
/// ## Algorithm
/// Maintains a circular buffer of the last `window` signal directions.
/// Confluence score = (dominant_count - minority_count) / window ∈ [-1, 1]
/// where +1 = all bullish, -1 = all bearish, 0 = equal split.
///
/// Fires `on_confluence(score, direction)` when |score| first crosses
/// `threshold` (default 0.6).
///
/// ## Feature flag
/// `LLMQUANT_ENABLE_CONFLUENCE_DETECTOR` (default ON).
///
/// Thread-safe.

#include <atomic>
#include <cstdint>
#include <functional>
#include <mutex>
#include <string>
#include <vector>

namespace llmquant {

class SignalConfluenceDetector {
public:
    struct Config {
        /// Window of recent signal directions. Default 10.
        int window{10};

        /// |score| above which on_confluence fires. Default 0.6.
        double threshold{0.6};

        /// Fired when |score| first crosses threshold (outside lock).
        std::function<void(double score, int direction)> on_confluence;
    };

    explicit SignalConfluenceDetector(Config cfg = Config{});

    void record(double bias);

    /// Confluence score ∈ [-1, 1]. +1 = all bullish, -1 = all bearish.
    [[nodiscard]] double confluence_score() const noexcept {
        return score_.load(std::memory_order_relaxed);
    }
    /// Dominant direction: +1, -1, or 0.
    [[nodiscard]] int dominant_direction() const noexcept {
        return dominant_.load(std::memory_order_relaxed);
    }
    [[nodiscard]] bool is_confluent() const noexcept {
        return confluent_.load(std::memory_order_relaxed);
    }
    [[nodiscard]] uint64_t observation_count() const noexcept {
        return obs_count_.load(std::memory_order_relaxed);
    }

    [[nodiscard]] std::string to_stats_json() const;
    void reset();
    void update_config(const Config& cfg);

private:
    mutable std::mutex mutex_;
    Config cfg_;

    std::vector<int> buf_;  // +1, -1, 0
    int head_{0};
    int fill_{0};
    bool prev_confluent_{false};

    std::atomic<double>   score_{0.0};
    std::atomic<int>      dominant_{0};
    std::atomic<bool>     confluent_{false};
    std::atomic<uint64_t> obs_count_{0};
};

} // namespace llmquant
