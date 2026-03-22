#pragma once

/// @file SentimentZScoreTracker.h
/// @brief Rolling z-score normalization of token bias values.
///
/// ## Motivation
/// Raw bias magnitudes vary with model, prompt, and market session.
/// Z-scoring normalizes the bias stream against its own recent history,
/// making downstream thresholds session-invariant:
///   z = (bias - rolling_mean) / rolling_sigma
///
/// High |z| → extreme reading relative to recent history.
/// Fires `on_extreme(z)` when |z| > extreme_threshold.
///
/// ## Feature flag
/// `LLMQUANT_ENABLE_ZSCORE_TRACKER` (default ON).
///
/// Thread-safe.

#include <atomic>
#include <cstdint>
#include <functional>
#include <mutex>
#include <string>
#include <vector>

namespace llmquant {

class SentimentZScoreTracker {
public:
    struct Config {
        /// Rolling window for mean/sigma computation. Default 50.
        int window{50};

        /// |z| above which on_extreme fires. Default 2.0.
        double extreme_threshold{2.0};

        /// Fired when |z| first crosses above extreme_threshold (outside lock).
        std::function<void(double z)> on_extreme;
    };

    explicit SentimentZScoreTracker(Config cfg = Config{});

    void record(double bias);

    [[nodiscard]] double z_score() const noexcept {
        return z_score_.load(std::memory_order_relaxed);
    }
    [[nodiscard]] double rolling_mean() const noexcept {
        return mean_.load(std::memory_order_relaxed);
    }
    [[nodiscard]] double rolling_sigma() const noexcept {
        return sigma_.load(std::memory_order_relaxed);
    }
    [[nodiscard]] bool is_extreme() const noexcept {
        return extreme_.load(std::memory_order_relaxed);
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

    std::vector<double> buf_;
    int head_{0};
    int fill_{0};
    bool prev_extreme_{false};

    std::atomic<double>   z_score_{0.0};
    std::atomic<double>   mean_{0.0};
    std::atomic<double>   sigma_{0.0};
    std::atomic<bool>     extreme_{false};
    std::atomic<uint64_t> obs_count_{0};
};

} // namespace llmquant
