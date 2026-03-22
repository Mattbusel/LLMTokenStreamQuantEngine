#pragma once

/// @file BiasImpulseDetector.h
/// @brief Detects impulsive (sudden, large) bias shifts that exceed a dynamic
///        threshold set by the local rolling standard deviation.
///
/// ## Motivation
/// An "impulse" in a time series is a sample that deviates from the recent
/// mean by more than `z_threshold` standard deviations.  For LLM bias streams
/// this captures:
///   - Breaking news events causing abrupt sentiment jumps.
///   - Model hallucinations producing unrealistic bias spikes.
///   - Genuine structural breaks vs ongoing trends.
///
/// Unlike simple thresholding, the dynamic z-score threshold adapts to local
/// volatility, preventing over-triggering in high-vol regimes and under-
/// triggering in quiet regimes.
///
/// ## Algorithm
/// - Rolling window mean μ and σ via Welford online algorithm.
/// - z_t = (x_t - μ) / (σ + ε).
/// - |z_t| > z_threshold → impulse event.
/// - Fires `on_impulse(z_score, bias)` per impulse.
/// - Tracks impulse count and running maximum |z|.
///
/// ## Feature flag
/// `LLMQUANT_ENABLE_IMPULSE_DETECTOR` (default ON).
///
/// Thread-safe.

#include <atomic>
#include <cstdint>
#include <functional>
#include <mutex>
#include <string>
#include <vector>

namespace llmquant {

class BiasImpulseDetector {
public:
    struct Config {
        /// Rolling window for mean/std computation. Default 30.
        int window{30};

        /// Minimum fill before firing events. Default 8.
        int min_samples{8};

        /// |z| threshold for impulse detection. Default 2.5.
        double z_threshold{2.5};

        /// Fired when |z_t| > z_threshold (outside lock).
        std::function<void(double z_score, double bias)> on_impulse;
    };

    explicit BiasImpulseDetector(Config cfg = Config{});

    // -----------------------------------------------------------------------
    // Data intake
    // -----------------------------------------------------------------------

    void record(double bias);

    // -----------------------------------------------------------------------
    // Accessors (lock-free)
    // -----------------------------------------------------------------------

    /// Most recent z-score.
    [[nodiscard]] double last_z() const noexcept {
        return last_z_.load(std::memory_order_relaxed);
    }

    /// Rolling mean.
    [[nodiscard]] double rolling_mean() const noexcept {
        return mean_.load(std::memory_order_relaxed);
    }

    /// Rolling standard deviation.
    [[nodiscard]] double rolling_std() const noexcept {
        return std_.load(std::memory_order_relaxed);
    }

    /// Maximum |z| seen.
    [[nodiscard]] double max_z() const noexcept {
        return max_z_.load(std::memory_order_relaxed);
    }

    /// Total impulse events fired.
    [[nodiscard]] uint64_t impulse_events() const noexcept {
        return impulse_events_.load(std::memory_order_relaxed);
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

    double max_z_val_{0.0};

    std::atomic<double>   last_z_{0.0};
    std::atomic<double>   mean_{0.0};
    std::atomic<double>   std_{0.0};
    std::atomic<double>   max_z_{0.0};
    std::atomic<uint64_t> impulse_events_{0};
    std::atomic<uint64_t> total_{0};
};

} // namespace llmquant
