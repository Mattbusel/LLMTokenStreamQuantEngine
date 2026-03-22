#pragma once

/// @file SignalGainLossRatio.h
/// @brief Rolling gain/loss ratio for LLM bias signal quality assessment.
///
/// ## Motivation
/// The gain/loss ratio G/L = Σ(positive Δbias) / Σ(|negative Δbias|) over a
/// rolling window directly estimates the raw edge of the signal stream.
/// A G/L > 1 means the signal is net bullish; G/L < 1 means net bearish.
/// A G/L far from 1 in either direction indicates directional edge above
/// the noise floor — suitable input for Kelly fraction computation.
///
/// ## Algorithm
/// Maintains ring buffers of positive and negative delta sums with O(1)
/// incremental update.
/// ratio = gain_sum / loss_sum  (loss_sum > 0),  clamped to [0, max_ratio].
/// Fires on_high_gain / on_high_loss when ratio crosses thresholds.
///
/// ## Feature flag
/// `LLMQUANT_ENABLE_GAIN_LOSS_RATIO` (default ON).
///
/// Thread-safe.

#include <atomic>
#include <cstdint>
#include <functional>
#include <mutex>
#include <string>
#include <vector>

namespace llmquant {

class SignalGainLossRatio {
public:
    struct Config {
        /// Rolling window for gain/loss accumulation. Default 32.
        int window{32};

        /// Minimum samples before reporting. Default 4.
        int min_samples{4};

        /// Ratio above this fires on_high_gain (net bullish edge). Default 1.5.
        double high_gain_threshold{1.5};

        /// Ratio below this fires on_high_loss (net bearish edge). Default 0.67.
        double high_loss_threshold{0.67};

        /// Clamp ratio output to [0, max_ratio]. Default 10.0.
        double max_ratio{10.0};

        /// Fired when ratio crosses above high_gain_threshold.
        std::function<void(double ratio)> on_high_gain;

        /// Fired when ratio crosses below high_loss_threshold.
        std::function<void(double ratio)> on_high_loss;
    };

    explicit SignalGainLossRatio(Config cfg = Config{});

    // -----------------------------------------------------------------------
    // Data intake
    // -----------------------------------------------------------------------

    void record(double bias);

    // -----------------------------------------------------------------------
    // Accessors (lock-free)
    // -----------------------------------------------------------------------

    /// G/L ratio ∈ [0, max_ratio]. 1.0 = balanced. >1 = net gain. <1 = net loss.
    [[nodiscard]] double ratio() const noexcept {
        return ratio_.load(std::memory_order_relaxed);
    }

    /// Sum of positive bias changes in the current window.
    [[nodiscard]] double gain_sum() const noexcept {
        return gain_sum_.load(std::memory_order_relaxed);
    }

    /// Sum of absolute negative bias changes in the current window.
    [[nodiscard]] double loss_sum() const noexcept {
        return loss_sum_.load(std::memory_order_relaxed);
    }

    [[nodiscard]] uint64_t high_gain_events() const noexcept {
        return high_gain_events_.load(std::memory_order_relaxed);
    }

    [[nodiscard]] uint64_t high_loss_events() const noexcept {
        return high_loss_events_.load(std::memory_order_relaxed);
    }

    [[nodiscard]] uint64_t total_records() const noexcept {
        return total_.load(std::memory_order_relaxed);
    }

    [[nodiscard]] std::string to_stats_json() const;

    void reset();
    void update_config(const Config& cfg);

private:
    mutable std::mutex mutex_;
    Config cfg_;

    // Each ring slot stores a signed Δbias; we derive gains/losses on eviction.
    std::vector<double> delta_ring_;
    int head_{0};
    int fill_{0};

    double gain_sum_v_{0.0};
    double loss_sum_v_{0.0};
    double prev_bias_{0.0};
    bool   seeded_{false};

    bool prev_high_gain_{false};
    bool prev_high_loss_{false};

    std::atomic<double>   ratio_{1.0};
    std::atomic<double>   gain_sum_{0.0};
    std::atomic<double>   loss_sum_{0.0};
    std::atomic<uint64_t> high_gain_events_{0};
    std::atomic<uint64_t> high_loss_events_{0};
    std::atomic<uint64_t> total_{0};
};

} // namespace llmquant
