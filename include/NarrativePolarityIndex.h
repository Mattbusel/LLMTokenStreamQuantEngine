#pragma once

/// @file NarrativePolarityIndex.h
/// @brief Tracks a composite polarity index for the LLM bias stream, separating
///        cumulative bullish and bearish pressure into a normalised score.
///
/// ## Motivation
/// Net bias (Σ positive − Σ negative) discards magnitude information.  The
/// Narrative Polarity Index (NPI) computes:
///
///   NPI = (bull_ema − bear_ema) / (bull_ema + bear_ema + ε)
///
/// where bull_ema = EMA of positive bias contributions and
///       bear_ema = EMA of |negative| bias contributions.
///
/// NPI ∈ (−1, +1):  +1 = purely bullish; −1 = purely bearish; 0 = balanced.
///
/// This is inspired by commodity market breadth indicators (Advance/Decline
/// line analogue) applied to LLM token sentiment streams.
///
/// Fires `on_polarity_flip(old_npi, new_npi)` when the sign of NPI flips.
/// Fires `on_extreme(npi)` when |NPI| > `extreme_threshold`.
///
/// ## Feature flag
/// `LLMQUANT_ENABLE_POLARITY_INDEX` (default ON).
///
/// Thread-safe.

#include <atomic>
#include <cstdint>
#include <functional>
#include <mutex>
#include <string>

namespace llmquant {

class NarrativePolarityIndex {
public:
    struct Config {
        /// EMA decay factor for bull and bear accumulators. Default 0.05 (20-period EMA).
        double alpha{0.05};

        /// |NPI| above this fires on_extreme. Default 0.7.
        double extreme_threshold{0.7};

        /// Minimum samples before emitting events. Default 5.
        int min_samples{5};

        /// Fired when NPI changes sign (outside lock). Params: old NPI, new NPI.
        std::function<void(double old_npi, double new_npi)> on_polarity_flip;

        /// Fired when |NPI| > extreme_threshold (outside lock).
        std::function<void(double npi)> on_extreme;
    };

    explicit NarrativePolarityIndex(Config cfg = Config{});

    // -----------------------------------------------------------------------
    // Data intake
    // -----------------------------------------------------------------------

    void record(double bias);

    // -----------------------------------------------------------------------
    // Accessors (lock-free)
    // -----------------------------------------------------------------------

    /// Narrative Polarity Index NPI ∈ (−1, +1).
    [[nodiscard]] double npi() const noexcept {
        return npi_.load(std::memory_order_relaxed);
    }

    /// Bullish EMA accumulator.
    [[nodiscard]] double bull_ema() const noexcept {
        return bull_.load(std::memory_order_relaxed);
    }

    /// Bearish EMA accumulator.
    [[nodiscard]] double bear_ema() const noexcept {
        return bear_.load(std::memory_order_relaxed);
    }

    /// Total polarity-flip events fired.
    [[nodiscard]] uint64_t flip_events() const noexcept {
        return flips_.load(std::memory_order_relaxed);
    }

    /// Total extreme events fired.
    [[nodiscard]] uint64_t extreme_events() const noexcept {
        return extremes_.load(std::memory_order_relaxed);
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

    double bull_ema_{0.0};
    double bear_ema_{0.0};
    double prev_npi_{0.0};
    bool   seeded_{false};

    std::atomic<double>   npi_{0.0};
    std::atomic<double>   bull_{0.0};
    std::atomic<double>   bear_{0.0};
    std::atomic<uint64_t> flips_{0};
    std::atomic<uint64_t> extremes_{0};
    std::atomic<uint64_t> total_{0};
};

} // namespace llmquant
