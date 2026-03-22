#pragma once

/// @file NarrativeTemperatureGauge.h
/// @brief Combined intensity×volatility "narrative temperature" index.
///
/// ## Motivation
/// A sentiment stream can be high-intensity (large bias EMA) but stable,
/// or noisy but directionless.  Neither alone is the full picture.
/// "Narrative temperature" combines both: a "hot" narrative is one that
/// is simultaneously strong in direction AND volatile.
///
/// Temperature = |EMA(bias)| × σ(bias)
///
/// High temperature → strong, unstable sentiment → high execution risk.
/// Low temperature → weak or stable sentiment → safer conditions.
///
/// ## Algorithm
/// Maintains a rolling window of bias values.  On each `record(bias)`:
///   - Updates an EMA (alpha=0.15, configurable)
///   - Computes rolling σ (std-dev over last `window` values)
///   - Temperature = clamp(|ema| × σ / normalizer, 0, 1)
///
/// Fires `on_hot(temperature)` when temperature crosses `hot_threshold`.
/// Fires `on_cool(temperature)` with hysteresis when it drops back below.
///
/// ## Feature flag
/// `LLMQUANT_ENABLE_NARRATIVE_TEMPERATURE` (default ON).
///
/// Thread-safe.

#include <atomic>
#include <cstdint>
#include <functional>
#include <mutex>
#include <string>
#include <vector>

namespace llmquant {

class NarrativeTemperatureGauge {
public:
    struct Config {
        /// Rolling window for σ computation. Default 32.
        int window{32};

        /// EMA smoothing factor for bias direction. Default 0.15.
        double ema_alpha{0.15};

        /// Normalizer: σ × |ema| value that maps to temperature = 1.0.
        /// Set to expected max(|bias_ema| × σ). Default 0.5.
        double normalizer{0.5};

        /// Temperature above which on_hot fires. Default 0.7.
        double hot_threshold{0.7};

        /// Hysteresis: on_cool fires when temperature < hot_threshold × this. Default 0.8.
        double cool_hysteresis{0.8};

        /// Fired when temperature first crosses above hot_threshold.
        std::function<void(double temperature)> on_hot;

        /// Fired when temperature drops back below threshold after being hot.
        std::function<void(double temperature)> on_cool;
    };

    explicit NarrativeTemperatureGauge(Config cfg = Config{});

    // -----------------------------------------------------------------------
    // Data intake
    // -----------------------------------------------------------------------

    /// Record a new bias observation.
    void record(double bias);

    // -----------------------------------------------------------------------
    // Accessors (lock-free)
    // -----------------------------------------------------------------------

    /// Current temperature ∈ [0, 1].
    [[nodiscard]] double temperature() const noexcept {
        return temperature_.load(std::memory_order_relaxed);
    }

    /// True when temperature > hot_threshold.
    [[nodiscard]] bool is_hot() const noexcept {
        return hot_.load(std::memory_order_relaxed);
    }

    /// Number of times temperature crossed above hot_threshold (rising edge).
    [[nodiscard]] uint64_t heat_events() const noexcept {
        return heat_events_.load(std::memory_order_relaxed);
    }

    /// EMA of the bias series.
    [[nodiscard]] double bias_ema() const noexcept {
        return bias_ema_.load(std::memory_order_relaxed);
    }

    /// Rolling standard deviation of bias.
    [[nodiscard]] double bias_sigma() const noexcept {
        return bias_sigma_.load(std::memory_order_relaxed);
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

    std::vector<double> buf_;
    int head_{0};
    int fill_{0};

    double ema_{0.0};
    bool   ema_seeded_{false};
    bool   prev_hot_{false};

    std::atomic<double>   temperature_{0.0};
    std::atomic<double>   bias_ema_{0.0};
    std::atomic<double>   bias_sigma_{0.0};
    std::atomic<bool>     hot_{false};
    std::atomic<uint64_t> heat_events_{0};
    std::atomic<uint64_t> total_{0};
};

} // namespace llmquant
