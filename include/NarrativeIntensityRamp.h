#pragma once

/// @file NarrativeIntensityRamp.h
/// @brief Tracks the rate of change (ramp) of narrative intensity.
///
/// ## Motivation
/// Narrative intensity is the smoothed magnitude of bias: |EMA(bias)|.
/// The ramp is its first derivative — how fast intensity is building up or
/// fading.  A sharply positive ramp signals an emerging narrative.
/// A negative ramp signals narrative exhaustion.
///
/// ## Algorithm
/// Intensity = alpha * |bias| + (1-alpha) * intensity  (EMA of |bias|)
/// Ramp = intensity - prev_intensity  (first finite difference)
/// ramp_score = clamp(ramp / saturation_ramp, -1, 1)
///
/// Fires `on_surge(ramp)` when ramp first crosses surge_threshold.
/// Fires `on_fade(ramp)` when ramp drops below -surge_threshold.
///
/// ## Feature flag
/// `LLMQUANT_ENABLE_INTENSITY_RAMP` (default ON).
///
/// Thread-safe.

#include <atomic>
#include <cstdint>
#include <functional>
#include <mutex>
#include <string>

namespace llmquant {

class NarrativeIntensityRamp {
public:
    struct Config {
        /// EMA alpha for smoothing |bias| into intensity. Default 0.15.
        double ema_alpha{0.15};

        /// Ramp above which on_surge fires. Default 0.02.
        double surge_threshold{0.02};

        /// Ramp value that maps to ramp_score = ±1. Default 0.05.
        double saturation_ramp{0.05};

        /// Fired when ramp first crosses +surge_threshold (outside lock).
        std::function<void(double ramp)> on_surge;

        /// Fired when ramp first drops below -surge_threshold (outside lock).
        std::function<void(double ramp)> on_fade;
    };

    explicit NarrativeIntensityRamp(Config cfg = Config{});

    void record(double bias);

    [[nodiscard]] double intensity() const noexcept {
        return intensity_.load(std::memory_order_relaxed);
    }
    [[nodiscard]] double ramp() const noexcept {
        return ramp_.load(std::memory_order_relaxed);
    }
    [[nodiscard]] double ramp_score() const noexcept {
        return ramp_score_.load(std::memory_order_relaxed);
    }
    [[nodiscard]] bool is_surging() const noexcept {
        return surging_.load(std::memory_order_relaxed);
    }
    [[nodiscard]] bool is_fading() const noexcept {
        return fading_.load(std::memory_order_relaxed);
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

    double ema_{0.0};
    double prev_ema_{0.0};
    bool   seeded_{false};
    bool   prev_surging_{false};
    bool   prev_fading_{false};

    std::atomic<double>   intensity_{0.0};
    std::atomic<double>   ramp_{0.0};
    std::atomic<double>   ramp_score_{0.0};
    std::atomic<bool>     surging_{false};
    std::atomic<bool>     fading_{false};
    std::atomic<uint64_t> obs_count_{0};
};

} // namespace llmquant
