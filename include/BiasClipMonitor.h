#pragma once

/// @file BiasClipMonitor.h
/// @brief Tracks the fraction of bias values that exceed a clipping threshold.
///
/// ## Motivation
/// Extreme bias values (|bias| > threshold) indicate that the LLM is producing
/// maximal-confidence output — useful for detecting saturation or adversarial
/// prompts.  The clip rate (fraction of recent values clipped) is a scalar
/// indicator of how often the signal is "pegged" to its limits.
///
/// Fires `on_clip_spike(rate)` when the clip rate first exceeds
/// `spike_threshold`.
///
/// ## Feature flag
/// `LLMQUANT_ENABLE_CLIP_MONITOR` (default ON).
///
/// Thread-safe.

#include <atomic>
#include <cstdint>
#include <functional>
#include <mutex>
#include <string>
#include <vector>

namespace llmquant {

class BiasClipMonitor {
public:
    struct Config {
        /// Absolute bias value above which a sample is "clipped". Default 0.7.
        double clip_threshold{0.7};

        /// Rolling window size. Default 50.
        int window{50};

        /// Clip rate fraction above which on_clip_spike fires. Default 0.3.
        double spike_threshold{0.3};

        /// Fired when clip rate first crosses spike_threshold (outside lock).
        std::function<void(double rate)> on_clip_spike;
    };

    explicit BiasClipMonitor(Config cfg = Config{});

    void record(double bias);

    [[nodiscard]] double clip_rate() const noexcept {
        return clip_rate_.load(std::memory_order_relaxed);
    }
    [[nodiscard]] uint64_t clip_count() const noexcept {
        return clip_count_.load(std::memory_order_relaxed);
    }
    [[nodiscard]] bool is_spiking() const noexcept {
        return spiking_.load(std::memory_order_relaxed);
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

    std::vector<int> win_;  // 1 if clipped, 0 otherwise
    int head_{0};
    int fill_{0};
    int window_clips_{0};
    bool prev_spiking_{false};

    std::atomic<double>   clip_rate_{0.0};
    std::atomic<uint64_t> clip_count_{0};
    std::atomic<bool>     spiking_{false};
    std::atomic<uint64_t> obs_count_{0};
};

} // namespace llmquant
