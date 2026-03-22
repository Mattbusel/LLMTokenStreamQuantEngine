#pragma once
/// NarrativeFatigueDetector — exponential-decay saturation sensor for bias streams.
/// Tracks the ratio of recent bias to its long-run EMA. When the ratio approaches 1.0
/// (new signals not moving the EMA) the narrative is "fatigued". Fires on_fatigue
/// when the ratio drops below fatigue_threshold and on_recovery when it recovers.
///
/// Feature flag: LLMQUANT_ENABLE_FATIGUE_DETECTOR / LLMQUANT_FATIGUE_DETECTOR_ENABLED

#ifdef LLMQUANT_FATIGUE_DETECTOR_ENABLED

#include <atomic>
#include <functional>
#include <mutex>
#include <string>

namespace llmquant {

class NarrativeFatigueDetector {
public:
    struct Config {
        double alpha_fast        = 0.3;   ///< fast EMA smoothing
        double alpha_slow        = 0.05;  ///< slow EMA smoothing
        double fatigue_threshold = 0.1;   ///< ratio delta below this → fatigued
        int    min_samples       = 10;    ///< minimum records before firing

        /// Fires on transition into fatigued state (ratio_delta < fatigue_threshold).
        std::function<void(double ratio_delta)> on_fatigue;
        /// Fires on recovery from fatigued state.
        std::function<void(double ratio_delta)> on_recovery;
    };

    explicit NarrativeFatigueDetector(Config cfg = {});

    void record(double bias);

    double fast_ema()    const noexcept;
    double slow_ema()    const noexcept;
    double ratio_delta() const noexcept; ///< |fast - slow| / (|slow| + eps)
    bool   is_fatigued() const noexcept;

    std::size_t fatigue_events()  const noexcept;
    std::size_t recovery_events() const noexcept;
    std::size_t total_records()   const noexcept;

    std::string to_stats_json() const;
    void reset();
    void update_config(const Config& cfg);

private:
    mutable std::mutex mutex_;
    Config cfg_;

    double ema_fast_ = 0.0;
    double ema_slow_ = 0.0;
    bool   was_fatigued_ = false;

    std::atomic<double>      ratio_delta_{0.0};
    std::atomic<bool>        is_fatigued_{false};
    std::atomic<std::size_t> fatigue_events_{0};
    std::atomic<std::size_t> recovery_events_{0};
    std::atomic<std::size_t> total_{0};
};

} // namespace llmquant

#endif // LLMQUANT_FATIGUE_DETECTOR_ENABLED
