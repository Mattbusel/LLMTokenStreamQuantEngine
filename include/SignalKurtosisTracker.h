#pragma once
/// SignalKurtosisTracker — rolling excess kurtosis of the bias sample distribution.
/// Excess kurtosis > 0 → leptokurtic (fat tails, outlier-prone).
/// Excess kurtosis < 0 → platykurtic (thin tails, moderate swings).
/// Fires on_fat_tail when kurtosis exceeds fat_tail_threshold.
///
/// Feature flag: LLMQUANT_ENABLE_KURTOSIS_TRACKER / LLMQUANT_KURTOSIS_TRACKER_ENABLED

#ifdef LLMQUANT_KURTOSIS_TRACKER_ENABLED

#include <atomic>
#include <functional>
#include <mutex>
#include <string>
#include <vector>

namespace llmquant {

class SignalKurtosisTracker {
public:
    struct Config {
        int    window            = 30;  ///< rolling window for kurtosis computation
        int    min_samples       = 10;  ///< minimum before emitting kurtosis
        double fat_tail_threshold = 1.0; ///< excess kurtosis above this → on_fat_tail

        /// Fires when excess kurtosis first exceeds fat_tail_threshold.
        std::function<void(double kurtosis)> on_fat_tail;
        /// Fires when excess kurtosis drops back below fat_tail_threshold.
        std::function<void(double kurtosis)> on_normal_tail;
    };

    explicit SignalKurtosisTracker(Config cfg = {});

    void record(double bias);

    double kurtosis() const noexcept; ///< excess kurtosis (kurtosis - 3)
    double mean()     const noexcept;

    std::size_t fat_tail_events()    const noexcept;
    std::size_t normal_tail_events() const noexcept;
    std::size_t total_records()      const noexcept;

    std::string to_stats_json() const;
    void reset();
    void update_config(const Config& cfg);

private:
    mutable std::mutex mutex_;
    Config cfg_;

    std::vector<double> buf_;
    int    head_         = 0;
    int    fill_         = 0;
    bool   prev_fat_tail_ = false;

    std::atomic<double>      kurtosis_{0.0};
    std::atomic<std::size_t> fat_events_{0};
    std::atomic<std::size_t> normal_events_{0};
    std::atomic<std::size_t> total_{0};
};

} // namespace llmquant

#endif // LLMQUANT_KURTOSIS_TRACKER_ENABLED
