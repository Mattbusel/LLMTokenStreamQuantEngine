#pragma once
/// BiasSkewnessTracker — rolling skewness of the bias sample distribution.
/// Skewness > 0 → distribution tail pulled positive (bullish asymmetry).
/// Skewness < 0 → tail pulled negative (bearish asymmetry).
/// Fires on_positive_skew / on_negative_skew on sign changes.
///
/// Feature flag: LLMQUANT_ENABLE_SKEWNESS_TRACKER / LLMQUANT_SKEWNESS_TRACKER_ENABLED

#ifdef LLMQUANT_SKEWNESS_TRACKER_ENABLED

#include <atomic>
#include <functional>
#include <mutex>
#include <string>
#include <vector>

namespace llmquant {

class BiasSkewnessTracker {
public:
    struct Config {
        int    window      = 30;   ///< rolling window for skewness computation
        int    min_samples = 10;   ///< minimum before emitting skewness

        /// Fires when skewness flips from negative to positive.
        std::function<void(double skew)> on_positive_skew;
        /// Fires when skewness flips from positive to negative.
        std::function<void(double skew)> on_negative_skew;
    };

    explicit BiasSkewnessTracker(Config cfg = {});

    void record(double bias);

    double skewness() const noexcept;
    double mean()     const noexcept;
    double variance() const noexcept;

    std::size_t positive_skew_events() const noexcept;
    std::size_t negative_skew_events() const noexcept;
    std::size_t total_records()        const noexcept;

    std::string to_stats_json() const;
    void reset();
    void update_config(const Config& cfg);

private:
    mutable std::mutex mutex_;
    Config cfg_;

    std::vector<double> buf_;
    int    head_   = 0;
    int    fill_   = 0;
    bool   prev_positive_ = false;

    std::atomic<double>      skewness_{0.0};
    std::atomic<std::size_t> pos_events_{0};
    std::atomic<std::size_t> neg_events_{0};
    std::atomic<std::size_t> total_{0};
};

} // namespace llmquant

#endif // LLMQUANT_SKEWNESS_TRACKER_ENABLED
