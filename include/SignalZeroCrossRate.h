#pragma once
/// SignalZeroCrossRate — rolling zero-crossing rate of the bias stream.
/// High ZCR → oscillating/choppy signal; low ZCR → sustained trend.
/// Fires on_high_zcr when rate exceeds high_threshold (rapid oscillation)
/// and on_low_zcr when rate drops below low_threshold (clear trending).
///
/// Feature flag: LLMQUANT_ENABLE_ZERO_CROSS_RATE / LLMQUANT_ZERO_CROSS_RATE_ENABLED

#ifdef LLMQUANT_ZERO_CROSS_RATE_ENABLED

#include <atomic>
#include <functional>
#include <mutex>
#include <string>
#include <vector>

namespace llmquant {

class SignalZeroCrossRate {
public:
    struct Config {
        int    window        = 20;   ///< rolling window for ZCR computation
        int    min_samples   = 5;    ///< minimum before emitting ZCR
        double high_threshold = 0.5; ///< ZCR above this → on_high_zcr
        double low_threshold  = 0.1; ///< ZCR below this → on_low_zcr

        std::function<void(double zcr)> on_high_zcr; ///< fires on high oscillation
        std::function<void(double zcr)> on_low_zcr;  ///< fires on sustained trend
    };

    explicit SignalZeroCrossRate(Config cfg = {});

    void record(double bias);

    double zcr() const noexcept; ///< crossings / (window - 1), ∈ [0, 1]

    std::size_t high_zcr_events() const noexcept;
    std::size_t low_zcr_events()  const noexcept;
    std::size_t total_records()   const noexcept;

    std::string to_stats_json() const;
    void reset();
    void update_config(const Config& cfg);

private:
    mutable std::mutex mutex_;
    Config cfg_;

    std::vector<double> buf_;
    int    head_    = 0;
    int    fill_    = 0;
    bool   prev_high_ = false;
    bool   prev_low_  = false;

    std::atomic<double>      zcr_{0.0};
    std::atomic<std::size_t> high_events_{0};
    std::atomic<std::size_t> low_events_{0};
    std::atomic<std::size_t> total_{0};
};

} // namespace llmquant

#endif // LLMQUANT_ZERO_CROSS_RATE_ENABLED
