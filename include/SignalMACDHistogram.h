#pragma once
/// SignalMACDHistogram — MACD histogram applied to the LLM bias stream.
/// MACD = fast_EMA − slow_EMA; Signal = EMA(MACD, signal_period).
/// Histogram = MACD − Signal. Positive → bullish momentum building.
/// Fires on_zero_cross when histogram changes sign,
/// and on_divergence when histogram exceeds divergence_threshold.
///
/// Feature flag: LLMQUANT_ENABLE_MACD_HISTOGRAM / LLMQUANT_MACD_HISTOGRAM_ENABLED

#ifdef LLMQUANT_MACD_HISTOGRAM_ENABLED

#include <atomic>
#include <functional>
#include <mutex>
#include <string>

namespace llmquant {

class SignalMACDHistogram {
public:
    struct Config {
        double alpha_fast   = 2.0 / (12 + 1); ///< fast EMA (≈12-period)
        double alpha_slow   = 2.0 / (26 + 1); ///< slow EMA (≈26-period)
        double alpha_signal = 2.0 / (9 + 1);  ///< signal EMA (≈9-period)
        int    min_samples  = 26;              ///< minimum before firing events
        double divergence_threshold = 0.01;    ///< |histogram| above this → on_divergence

        /// Fires when histogram crosses zero (momentum reversal).
        std::function<void(double hist)> on_zero_cross;
        /// Fires when |histogram| exceeds divergence_threshold.
        std::function<void(double hist)> on_divergence;
    };

    explicit SignalMACDHistogram(Config cfg = {});

    void record(double bias);

    double fast_ema()    const noexcept;
    double slow_ema()    const noexcept;
    double macd()        const noexcept; ///< fast_ema - slow_ema
    double signal_line() const noexcept; ///< EMA of macd
    double histogram()   const noexcept; ///< macd - signal_line

    std::size_t zero_cross_events() const noexcept;
    std::size_t divergence_events() const noexcept;
    std::size_t total_records()     const noexcept;

    std::string to_stats_json() const;
    void reset();
    void update_config(const Config& cfg);

private:
    mutable std::mutex mutex_;
    Config cfg_;

    double ema_fast_   = 0.0;
    double ema_slow_   = 0.0;
    double ema_signal_ = 0.0;
    bool   seeded_     = false;
    bool   prev_pos_   = false; // previous histogram > 0

    std::atomic<double>      fast_ema_{0.0};
    std::atomic<double>      slow_ema_{0.0};
    std::atomic<double>      macd_{0.0};
    std::atomic<double>      signal_{0.0};
    std::atomic<double>      hist_{0.0};
    std::atomic<std::size_t> zero_cross_{0};
    std::atomic<std::size_t> divergence_{0};
    std::atomic<std::size_t> total_{0};
};

} // namespace llmquant

#endif // LLMQUANT_MACD_HISTOGRAM_ENABLED
