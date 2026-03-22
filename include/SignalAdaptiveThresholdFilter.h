#pragma once
/// SignalAdaptiveThresholdFilter — dynamic signal filter based on rolling ATR.
/// Threshold = multiplier × ATR (EMA of |Δbias|). Bias exceeding the threshold
/// is considered a "significant" move; smaller moves are filtered as noise.
/// Fires on_above / on_below when bias breaks out of the adaptive band.
///
/// Feature flag: LLMQUANT_ENABLE_ADAPTIVE_FILTER / LLMQUANT_ADAPTIVE_FILTER_ENABLED

#ifdef LLMQUANT_ADAPTIVE_FILTER_ENABLED

#include <atomic>
#include <functional>
#include <mutex>
#include <string>

namespace llmquant {

class SignalAdaptiveThresholdFilter {
public:
    struct Config {
        double alpha_atr  = 0.1;  ///< EMA decay for ATR (|Δbias|)
        double multiplier = 2.0;  ///< threshold = multiplier × ATR
        int    min_samples = 5;   ///< warm-up before firing events

        /// Fires when bias > +threshold (positive breakout).
        std::function<void(double bias, double threshold)> on_above;
        /// Fires when bias < -threshold (negative breakout).
        std::function<void(double bias, double threshold)> on_below;
    };

    explicit SignalAdaptiveThresholdFilter(Config cfg = {});

    void record(double bias);

    double atr()       const noexcept; ///< current ATR value
    double threshold() const noexcept; ///< = multiplier × atr
    bool   is_above()  const noexcept; ///< bias > +threshold
    bool   is_below()  const noexcept; ///< bias < -threshold

    std::size_t above_events() const noexcept;
    std::size_t below_events() const noexcept;
    std::size_t total_records() const noexcept;

    std::string to_stats_json() const;
    void reset();
    void update_config(const Config& cfg);

private:
    mutable std::mutex mutex_;
    Config cfg_;

    bool   seeded_     = false;
    double atr_raw_    = 0.0;
    double prev_bias_  = 0.0;
    bool   prev_above_ = false;
    bool   prev_below_ = false;

    std::atomic<double>      atr_{0.0};
    std::atomic<double>      thresh_{0.0};
    std::atomic<bool>        above_{false};
    std::atomic<bool>        below_{false};
    std::atomic<std::size_t> above_ev_{0};
    std::atomic<std::size_t> below_ev_{0};
    std::atomic<std::size_t> total_{0};
};

} // namespace llmquant

#endif // LLMQUANT_ADAPTIVE_FILTER_ENABLED
