#pragma once
/// SignalDrawdownMeter — running max-drawdown tracker for the cumulative bias stream.
/// Tracks the cumulative sum, running peak, current drawdown = (peak - current) / |peak|.
/// Fires on_new_drawdown_high when a new maximum drawdown is set, and on_recovery when
/// cumulative bias recovers back to its previous peak.
///
/// Feature flag: LLMQUANT_ENABLE_DRAWDOWN_METER / LLMQUANT_DRAWDOWN_METER_ENABLED

#ifdef LLMQUANT_DRAWDOWN_METER_ENABLED

#include <atomic>
#include <functional>
#include <mutex>
#include <string>

namespace llmquant {

class SignalDrawdownMeter {
public:
    struct Config {
        double drawdown_threshold = 0.1; ///< relative drawdown above this → on_new_drawdown_high

        /// Fires when max_drawdown is updated beyond the threshold.
        std::function<void(double drawdown)> on_new_drawdown_high;
        /// Fires when cumulative bias recovers back to (or above) its running peak.
        std::function<void(double cum_bias)> on_recovery;
    };

    explicit SignalDrawdownMeter(Config cfg = {});

    void record(double bias);

    double cum_bias()      const noexcept; ///< current cumulative bias
    double peak()          const noexcept; ///< running peak of cumulative bias
    double drawdown()      const noexcept; ///< current relative drawdown (peak - cur) / |peak|
    double max_drawdown()  const noexcept; ///< worst drawdown seen this session

    std::size_t drawdown_events()  const noexcept;
    std::size_t recovery_events()  const noexcept;
    std::size_t total_records()    const noexcept;

    std::string to_stats_json() const;
    void reset();
    void update_config(const Config& cfg);

private:
    mutable std::mutex mutex_;
    Config cfg_;

    double cum_bias_    = 0.0;
    double peak_        = 0.0;
    double max_drawdown_ = 0.0;
    bool   in_drawdown_ = false;

    std::atomic<double>      cum_{0.0};
    std::atomic<double>      peak_atomic_{0.0};
    std::atomic<double>      drawdown_{0.0};
    std::atomic<double>      max_dd_{0.0};
    std::atomic<std::size_t> drawdown_events_{0};
    std::atomic<std::size_t> recovery_events_{0};
    std::atomic<std::size_t> total_{0};
};

} // namespace llmquant

#endif // LLMQUANT_DRAWDOWN_METER_ENABLED
