#pragma once
#include <atomic>
#include <functional>
#include <mutex>
#include <string>

namespace llmquant {

/**
 * @brief Triple Exponential Smoothing (Holt-Winters additive) for bias signals.
 *
 * Decomposes a bias stream into level, trend, and smoothed forecast.
 * Level EMA tracks the current value; trend EMA tracks the rate of change.
 * Forecast = level + trend.  A callback fires when the absolute forecast
 * error exceeds a threshold, indicating prediction breakdown.
 */
class BiasExponentialSmoothing {
public:
    struct Config {
        double alpha{0.2};             ///< Level smoothing factor (0 < α ≤ 1)
        double beta{0.1};              ///< Trend smoothing factor  (0 < β ≤ 1)
        double error_threshold{0.5};   ///< Absolute forecast error that triggers callback
        int    min_samples{3};         ///< Warm-up samples before forecasting

        /// Fired when |actual − forecast| > error_threshold.
        std::function<void(double /*actual*/, double /*forecast*/)> on_large_error;
    };

    explicit BiasExponentialSmoothing(Config cfg = {});

    /// Feed a new bias sample.  Updates level, trend, and last forecast error.
    void record(double bias);

    double level()          const noexcept;  ///< Current smoothed level
    double trend()          const noexcept;  ///< Current trend estimate
    double forecast()       const noexcept;  ///< One-step-ahead forecast (level + trend)
    double last_error()     const noexcept;  ///< |actual − previous forecast|

    std::size_t error_events()  const noexcept;
    std::size_t total_records() const noexcept;

    std::string to_stats_json() const;
    void reset();
    void update_config(const Config& cfg);

private:
    mutable std::mutex mutex_;
    Config  cfg_;
    int     n_{0};
    double  level_raw_{0.0};
    double  trend_raw_{0.0};
    double  prev_forecast_{0.0};
    bool    forecasting_{false};

    std::atomic<double>   level_{0.0};
    std::atomic<double>   trend_{0.0};
    std::atomic<double>   forecast_{0.0};
    std::atomic<double>   last_error_{0.0};
    std::atomic<uint64_t> error_events_{0};
    std::atomic<uint64_t> total_{0};
};

} // namespace llmquant
