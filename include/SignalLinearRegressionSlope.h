#pragma once
#include <atomic>
#include <functional>
#include <mutex>
#include <string>
#include <vector>

namespace llmquant {

/**
 * @brief Rolling OLS linear regression slope of bias stream.
 *
 * Fits y = a + b*t to the most recent `window` samples via the closed-form
 * OLS formula.  The slope b is the primary output — positive = uptrend,
 * negative = downtrend.  Fires callbacks when slope crosses signed thresholds.
 */
class SignalLinearRegressionSlope {
public:
    struct Config {
        int    window{20};              ///< Rolling window for OLS fit
        int    min_samples{5};          ///< Minimum samples before slope is valid
        double pos_slope_threshold{0.0}; ///< Slope above this → positive trend callback
        double neg_slope_threshold{0.0}; ///< Slope below this → negative trend callback

        /// Fired when slope rises above pos_slope_threshold.
        std::function<void(double /*slope*/)> on_positive_trend;
        /// Fired when slope falls below neg_slope_threshold.
        std::function<void(double /*slope*/)> on_negative_trend;
    };

    explicit SignalLinearRegressionSlope(Config cfg = {});

    /// Feed a new bias sample.
    void record(double bias);

    double slope()     const noexcept;
    double intercept() const noexcept;
    double r_squared() const noexcept;

    std::size_t positive_trend_events() const noexcept;
    std::size_t negative_trend_events() const noexcept;
    std::size_t total_records()         const noexcept;

    std::string to_stats_json() const;
    void reset();
    void update_config(const Config& cfg);

private:
    mutable std::mutex  mutex_;
    Config              cfg_;
    std::vector<double> buf_;
    int head_{0}, fill_{0};
    bool prev_pos_{false}, prev_neg_{false};

    std::atomic<double>   slope_{0.0};
    std::atomic<double>   intercept_{0.0};
    std::atomic<double>   r2_{0.0};
    std::atomic<uint64_t> pos_ev_{0};
    std::atomic<uint64_t> neg_ev_{0};
    std::atomic<uint64_t> total_{0};
};

} // namespace llmquant
