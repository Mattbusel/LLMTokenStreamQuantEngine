#pragma once
/// TokenBiasCorrelogram — sliding autocorrelation at multiple lags for the bias stream.
/// Stores ACF values at lags 1..max_lag. Detects cyclical bias patterns by finding
/// the dominant positive-peak lag. Fires on_cycle_detected when peak ACF exceeds
/// cycle_threshold.
///
/// Feature flag: LLMQUANT_ENABLE_BIAS_CORRELOGRAM / LLMQUANT_BIAS_CORRELOGRAM_ENABLED

#ifdef LLMQUANT_BIAS_CORRELOGRAM_ENABLED

#include <atomic>
#include <functional>
#include <mutex>
#include <string>
#include <vector>

namespace llmquant {

class TokenBiasCorrelogram {
public:
    struct Config {
        int    window          = 64;   ///< rolling window for ACF computation
        int    max_lag         = 16;   ///< maximum autocorrelation lag to compute
        int    min_samples     = 20;   ///< minimum before computing ACF
        double cycle_threshold = 0.4;  ///< peak ACF above this fires on_cycle_detected

        /// Fires when a dominant cyclical lag is detected.
        /// Arguments: dominant_lag, peak_acf_value
        std::function<void(int lag, double acf)> on_cycle_detected;
    };

    explicit TokenBiasCorrelogram(Config cfg = {});

    void record(double bias);

    /// ACF value at a specific lag (1-indexed). Returns 0 if not yet computed.
    double acf_at(int lag) const;
    /// Dominant lag with highest positive ACF.
    int    dominant_lag() const noexcept;
    double peak_acf()     const noexcept;

    std::size_t cycle_events()  const noexcept;
    std::size_t total_records() const noexcept;

    std::string to_stats_json() const;
    void reset();
    void update_config(const Config& cfg);

private:
    mutable std::mutex mutex_;
    Config cfg_;

    std::vector<double> buf_;
    int    head_  = 0;
    int    fill_  = 0;

    std::vector<double> acf_values_; // acf_values_[i] = ACF at lag i+1

    std::atomic<int>         dominant_lag_{0};
    std::atomic<double>      peak_acf_{0.0};
    std::atomic<std::size_t> cycle_events_{0};
    std::atomic<std::size_t> total_{0};
};

} // namespace llmquant

#endif // LLMQUANT_BIAS_CORRELOGRAM_ENABLED
