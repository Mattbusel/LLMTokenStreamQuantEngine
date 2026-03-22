#pragma once
#include <atomic>
#include <functional>
#include <mutex>
#include <string>
#include <vector>

namespace llmquant {

/**
 * @brief Median-based bias regime filter.
 *
 * Computes the rolling median of bias values.  When the current bias
 * deviates from the median by more than filter_threshold, the move is
 * classified as a regime signal; otherwise, noise.  Fires callbacks on
 * regime signal events.
 */
class BiasRegimeFilter {
public:
    struct Config {
        int    window{20};               ///< Rolling window for median
        int    min_samples{5};           ///< Minimum samples before filtering
        double filter_threshold{0.01};   ///< |bias - median| above this = signal

        /// Fired when bias is significantly above median (bullish signal).
        std::function<void(double /*bias*/, double /*median*/)> on_bullish_signal;
        /// Fired when bias is significantly below median (bearish signal).
        std::function<void(double /*bias*/, double /*median*/)> on_bearish_signal;
    };

    explicit BiasRegimeFilter(Config cfg = {});

    /// Feed a new bias sample.
    void record(double bias);

    double rolling_median()   const noexcept;
    double current_deviation() const noexcept;  ///< bias - median
    bool   is_bullish_signal() const noexcept;
    bool   is_bearish_signal() const noexcept;

    std::size_t bullish_signals() const noexcept;
    std::size_t bearish_signals() const noexcept;
    std::size_t total_records()   const noexcept;

    std::string to_stats_json() const;
    void reset();
    void update_config(const Config& cfg);

private:
    mutable std::mutex  mutex_;
    Config              cfg_;
    std::vector<double> buf_;
    int head_{0}, fill_{0};
    bool prev_bull_{false}, prev_bear_{false};

    std::atomic<double>   median_{0.0};
    std::atomic<double>   deviation_{0.0};
    std::atomic<bool>     bull_{false};
    std::atomic<bool>     bear_{false};
    std::atomic<uint64_t> bull_ev_{0};
    std::atomic<uint64_t> bear_ev_{0};
    std::atomic<uint64_t> total_{0};
};

} // namespace llmquant
