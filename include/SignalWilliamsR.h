#pragma once
/// SignalWilliamsR — Williams %R oscillator applied to the bias stream.
/// %R = 100 × (H_n - x_n) / (H_n - L_n) — inverse of stochastic %K.
/// %R near -100 → near recent trough (oversold); near 0 → near recent peak (overbought).
/// Fires on_overbought (%R > -20) and on_oversold (%R < -80).
///
/// Feature flag: LLMQUANT_ENABLE_WILLIAMS_R / LLMQUANT_WILLIAMS_R_ENABLED

#ifdef LLMQUANT_WILLIAMS_R_ENABLED

#include <atomic>
#include <functional>
#include <mutex>
#include <string>
#include <vector>

namespace llmquant {

class SignalWilliamsR {
public:
    struct Config {
        int    period              = 14;   ///< lookback window
        int    min_samples         = 5;    ///< minimum before emitting %R
        double overbought_threshold = -20.0; ///< %R above this → overbought
        double oversold_threshold   = -80.0; ///< %R below this → oversold

        std::function<void(double wr)> on_overbought; ///< fires on overbought entry
        std::function<void(double wr)> on_oversold;   ///< fires on oversold entry
    };

    explicit SignalWilliamsR(Config cfg = {});

    void record(double bias);

    double williams_r() const noexcept; ///< %R ∈ [-100, 0]
    bool   is_overbought() const noexcept;
    bool   is_oversold()   const noexcept;

    std::size_t overbought_events() const noexcept;
    std::size_t oversold_events()   const noexcept;
    std::size_t total_records()     const noexcept;

    std::string to_stats_json() const;
    void reset();
    void update_config(const Config& cfg);

private:
    mutable std::mutex mutex_;
    Config cfg_;

    std::vector<double> buf_;
    int    head_           = 0;
    int    fill_           = 0;
    bool   prev_overbought_ = false;
    bool   prev_oversold_   = false;

    std::atomic<double>      wr_{-50.0}; // neutral default
    std::atomic<bool>        is_overbought_{false};
    std::atomic<bool>        is_oversold_{false};
    std::atomic<std::size_t> ob_events_{0};
    std::atomic<std::size_t> os_events_{0};
    std::atomic<std::size_t> total_{0};
};

} // namespace llmquant

#endif // LLMQUANT_WILLIAMS_R_ENABLED
