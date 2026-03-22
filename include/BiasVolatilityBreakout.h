#pragma once
/// BiasVolatilityBreakout — volatility regime breakout detector.
/// Computes a short-window rolling std and a long-window rolling std.
/// When short_vol / long_vol exceeds breakout_ratio → vol expansion (on_expansion).
/// When the ratio drops below contraction_ratio → vol contraction (on_contraction).
///
/// Feature flag: LLMQUANT_ENABLE_VOL_BREAKOUT / LLMQUANT_VOL_BREAKOUT_ENABLED

#ifdef LLMQUANT_VOL_BREAKOUT_ENABLED

#include <atomic>
#include <functional>
#include <mutex>
#include <string>
#include <vector>

namespace llmquant {

class BiasVolatilityBreakout {
public:
    struct Config {
        int    short_window       = 10;  ///< short rolling window for vol
        int    long_window        = 30;  ///< long rolling window for vol
        int    min_samples        = 10;  ///< minimum before emitting signals
        double breakout_ratio     = 1.5; ///< short/long vol above this → expansion
        double contraction_ratio  = 0.7; ///< short/long vol below this → contraction

        std::function<void(double ratio)> on_expansion;   ///< fires on vol expansion regime
        std::function<void(double ratio)> on_contraction; ///< fires on vol contraction regime
    };

    explicit BiasVolatilityBreakout(Config cfg = {});

    void record(double bias);

    double short_vol()  const noexcept; ///< short-window rolling std
    double long_vol()   const noexcept; ///< long-window rolling std
    double vol_ratio()  const noexcept; ///< short_vol / long_vol

    std::size_t expansion_events()   const noexcept;
    std::size_t contraction_events() const noexcept;
    std::size_t total_records()      const noexcept;

    std::string to_stats_json() const;
    void reset();
    void update_config(const Config& cfg);

private:
    mutable std::mutex mutex_;
    Config cfg_;

    std::vector<double> short_buf_;
    std::vector<double> long_buf_;
    int    s_head_ = 0, s_fill_ = 0;
    int    l_head_ = 0, l_fill_ = 0;
    bool   prev_expansion_   = false;
    bool   prev_contraction_ = false;

    std::atomic<double>      short_vol_{0.0};
    std::atomic<double>      long_vol_{0.0};
    std::atomic<double>      vol_ratio_{1.0};
    std::atomic<std::size_t> expansion_events_{0};
    std::atomic<std::size_t> contraction_events_{0};
    std::atomic<std::size_t> total_{0};

    static double rolling_std(const std::vector<double>& buf, int head, int fill);
};

} // namespace llmquant

#endif // LLMQUANT_VOL_BREAKOUT_ENABLED
