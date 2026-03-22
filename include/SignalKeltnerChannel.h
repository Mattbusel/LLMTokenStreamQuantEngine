#pragma once
#include <atomic>
#include <functional>
#include <mutex>
#include <string>

namespace llmquant {

/**
 * @brief Keltner Channel for bias streams: EMA ± multiplier × ATR.
 *
 * Upper band = EMA + k × ATR; lower band = EMA - k × ATR.
 * ATR is approximated as EMA of |bias - prev_bias|.
 * Fires callbacks when bias breaks above upper or below lower band.
 */
class SignalKeltnerChannel {
public:
    struct Config {
        double alpha_ema{0.1};   ///< EMA smoothing factor
        double alpha_atr{0.1};   ///< ATR EMA smoothing factor
        double multiplier{2.0};  ///< Band width multiplier (k)
        int    min_samples{5};   ///< Warm-up samples

        /// Fired when bias breaks above upper Keltner band.
        std::function<void(double /*bias*/, double /*upper*/)> on_upper_break;
        /// Fired when bias breaks below lower Keltner band.
        std::function<void(double /*bias*/, double /*lower*/)> on_lower_break;
    };

    explicit SignalKeltnerChannel(Config cfg = {});

    /// Feed a new bias sample.
    void record(double bias);

    double ema()         const noexcept;
    double atr()         const noexcept;
    double upper_band()  const noexcept;
    double lower_band()  const noexcept;
    bool   above_upper() const noexcept;
    bool   below_lower() const noexcept;

    std::size_t upper_breaks() const noexcept;
    std::size_t lower_breaks() const noexcept;
    std::size_t total_records() const noexcept;

    std::string to_stats_json() const;
    void reset();
    void update_config(const Config& cfg);

private:
    mutable std::mutex mutex_;
    Config cfg_;
    bool seeded_{false};
    double ema_raw_{0.0};
    double atr_raw_{0.0};
    double prev_bias_{0.0};
    bool prev_above_{false};
    bool prev_below_{false};

    std::atomic<double>   ema_{0.0};
    std::atomic<double>   atr_{0.0};
    std::atomic<double>   upper_{0.0};
    std::atomic<double>   lower_{0.0};
    std::atomic<bool>     above_{false};
    std::atomic<bool>     below_{false};
    std::atomic<uint64_t> upper_ev_{0};
    std::atomic<uint64_t> lower_ev_{0};
    std::atomic<uint64_t> total_{0};
};

} // namespace llmquant
