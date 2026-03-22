#pragma once
#include <atomic>
#include <functional>
#include <mutex>
#include <string>
#include <vector>

namespace llmquant {

/**
 * @brief Relative Vigor Index (RVI) adapted for bias streams.
 *
 * RVI = EMA(close - open) / EMA(high - low) over a rolling window,
 * approximated here as: numerator EMA of bias change, denominator EMA of
 * absolute bias range.  A signal line (EMA of RVI) is also tracked.
 * Cross of RVI above signal → bullish; below → bearish.
 */
class SignalRelativeVigorIndex {
public:
    struct Config {
        double alpha_rvi{0.2};        ///< EMA factor for RVI numerator/denominator
        double alpha_signal{0.25};    ///< EMA factor for signal line
        int    min_samples{5};        ///< Warm-up samples before signal generation
        double cross_threshold{0.0};  ///< Dead-band around signal cross

        /// Fired on bullish cross (RVI rises above signal line).
        std::function<void(double /*rvi*/)> on_bullish_cross;
        /// Fired on bearish cross (RVI falls below signal line).
        std::function<void(double /*rvi*/)> on_bearish_cross;
    };

    explicit SignalRelativeVigorIndex(Config cfg = {});

    /// Feed a new bias sample.
    void record(double bias);

    double rvi()         const noexcept;
    double signal_line() const noexcept;
    bool   is_bullish()  const noexcept;

    std::size_t bullish_crosses() const noexcept;
    std::size_t bearish_crosses() const noexcept;
    std::size_t total_records()   const noexcept;

    std::string to_stats_json() const;
    void reset();
    void update_config(const Config& cfg);

private:
    mutable std::mutex mutex_;
    Config cfg_;
    bool seeded_{false};
    double prev_bias_{0.0};
    double ema_num_{0.0};   ///< EMA of bias changes (close-open proxy)
    double ema_den_{0.0};   ///< EMA of |bias| (high-low proxy)
    double ema_sig_{0.0};   ///< Signal line EMA of RVI
    bool prev_bullish_{false};

    std::atomic<double>   rvi_{0.0};
    std::atomic<double>   sig_{0.0};
    std::atomic<bool>     bullish_{false};
    std::atomic<uint64_t> bull_crosses_{0};
    std::atomic<uint64_t> bear_crosses_{0};
    std::atomic<uint64_t> total_{0};
};

} // namespace llmquant
