#pragma once
#include <atomic>
#include <functional>
#include <mutex>
#include <string>
#include <vector>

namespace llmquant {

/**
 * @brief Chande Momentum Oscillator (CMO) for bias streams.
 *
 * CMO = 100 × (ΣUp − ΣDown) / (ΣUp + ΣDown) over a rolling window.
 * Range: [−100, +100].  Overbought > +threshold; oversold < −threshold.
 */
class SignalChandeOscillator {
public:
    struct Config {
        int    window{14};               ///< Rolling look-back period
        int    min_samples{3};           ///< Minimum samples before CMO is valid
        double overbought_threshold{50}; ///< CMO above this → overbought
        double oversold_threshold{-50};  ///< CMO below this → oversold

        /// Fired when CMO crosses above overbought_threshold.
        std::function<void(double /*cmo*/)> on_overbought;
        /// Fired when CMO crosses below oversold_threshold.
        std::function<void(double /*cmo*/)> on_oversold;
    };

    explicit SignalChandeOscillator(Config cfg = {});

    /// Feed a new bias sample.
    void record(double bias);

    double value()         const noexcept;
    bool   is_overbought() const noexcept;
    bool   is_oversold()   const noexcept;

    std::size_t overbought_events() const noexcept;
    std::size_t oversold_events()   const noexcept;
    std::size_t total_records()     const noexcept;

    std::string to_stats_json() const;
    void reset();
    void update_config(const Config& cfg);

private:
    mutable std::mutex  mutex_;
    Config              cfg_;
    std::vector<double> win_buf_;   ///< Rolling window of bias values
    int  win_head_{0};
    int  win_fill_{0};
    bool prev_valid_{false};
    double prev_val_{0.0};

    std::atomic<double>   cmo_{0.0};
    std::atomic<bool>     overbought_{false};
    std::atomic<bool>     oversold_{false};
    std::atomic<uint64_t> ob_events_{0};
    std::atomic<uint64_t> os_events_{0};
    std::atomic<uint64_t> total_{0};
};

} // namespace llmquant
