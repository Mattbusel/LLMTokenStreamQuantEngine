#pragma once
/// SignalTripleEMAOscillator — TRIX oscillator applied to the LLM bias stream.
/// TRIX = 1-period rate-of-change of triple-smoothed EMA.
/// Eliminates noise up to 3× the EMA period; only sustained trends register.
/// Fires on_zero_cross when TRIX changes sign (trend flip confirmation)
/// and on_signal_cross when TRIX crosses its signal line.
///
/// Feature flag: LLMQUANT_ENABLE_TRIPLE_EMA / LLMQUANT_TRIPLE_EMA_ENABLED

#ifdef LLMQUANT_TRIPLE_EMA_ENABLED

#include <atomic>
#include <functional>
#include <mutex>
#include <string>

namespace llmquant {

class SignalTripleEMAOscillator {
public:
    struct Config {
        double alpha        = 2.0 / (14 + 1); ///< smoothing factor for triple EMA (≈14-period)
        double alpha_signal = 2.0 / (9 + 1);  ///< smoothing factor for signal line
        int    min_samples  = 14;              ///< warm-up before firing events

        /// Fires when TRIX changes sign (strong trend confirmation).
        std::function<void(double trix)> on_zero_cross;
        /// Fires when TRIX crosses its signal line (earlier trigger).
        std::function<void(double trix, double signal)> on_signal_cross;
    };

    explicit SignalTripleEMAOscillator(Config cfg = {});

    void record(double bias);

    double ema1()        const noexcept; ///< single-smoothed EMA
    double ema3()        const noexcept; ///< triple-smoothed EMA
    double trix()        const noexcept; ///< TRIX value (rate of change %)
    double signal_line() const noexcept; ///< EMA of TRIX

    std::size_t zero_cross_events()   const noexcept;
    std::size_t signal_cross_events() const noexcept;
    std::size_t total_records()       const noexcept;

    std::string to_stats_json() const;
    void reset();
    void update_config(const Config& cfg);

private:
    mutable std::mutex mutex_;
    Config cfg_;

    bool   seeded_       = false;
    double ema1_raw_     = 0.0;
    double ema2_raw_     = 0.0;
    double ema3_raw_     = 0.0;
    double prev_ema3_    = 0.0;
    double ema_sig_raw_  = 0.0;
    bool   prev_trix_pos_  = false; ///< previous TRIX > 0
    bool   prev_above_sig_ = false; ///< previous TRIX > signal

    std::atomic<double>      ema1_{0.0};
    std::atomic<double>      ema3_{0.0};
    std::atomic<double>      trix_{0.0};
    std::atomic<double>      signal_{0.0};
    std::atomic<std::size_t> zero_cross_{0};
    std::atomic<std::size_t> sig_cross_{0};
    std::atomic<std::size_t> total_{0};
};

} // namespace llmquant

#endif // LLMQUANT_TRIPLE_EMA_ENABLED
