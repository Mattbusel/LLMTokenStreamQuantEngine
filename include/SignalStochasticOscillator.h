#pragma once

/// @file SignalStochasticOscillator.h
/// @brief Stochastic %K/%D oscillator adapted for LLM sentiment signal streams.
///
/// ## Motivation
/// The stochastic oscillator is a momentum indicator that measures where the current
/// signal level stands relative to its n-period high/low range.  Applied to sentiment:
///   - %K near 100% → sentiment is near its recent peak (overbought narrative)
///   - %K near 0%   → sentiment is near its recent trough (oversold narrative)
///
/// This is fundamentally different from MACD (which tracks trend) or CUSUM (which
/// tracks step changes) — the stochastic detects EXHAUSTION at extremes.
///
/// ## Algorithm
///   %K = 100 × (x_n - L_n) / (H_n - L_n)
///
/// where H_n = max(x_{n-k}, ..., x_n), L_n = min(x_{n-k}, ..., x_n) over k-period window.
///
///   %D = SMA(%K, d_period)   (signal line smoothing)
///
/// Fires `on_overbought` when %K crosses above overbought_threshold (default 80).
/// Fires `on_oversold`   when %K crosses below oversold_threshold  (default 20).
/// Fires `on_kd_bullish_cross` when %K crosses above %D from below (mean-reversion buy).
/// Fires `on_kd_bearish_cross` when %K crosses below %D from above (mean-reversion sell).
///
/// ## Feature flag
/// `LLMQUANT_ENABLE_STOCHASTIC_OSC` (default ON).
///
/// Thread-safe.

#include <atomic>
#include <cstdint>
#include <functional>
#include <mutex>
#include <string>
#include <vector>

namespace llmquant {

class SignalStochasticOscillator {
public:
    struct Config {
        /// %K lookback window (k period). Default 14.
        int k_period{14};

        /// %D smoothing period (SMA of %K). Default 3.
        int d_period{3};

        /// Overbought threshold (0-100). Default 80.
        double overbought_threshold{80.0};

        /// Oversold threshold (0-100). Default 20.
        double oversold_threshold{20.0};

        /// Minimum samples before oscillator outputs are valid. Default == k_period.
        int min_samples{0};  // 0 → use k_period

        /// Fired (outside lock) when %K crosses above overbought_threshold.
        std::function<void(double k_pct)> on_overbought;

        /// Fired (outside lock) when %K crosses below oversold_threshold.
        std::function<void(double k_pct)> on_oversold;

        /// Fired (outside lock) when %K crosses above %D (bullish signal).
        std::function<void(double k_pct, double d_pct)> on_kd_bullish_cross;

        /// Fired (outside lock) when %K crosses below %D (bearish signal).
        std::function<void(double k_pct, double d_pct)> on_kd_bearish_cross;
    };

    explicit SignalStochasticOscillator(Config cfg = Config{});

    // -----------------------------------------------------------------------
    // Data intake
    // -----------------------------------------------------------------------

    void record(double x);

    // -----------------------------------------------------------------------
    // Accessors (lock-free)
    // -----------------------------------------------------------------------

    /// Current %K value (0-100), or NaN if not enough data.
    [[nodiscard]] double k_pct() const noexcept {
        return k_.load(std::memory_order_relaxed);
    }

    /// Current %D value (smoothed %K), or NaN if not enough data.
    [[nodiscard]] double d_pct() const noexcept {
        return d_.load(std::memory_order_relaxed);
    }

    [[nodiscard]] bool is_overbought() const noexcept {
        return overbought_.load(std::memory_order_relaxed);
    }

    [[nodiscard]] bool is_oversold() const noexcept {
        return oversold_.load(std::memory_order_relaxed);
    }

    [[nodiscard]] uint64_t total_records() const noexcept {
        return total_.load(std::memory_order_relaxed);
    }

    [[nodiscard]] uint64_t overbought_events() const noexcept {
        return ob_events_.load(std::memory_order_relaxed);
    }

    [[nodiscard]] uint64_t oversold_events() const noexcept {
        return os_events_.load(std::memory_order_relaxed);
    }

    [[nodiscard]] std::string to_stats_json() const;

    void reset();
    void update_config(const Config& cfg);

private:
    mutable std::mutex mutex_;
    Config             cfg_;

    std::vector<double> buf_;    ///< circular buffer for k_period window
    int head_{0};
    int fill_{0};

    std::vector<double> d_buf_;  ///< circular buffer for d_period SMA of %K
    int d_head_{0};
    int d_fill_{0};
    double d_sum_{0.0};

    bool prev_overbought_{false};
    bool prev_oversold_{false};
    bool prev_k_above_d_{false};
    bool d_valid_{false};

    std::atomic<double>   k_{std::numeric_limits<double>::quiet_NaN()};
    std::atomic<double>   d_{std::numeric_limits<double>::quiet_NaN()};
    std::atomic<bool>     overbought_{false};
    std::atomic<bool>     oversold_{false};
    std::atomic<uint64_t> total_{0};
    std::atomic<uint64_t> ob_events_{0};
    std::atomic<uint64_t> os_events_{0};
};

} // namespace llmquant
