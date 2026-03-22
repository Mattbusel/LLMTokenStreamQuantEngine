#pragma once

/// @file SignalStrengthIndexer.h
/// @brief Relative Strength Index (RSI) applied to the sentiment signal itself.
///
/// ## Motivation
/// RSI is a classic momentum oscillator.  Applied to the token bias series
/// rather than prices, it produces a "Sentiment SSI" that identifies:
/// - **Overbought** (SSI > overbought_threshold, default 70): sentiment has
///   risen too fast; high probability of reversion or narrative exhaustion.
/// - **Oversold**  (SSI < oversold_threshold, default 30): sentiment has
///   fallen too fast; potential for bullish narrative recovery.
///
/// ## Algorithm
/// SSI = 100 − 100 / (1 + RS)
/// RS  = avg_gain / avg_loss   over `period` observations
/// where gain_i = max(0, bias_i − bias_{i−1}) and loss_i = max(0, bias_{i-1} − bias_i).
/// Uses Wilder's smoothing (EMA with alpha = 1/period).
///
/// Fires `on_overbought(ssi)` and `on_oversold(ssi)` outside the lock.
///
/// ## Feature flag
/// `LLMQUANT_ENABLE_SIGNAL_SSI` (default ON).
///
/// Thread-safe.

#include <atomic>
#include <cstdint>
#include <functional>
#include <mutex>
#include <string>

namespace llmquant {

class SignalStrengthIndexer {
public:
    struct Config {
        /// RSI period (Wilder smoothing). Default 14.
        int period{14};

        /// SSI > this → overbought callback. Default 70.
        double overbought_threshold{70.0};

        /// SSI < this → oversold callback. Default 30.
        double oversold_threshold{30.0};

        /// Fired when SSI crosses above overbought_threshold.
        std::function<void(double ssi)> on_overbought;

        /// Fired when SSI crosses below oversold_threshold.
        std::function<void(double ssi)> on_oversold;
    };

    explicit SignalStrengthIndexer(Config cfg = Config{});

    // -----------------------------------------------------------------------
    // Data intake
    // -----------------------------------------------------------------------

    /// Record a new bias observation.
    void record(double bias);

    // -----------------------------------------------------------------------
    // Accessors (lock-free)
    // -----------------------------------------------------------------------

    /// Current SSI ∈ [0, 100].
    [[nodiscard]] double ssi() const noexcept {
        return ssi_.load(std::memory_order_relaxed);
    }

    /// True when SSI > overbought_threshold.
    [[nodiscard]] bool is_overbought() const noexcept {
        return overbought_.load(std::memory_order_relaxed);
    }

    /// True when SSI < oversold_threshold.
    [[nodiscard]] bool is_oversold() const noexcept {
        return oversold_.load(std::memory_order_relaxed);
    }

    /// Total observations recorded.
    [[nodiscard]] uint64_t observation_count() const noexcept {
        return obs_count_.load(std::memory_order_relaxed);
    }

    [[nodiscard]] std::string to_stats_json() const;

    void reset();
    void update_config(const Config& cfg);

private:
    mutable std::mutex mutex_;
    Config cfg_;

    double prev_bias_{0.0};
    double avg_gain_{0.0};
    double avg_loss_{0.0};
    bool   seeded_{false};
    int    warmup_{0};

    // Overbought/oversold state (hysteresis via crossover tracking).
    bool   prev_overbought_{false};
    bool   prev_oversold_{false};

    std::atomic<double>   ssi_{50.0};
    std::atomic<bool>     overbought_{false};
    std::atomic<bool>     oversold_{false};
    std::atomic<uint64_t> obs_count_{0};
};

} // namespace llmquant
