#pragma once

/// @file SignalBollingerBands.h
/// @brief Bollinger Bands (μ ± k·σ) applied to the LLM bias stream, using
///        rolling EMA and standard deviation for adaptive volatility-scaled bands.
///
/// ## Motivation
/// Bollinger Bands provide a volatility-normalised view of the bias stream:
///   - Bias touching the upper band → overbought / peak sentiment momentum.
///   - Bias touching the lower band → oversold / sentiment capitulation.
///   - Band width (%B) → volatility expansion/contraction indicator.
///   - %B = (bias - lower) / (upper - lower) ∈ [0, 1] (clamped).
///
/// Unlike BiasGarchEstimator (which is purely variance-focused), Bollinger Bands
/// produce actionable level crossings that integrate naturally with existing
/// signal gates.
///
/// Fires `on_upper_touch(bias)` when bias > upper band.
/// Fires `on_lower_touch(bias)` when bias < lower band.
/// Fires `on_squeeze(width)` when band width < `squeeze_threshold`.
///
/// ## Feature flag
/// `LLMQUANT_ENABLE_BOLLINGER_BANDS` (default ON).
///
/// Thread-safe.

#include <atomic>
#include <cstdint>
#include <functional>
#include <mutex>
#include <string>
#include <vector>

namespace llmquant {

class SignalBollingerBands {
public:
    struct Config {
        /// Rolling window N. Default 20.
        int window{20};

        /// Number of std devs for band width k. Default 2.0.
        double k{2.0};

        /// Minimum fill before computing. Default 5.
        int min_samples{5};

        /// Band width below this fires on_squeeze. Default 0.05.
        double squeeze_threshold{0.05};

        /// Fired when bias > upper band (outside lock).
        std::function<void(double bias)> on_upper_touch;

        /// Fired when bias < lower band (outside lock).
        std::function<void(double bias)> on_lower_touch;

        /// Fired when band width < squeeze_threshold (outside lock).
        std::function<void(double width)> on_squeeze;
    };

    explicit SignalBollingerBands(Config cfg = Config{});

    // -----------------------------------------------------------------------
    // Data intake
    // -----------------------------------------------------------------------

    void record(double bias);

    // -----------------------------------------------------------------------
    // Accessors (lock-free)
    // -----------------------------------------------------------------------

    [[nodiscard]] double middle()  const noexcept { return mid_.load(std::memory_order_relaxed); }
    [[nodiscard]] double upper()   const noexcept { return up_.load(std::memory_order_relaxed);  }
    [[nodiscard]] double lower()   const noexcept { return lo_.load(std::memory_order_relaxed);  }
    [[nodiscard]] double band_width() const noexcept {
        return up_.load(std::memory_order_relaxed) - lo_.load(std::memory_order_relaxed);
    }

    /// %B = (bias - lower) / (upper - lower + ε). 0 = at lower, 1 = at upper.
    [[nodiscard]] double pct_b() const noexcept { return pct_b_.load(std::memory_order_relaxed); }

    /// Total upper-band touch events.
    [[nodiscard]] uint64_t upper_events() const noexcept {
        return upper_events_.load(std::memory_order_relaxed);
    }
    /// Total lower-band touch events.
    [[nodiscard]] uint64_t lower_events() const noexcept {
        return lower_events_.load(std::memory_order_relaxed);
    }
    /// Total squeeze events.
    [[nodiscard]] uint64_t squeeze_events() const noexcept {
        return squeeze_events_.load(std::memory_order_relaxed);
    }
    /// Total observations recorded.
    [[nodiscard]] uint64_t total_records() const noexcept {
        return total_.load(std::memory_order_relaxed);
    }

    [[nodiscard]] std::string to_stats_json() const;

    void reset();
    void update_config(const Config& cfg);

private:
    mutable std::mutex mutex_;
    Config cfg_;

    std::vector<double> win_buf_;
    int win_head_{0};
    int win_fill_{0};

    std::atomic<double>   mid_{0.0};
    std::atomic<double>   up_{0.0};
    std::atomic<double>   lo_{0.0};
    std::atomic<double>   pct_b_{0.5};
    std::atomic<uint64_t> upper_events_{0};
    std::atomic<uint64_t> lower_events_{0};
    std::atomic<uint64_t> squeeze_events_{0};
    std::atomic<uint64_t> total_{0};
};

} // namespace llmquant
