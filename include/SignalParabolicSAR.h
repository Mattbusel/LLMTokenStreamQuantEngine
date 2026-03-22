#pragma once

/// @file SignalParabolicSAR.h
/// @brief Parabolic SAR (Stop and Reverse) indicator adapted for the LLM bias
///        stream, providing trend direction and trailing stop levels.
///
/// ## Motivation
/// Parabolic SAR is a classic technical indicator that:
///   - Tracks trend direction (uptrend / downtrend).
///   - Provides an adaptive trailing stop (SAR value).
///   - Fires a reversal when bias crosses the SAR from the trending side.
///
/// Applied to a bias stream: "uptrend" = sustained positive bias;
/// "downtrend" = sustained negative bias.  SAR value acts as the
/// regime-change threshold for the current trend.
///
/// ## Algorithm
/// Standard Parabolic SAR:
///   SAR_{t+1} = SAR_t + AF * (EP - SAR_t)
/// where EP = extreme point (max/min bias seen in current trend),
///       AF = acceleration factor, starts at `af_start`, grows by `af_step`
///       each period, capped at `af_max`.
/// Reversal: if bias < SAR (in uptrend) or bias > SAR (in downtrend), flip.
///
/// Fires `on_reversal(new_direction, sar_value)` on trend flip.
///
/// ## Feature flag
/// `LLMQUANT_ENABLE_PARABOLIC_SAR` (default ON).
///
/// Thread-safe.

#include <atomic>
#include <cstdint>
#include <functional>
#include <mutex>
#include <string>

namespace llmquant {

class SignalParabolicSAR {
public:
    struct Config {
        /// Initial acceleration factor. Default 0.02.
        double af_start{0.02};

        /// AF increment per period. Default 0.02.
        double af_step{0.02};

        /// Maximum AF. Default 0.20.
        double af_max{0.20};

        /// Minimum samples before computing. Default 3.
        int min_samples{3};

        /// Fired on trend reversal (outside lock). Params: direction (+1=up, -1=down), SAR.
        std::function<void(int direction, double sar)> on_reversal;
    };

    explicit SignalParabolicSAR(Config cfg = Config{});

    // -----------------------------------------------------------------------
    // Data intake
    // -----------------------------------------------------------------------

    void record(double bias);

    // -----------------------------------------------------------------------
    // Accessors (lock-free)
    // -----------------------------------------------------------------------

    /// Current SAR value.
    [[nodiscard]] double sar() const noexcept {
        return sar_.load(std::memory_order_relaxed);
    }

    /// Current trend direction: +1 = uptrend, -1 = downtrend.
    [[nodiscard]] int trend() const noexcept {
        return trend_.load(std::memory_order_relaxed);
    }

    /// Current acceleration factor.
    [[nodiscard]] double af() const noexcept {
        return af_.load(std::memory_order_relaxed);
    }

    /// Total reversals detected.
    [[nodiscard]] uint64_t reversal_events() const noexcept {
        return reversals_.load(std::memory_order_relaxed);
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

    // SAR state
    double sar_val_{0.0};
    double ep_{0.0};        // extreme point
    double af_val_{0.02};
    int    dir_{1};         // +1 = uptrend, -1 = downtrend
    bool   seeded_{false};

    std::atomic<double>   sar_{0.0};
    std::atomic<int>      trend_{1};
    std::atomic<double>   af_{0.02};
    std::atomic<uint64_t> reversals_{0};
    std::atomic<uint64_t> total_{0};
};

} // namespace llmquant
