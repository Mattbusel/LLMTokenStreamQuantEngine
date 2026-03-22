#pragma once

/// @file AdaptiveThresholdController.h
/// @brief Dynamically adjusts signal thresholds based on rolling bias volatility
///        so that threshold-based signals remain calibrated across regimes.
///
/// ## Motivation
/// Static overbought/oversold thresholds (e.g. SSI > 70) go stale when the
/// signal volatility changes.  In a high-volatility regime, SSI frequently
/// exceeds 70 just from noise; in a calm regime it never does.  This controller
/// widens thresholds proportionally to rolling volatility, keeping the
/// signal rate approximately constant regardless of market conditions.
///
/// ## Algorithm
/// rolling_sigma = EMA of |Δbias| with alpha = 1/period.
/// adjusted_overbought = base_overbought + k_sigma * rolling_sigma
/// adjusted_oversold   = base_oversold   - k_sigma * rolling_sigma
/// Both clamped to [clamp_lo, clamp_hi].
///
/// Fires `on_threshold_change(new_ob, new_os)` when either threshold changes
/// by more than `change_threshold`.
///
/// ## Feature flag
/// `LLMQUANT_ENABLE_ADAPTIVE_THRESHOLD` (default ON).
///
/// Thread-safe.

#include <atomic>
#include <cstdint>
#include <functional>
#include <mutex>
#include <string>

namespace llmquant {

class AdaptiveThresholdController {
public:
    struct Config {
        /// EMA period for rolling |Δbias| volatility. Default 20.
        int period{20};

        /// Base overbought threshold (before vol adjustment). Default 70.0.
        double base_overbought{70.0};

        /// Base oversold threshold (before vol adjustment). Default 30.0.
        double base_oversold{30.0};

        /// Multiplier for rolling sigma added/subtracted from thresholds. Default 50.0.
        double k_sigma{50.0};

        /// Minimum allowed overbought threshold. Default 55.0.
        double clamp_lo{55.0};

        /// Maximum allowed overbought threshold. Default 95.0.
        double clamp_hi{95.0};

        /// Minimum |Δthreshold| to fire callback. Default 1.0.
        double change_threshold{1.0};

        /// Fired when either threshold changes (outside lock).
        std::function<void(double ob, double os)> on_threshold_change;
    };

    explicit AdaptiveThresholdController(Config cfg = Config{});

    // -----------------------------------------------------------------------
    // Data intake
    // -----------------------------------------------------------------------

    /// Record a new bias value. Updates rolling volatility and adjusts thresholds.
    void record(double bias);

    // -----------------------------------------------------------------------
    // Accessors (lock-free)
    // -----------------------------------------------------------------------

    /// Current adaptive overbought threshold.
    [[nodiscard]] double overbought_threshold() const noexcept {
        return ob_.load(std::memory_order_relaxed);
    }

    /// Current adaptive oversold threshold.
    [[nodiscard]] double oversold_threshold() const noexcept {
        return os_.load(std::memory_order_relaxed);
    }

    /// Current rolling volatility estimate (EMA of |Δbias|).
    [[nodiscard]] double rolling_sigma() const noexcept {
        return sigma_.load(std::memory_order_relaxed);
    }

    /// Number of threshold-change events fired.
    [[nodiscard]] uint64_t change_events() const noexcept {
        return change_events_.load(std::memory_order_relaxed);
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

    double sigma_ema_{0.0};
    double prev_bias_{0.0};
    bool   seeded_{false};

    double prev_ob_{-1.0};  // <0 = uninitialised
    double prev_os_{-1.0};

    std::atomic<double>   ob_{70.0};
    std::atomic<double>   os_{30.0};
    std::atomic<double>   sigma_{0.0};
    std::atomic<uint64_t> change_events_{0};
    std::atomic<uint64_t> total_{0};
};

} // namespace llmquant
