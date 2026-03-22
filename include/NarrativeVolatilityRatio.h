#pragma once

/// @file NarrativeVolatilityRatio.h
/// @brief Computes the ratio of short-term to long-term bias volatility, analogous
///        to the VIX term structure ratio used in options markets.
///
/// ## Motivation
/// The ratio σ_short / σ_long reveals:
///   > 1 → short-term vol > long-term vol: near-term uncertainty elevated,
///          backwardation-like vol term structure (fear/panic regime).
///   < 1 → contango-like structure: near-term calm, long-term risk builds.
///   ≈ 1 → flat vol term structure: uniform uncertainty.
///
/// Both σ estimates use exponential weighting (EMA of squared deviations)
/// with different decay constants, keeping computation O(1) per sample.
///
/// Fires `on_vol_regime_change(old_ratio, new_ratio)` when ratio crosses 1.0
/// or changes by more than `change_threshold`.
///
/// ## Feature flag
/// `LLMQUANT_ENABLE_VOL_RATIO` (default ON).
///
/// Thread-safe.

#include <atomic>
#include <cstdint>
#include <functional>
#include <mutex>
#include <string>

namespace llmquant {

class NarrativeVolatilityRatio {
public:
    struct Config {
        /// Short-term EMA decay α_s. Default 0.20 (≈5-period).
        double alpha_short{0.20};

        /// Long-term EMA decay α_l. Default 0.04 (≈25-period).
        double alpha_long{0.04};

        /// |Δratio| above this fires on_vol_regime_change. Default 0.2.
        double change_threshold{0.2};

        /// Minimum samples before publishing. Default 10.
        int min_samples{10};

        /// Fired when vol ratio changes significantly (outside lock).
        std::function<void(double old_ratio, double new_ratio)> on_vol_regime_change;
    };

    explicit NarrativeVolatilityRatio(Config cfg = Config{});

    // -----------------------------------------------------------------------
    // Data intake
    // -----------------------------------------------------------------------

    void record(double bias);

    // -----------------------------------------------------------------------
    // Accessors (lock-free)
    // -----------------------------------------------------------------------

    /// Short-term EMA volatility.
    [[nodiscard]] double short_vol() const noexcept {
        return short_vol_.load(std::memory_order_relaxed);
    }

    /// Long-term EMA volatility.
    [[nodiscard]] double long_vol() const noexcept {
        return long_vol_.load(std::memory_order_relaxed);
    }

    /// Volatility ratio σ_short / σ_long. >1 = elevated near-term uncertainty.
    [[nodiscard]] double vol_ratio() const noexcept {
        return ratio_.load(std::memory_order_relaxed);
    }

    /// Total vol-regime-change events.
    [[nodiscard]] uint64_t regime_change_events() const noexcept {
        return changes_.load(std::memory_order_relaxed);
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

    double mu_{0.0};         // running mean (fast EMA)
    double var_s_{0.0};      // short-term EMA variance
    double var_l_{0.0};      // long-term EMA variance
    double prev_ratio_{-1.0};
    bool   seeded_{false};

    std::atomic<double>   short_vol_{0.0};
    std::atomic<double>   long_vol_{0.0};
    std::atomic<double>   ratio_{1.0};
    std::atomic<uint64_t> changes_{0};
    std::atomic<uint64_t> total_{0};
};

} // namespace llmquant
