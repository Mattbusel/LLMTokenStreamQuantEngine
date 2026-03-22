#pragma once

/// @file LLMConfidenceBandTracker.h
/// @brief Kalman-filter confidence bands around the sentiment EMA.
///
/// ## Motivation
/// A raw EMA of sentiment bias is noisy.  Standard deviation alone cannot
/// distinguish between *trending* sentiment (bands should narrow as the filter
/// gains confidence) and *reverting* sentiment (bands should widen as uncertainty
/// grows).  A scalar Kalman filter with a tuneable process-noise / measurement-noise
/// ratio achieves exactly this: it integrates sequentially, tightening the band
/// when observations are consistent and widening it when the series is erratic.
///
/// ## Algorithm
/// State: `x` = latent true sentiment bias.
/// - **Predict**: `x_pred = x`, `P_pred = P + Q`  (Q = process noise variance)
/// - **Update** : `K = P_pred / (P_pred + R)`,  `x = x_pred + K*(obs - x_pred)`,
///               `P = (1 - K) * P_pred`
/// - **Bands**  : `[x − z*sqrt(P), x + z*sqrt(P)]`
///
/// Fires `on_band_change(lower, center, upper)` when the band width changes by
/// more than `band_change_threshold` (fractional).
///
/// ## Feature flag
/// `LLMQUANT_ENABLE_CONFIDENCE_BAND` (default ON).
///
/// Thread-safe.

#include <atomic>
#include <cstdint>
#include <functional>
#include <mutex>
#include <string>

namespace llmquant {

class LLMConfidenceBandTracker {
public:
    struct Config {
        /// Kalman process noise variance Q. Higher = faster adaptation. Default 0.001.
        double process_noise{0.001};

        /// Kalman measurement noise variance R. Higher = more smoothing. Default 0.01.
        double measurement_noise{0.01};

        /// Initial state variance P₀. Default 1.0.
        double initial_variance{1.0};

        /// z-score multiplier for the confidence band half-width. Default 2.0 (≈95%).
        double z_score{2.0};

        /// Fractional band-width change threshold to fire callback. Default 0.1 (10%).
        double band_change_threshold{0.10};

        /// Fired when band width changes significantly.
        /// Parameters: (lower_bound, center_estimate, upper_bound).
        std::function<void(double lower, double center, double upper)> on_band_change;
    };

    explicit LLMConfidenceBandTracker(Config cfg = Config{});

    // -----------------------------------------------------------------------
    // Data intake
    // -----------------------------------------------------------------------

    /// Record a new bias observation and update the Kalman filter.
    void update(double bias);

    // -----------------------------------------------------------------------
    // Accessors (lock-free)
    // -----------------------------------------------------------------------

    /// Kalman-filtered sentiment estimate (center of band).
    [[nodiscard]] double center() const noexcept {
        return center_.load(std::memory_order_relaxed);
    }

    /// Lower confidence band.
    [[nodiscard]] double lower() const noexcept {
        return lower_.load(std::memory_order_relaxed);
    }

    /// Upper confidence band.
    [[nodiscard]] double upper() const noexcept {
        return upper_.load(std::memory_order_relaxed);
    }

    /// Current band half-width (z * sqrt(P)).
    [[nodiscard]] double half_width() const noexcept {
        return half_width_.load(std::memory_order_relaxed);
    }

    /// Current posterior variance P.
    [[nodiscard]] double variance() const noexcept {
        return variance_.load(std::memory_order_relaxed);
    }

    /// Total observations processed.
    [[nodiscard]] uint64_t observation_count() const noexcept {
        return obs_count_.load(std::memory_order_relaxed);
    }

    /// Total band-change callbacks fired.
    [[nodiscard]] uint64_t band_change_count() const noexcept {
        return band_changes_.load(std::memory_order_relaxed);
    }

    /// JSON diagnostics snapshot.
    [[nodiscard]] std::string to_stats_json() const;

    /// Reset to initial state.
    void reset();

    /// Update configuration (resets state).
    void update_config(const Config& cfg);

private:
    mutable std::mutex mutex_;
    Config cfg_;

    double x_{0.0};    // state estimate
    double P_{1.0};    // posterior variance
    bool   seeded_{false};
    double prev_half_width_{-1.0};

    std::atomic<double>   center_{0.0};
    std::atomic<double>   lower_{0.0};
    std::atomic<double>   upper_{0.0};
    std::atomic<double>   half_width_{0.0};
    std::atomic<double>   variance_{1.0};
    std::atomic<uint64_t> obs_count_{0};
    std::atomic<uint64_t> band_changes_{0};
};

} // namespace llmquant
