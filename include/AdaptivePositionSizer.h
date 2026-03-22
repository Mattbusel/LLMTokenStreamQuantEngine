#pragma once

/// @file AdaptivePositionSizer.h
/// @brief Combines multiple signal-quality dimensions into a single normalised
///        position-size multiplier in [0, 1].
///
/// ## Motivation
/// Position sizing is the single highest-impact lever in a quantitative strategy.
/// This module synthesises coherence, volatility, confidence, and regime signals
/// into one multiplier that can be applied directly to the raw signal bias before
/// passing to the OMS.  All inputs are optional; missing inputs default to their
/// neutral values so the module degrades gracefully.
///
/// ## Algorithm
/// multiplier = prod_i( weight_i * factor_i + (1 - weight_i) )
///
/// where each factor ∈ [0, 1]:
///   - coherence_factor: raw coherence score clamped to [0, 1].
///   - confidence_factor: raw confidence clamped to [0, 1].
///   - vol_factor:        1 - clamp(rv / rv_max, 0, 1)  (high vol → small size).
///   - regime_factor:     per-regime multiplier (Trending=1, Ranging=0.5, Crash=0.1).
///
/// Final multiplier is clamped to [min_multiplier, 1].
/// Fires `on_size_change(old_mult, new_mult)` when multiplier changes by >
/// `change_threshold`.
///
/// ## Feature flag
/// `LLMQUANT_ENABLE_ADAPTIVE_SIZER` (default ON).
///
/// Thread-safe.

#include <atomic>
#include <cstdint>
#include <functional>
#include <mutex>
#include <string>

namespace llmquant {

class AdaptivePositionSizer {
public:
    enum class Regime { Trending = 0, Ranging = 1, Crash = 2 };

    struct Config {
        /// Weight applied to coherence_factor. Default 0.25.
        double coherence_weight{0.25};

        /// Weight applied to confidence_factor. Default 0.25.
        double confidence_weight{0.25};

        /// Weight applied to vol_factor. Default 0.25.
        double vol_weight{0.25};

        /// Weight applied to regime_factor. Default 0.25.
        double regime_weight{0.25};

        /// Realised-vol value that maps to vol_factor=0 (maximum de-sizing). Default 0.5.
        double rv_max{0.5};

        /// Per-regime size multipliers. Defaults: Trending=1.0, Ranging=0.5, Crash=0.1.
        double regime_factor_trending{1.0};
        double regime_factor_ranging{0.5};
        double regime_factor_crash{0.1};

        /// Hard floor for multiplier output. Default 0.05.
        double min_multiplier{0.05};

        /// |Δmult| above this fires on_size_change. Default 0.05.
        double change_threshold{0.05};

        /// Fired when multiplier changes by > change_threshold (outside lock).
        std::function<void(double old_mult, double new_mult)> on_size_change;
    };

    explicit AdaptivePositionSizer(Config cfg = Config{});

    // -----------------------------------------------------------------------
    // Input updates (each call recomputes the multiplier)
    // -----------------------------------------------------------------------

    /// Update coherence factor ∈ [0, 1] and recompute.
    void set_coherence(double coherence);

    /// Update confidence factor ∈ [0, 1] and recompute.
    void set_confidence(double confidence);

    /// Update realised volatility and recompute.
    void set_realized_vol(double rv);

    /// Update the current regime and recompute.
    void set_regime(Regime r);

    // -----------------------------------------------------------------------
    // Accessors (lock-free)
    // -----------------------------------------------------------------------

    /// Current position-size multiplier ∈ [min_multiplier, 1].
    [[nodiscard]] double multiplier() const noexcept {
        return mult_.load(std::memory_order_relaxed);
    }

    /// Number of size-change events fired.
    [[nodiscard]] uint64_t change_events() const noexcept {
        return change_events_.load(std::memory_order_relaxed);
    }

    [[nodiscard]] std::string to_stats_json() const;

    void reset();
    void update_config(const Config& cfg);

private:
    void recompute_locked();

    mutable std::mutex mutex_;
    Config cfg_;

    double coherence_input_{1.0};
    double confidence_input_{1.0};
    double rv_input_{0.0};
    Regime regime_input_{Regime::Trending};

    double prev_mult_{-1.0};  // <0 = uninitialised

    std::atomic<double>   mult_{1.0};
    std::atomic<uint64_t> change_events_{0};
};

} // namespace llmquant
