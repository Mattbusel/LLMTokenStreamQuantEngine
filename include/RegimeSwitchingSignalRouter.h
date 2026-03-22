#pragma once

/// @file RegimeSwitchingSignalRouter.h
/// @brief Three-state finite-state-machine signal router that selects strategy
///        configuration (bias multiplier, cooldown, risk limits) based on the
///        detected market regime (Trending / Ranging / Crash).
///
/// ## Motivation
/// A fixed signal mapping fails across regimes: the same bullish bias shift that
/// is a strong buy signal in a trending market is noise in a ranging market and
/// a dead-cat-bounce in a crash.  This router maintains a 3-state FSM driven by
/// two observables — recent volatility and momentum — and maps each regime to its
/// own tuned configuration, dynamically switching as observables cross thresholds.
///
/// ## Algorithm
/// State machine:
///   - **Trending** : `|momentum| > trend_threshold AND vol < crash_vol_threshold`
///   - **Ranging**  : `|momentum| <= trend_threshold AND vol < crash_vol_threshold`
///   - **Crash**    : `vol >= crash_vol_threshold`
///
/// Transitions are hysteresis-gated to prevent rapid flipping.
///
/// Each state carries:
///   - `bias_multiplier` — scale factor applied to incoming signal bias.
///   - `cooldown_ms`     — override cooldown between signals.
///   - `max_risk_fraction` — risk limit override (fraction of full position).
///
/// Fires `on_regime_change(old_state, new_state)` outside the lock.
///
/// ## Feature flag
/// `LLMQUANT_ENABLE_REGIME_ROUTER` (default ON).
///
/// Thread-safe.

#include <atomic>
#include <cstdint>
#include <functional>
#include <mutex>
#include <string>

namespace llmquant {

class RegimeSwitchingSignalRouter {
public:
    enum class Regime : int {
        Trending = 0,
        Ranging  = 1,
        Crash    = 2,
    };

    struct RegimeConfig {
        double bias_multiplier{1.0};    ///< Scale bias by this factor.
        double max_risk_fraction{1.0};  ///< Fraction of full position allowed.
        int    cooldown_ms{500};        ///< Cooldown between signals (ms).
    };

    struct Config {
        /// |Momentum| threshold to enter trending regime. Default 0.05.
        double trend_threshold{0.05};

        /// Volatility threshold to enter crash regime. Default 0.10.
        double crash_vol_threshold{0.10};

        /// Hysteresis: thresholds are scaled by (1 ± hysteresis) for re-entry. Default 0.15.
        double hysteresis{0.15};

        /// EMA alpha for volatility smoothing. Default 0.2.
        double vol_alpha{0.2};

        /// EMA alpha for momentum smoothing. Default 0.15.
        double momentum_alpha{0.15};

        /// Per-regime configurations.
        RegimeConfig trending_cfg;
        RegimeConfig ranging_cfg;
        RegimeConfig crash_cfg;

        /// Fired when regime changes.
        /// Parameters: (old_regime, new_regime, new_bias_multiplier).
        std::function<void(Regime old_r, Regime new_r, double bias_mult)> on_regime_change;
    };

    explicit RegimeSwitchingSignalRouter(Config cfg = Config{});

    // -----------------------------------------------------------------------
    // Data intake
    // -----------------------------------------------------------------------

    /// Update with a new return observation (used to track vol + momentum).
    /// @param ret  Period return (e.g. 5-second mark-to-market).
    void record_return(double ret);

    // -----------------------------------------------------------------------
    // Signal routing
    // -----------------------------------------------------------------------

    /// Apply the current regime's bias multiplier to an incoming signal bias.
    [[nodiscard]] double route_bias(double raw_bias) const noexcept {
        return raw_bias * bias_mult_.load(std::memory_order_relaxed);
    }

    /// Current regime cooldown override (ms).
    [[nodiscard]] int cooldown_ms() const noexcept {
        return cooldown_ms_.load(std::memory_order_relaxed);
    }

    /// Current regime max risk fraction.
    [[nodiscard]] double max_risk_fraction() const noexcept {
        return risk_frac_.load(std::memory_order_relaxed);
    }

    // -----------------------------------------------------------------------
    // Accessors
    // -----------------------------------------------------------------------

    [[nodiscard]] Regime current_regime() const noexcept {
        return static_cast<Regime>(regime_.load(std::memory_order_relaxed));
    }

    [[nodiscard]] std::string regime_name() const noexcept;

    [[nodiscard]] double smoothed_volatility() const noexcept {
        return vol_ema_.load(std::memory_order_relaxed);
    }

    [[nodiscard]] double smoothed_momentum() const noexcept {
        return mom_ema_.load(std::memory_order_relaxed);
    }

    [[nodiscard]] uint64_t regime_change_count() const noexcept {
        return regime_changes_.load(std::memory_order_relaxed);
    }

    [[nodiscard]] std::string to_stats_json() const;

    void reset();
    void update_config(const Config& cfg);

private:
    void apply_regime_locked(Regime r);
    Regime classify_locked() const;

    mutable std::mutex mutex_;
    Config cfg_;

    double vol_ema_val_{0.0};
    double mom_ema_val_{0.0};
    double prev_ret_{0.0};
    bool   seeded_{false};
    Regime current_regime_val_{Regime::Ranging};

    std::atomic<int>      regime_{static_cast<int>(Regime::Ranging)};
    std::atomic<double>   bias_mult_{1.0};
    std::atomic<int>      cooldown_ms_{500};
    std::atomic<double>   risk_frac_{1.0};
    std::atomic<double>   vol_ema_{0.0};
    std::atomic<double>   mom_ema_{0.0};
    std::atomic<uint64_t> regime_changes_{0};
};

} // namespace llmquant
