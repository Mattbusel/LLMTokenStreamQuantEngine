#pragma once

/// @file OptionsFlowSentimentBridge.h
/// @brief Detects divergence between LLM sentiment velocity and options-market
///        implied-volatility skew, flagging smart-money positioning signals.
///
/// ## Motivation
/// When narrative sentiment accelerates bullishly but the IV skew (put IV − call IV)
/// simultaneously widens, it suggests institutional hedgers are quietly buying
/// downside protection against the retail-driven bullish narrative.  Conversely,
/// bullish skew collapse (calls bid aggressively) ahead of positive LLM sentiment
/// peaks can front-run an earnings catalyst.
///
/// ## Algorithm
/// - Maintains an EMA of sentiment velocity `v_t = (bias_t − bias_{t-1}) / dt`.
/// - Accepts `record_skew(put_iv, call_iv)` observations from the options feed.
/// - Computes `skew_t = put_iv − call_iv` and its EMA.
/// - Divergence score `D = sentiment_velocity_ema * sign(−skew_ema)`:
///     - D > `div_threshold`  → smart-money BEAR divergence (narrative up, hedging up)
///     - D < −`div_threshold` → smart-money BULL divergence (narrative down, calls bid)
/// - Fires `on_divergence(DivergenceKind, score, sentiment_vel, skew)` outside lock.
///
/// ## Feature flag
/// `LLMQUANT_ENABLE_OPTIONS_FLOW_BRIDGE` (default ON).
///
/// Thread-safe: all public methods are thread-safe.

#include <atomic>
#include <cstdint>
#include <functional>
#include <mutex>
#include <string>

namespace llmquant {

class OptionsFlowSentimentBridge {
public:
    enum class DivergenceKind {
        None,
        SmartMoneyBear,  ///< Narrative bullish, IV skew widening → stealth hedging
        SmartMoneyBull,  ///< Narrative bearish, calls bid aggressively → stealth accumulation
    };

    struct Config {
        /// EMA alpha for sentiment velocity smoothing (0 < alpha ≤ 1). Default 0.15.
        double velocity_alpha{0.15};

        /// EMA alpha for IV skew smoothing. Default 0.2.
        double skew_alpha{0.20};

        /// Divergence score threshold to fire the callback. Default 0.03.
        double div_threshold{0.03};

        /// Hysteresis: once a divergence fires, score must drop below
        /// (1 − hysteresis) × div_threshold before re-arming. Default 0.3.
        double hysteresis{0.30};

        /// Minimum sentiment observations before divergence detection is active.
        int min_warmup{10};

        /// Fired when divergence is detected or recovers.
        /// Parameters: (kind, divergence_score, sentiment_velocity_ema, skew_ema).
        std::function<void(DivergenceKind kind,
                           double score,
                           double sentiment_vel_ema,
                           double skew_ema)> on_divergence;
    };

    explicit OptionsFlowSentimentBridge(Config cfg = Config{});

    // -----------------------------------------------------------------------
    // Data intake
    // -----------------------------------------------------------------------

    /// Record a new sentiment bias sample.
    /// @param bias  Current aggregated sentiment bias ∈ (−∞, ∞).
    /// @param dt_s  Seconds since last bias sample (must be > 0).
    void record_bias(double bias, double dt_s);

    /// Record an options market IV observation.
    /// @param put_iv   Implied volatility of the at-money put.
    /// @param call_iv  Implied volatility of the at-money call.
    void record_skew(double put_iv, double call_iv);

    // -----------------------------------------------------------------------
    // Accessors (lock-free)
    // -----------------------------------------------------------------------

    /// Current smoothed sentiment velocity (bias units / second).
    [[nodiscard]] double sentiment_velocity_ema() const noexcept {
        return vel_ema_.load(std::memory_order_relaxed);
    }

    /// Current smoothed IV skew (put IV − call IV).
    [[nodiscard]] double skew_ema() const noexcept {
        return skew_ema_.load(std::memory_order_relaxed);
    }

    /// Current divergence score.
    [[nodiscard]] double divergence_score() const noexcept {
        return div_score_.load(std::memory_order_relaxed);
    }

    /// Most recently detected divergence kind.
    [[nodiscard]] DivergenceKind last_divergence() const noexcept {
        return static_cast<DivergenceKind>(last_kind_.load(std::memory_order_relaxed));
    }

    /// Total divergence events fired.
    [[nodiscard]] uint64_t divergence_count() const noexcept {
        return div_count_.load(std::memory_order_relaxed);
    }

    /// JSON diagnostics snapshot.
    [[nodiscard]] std::string to_stats_json() const;

    /// Reset all state.
    void reset();

    /// Update configuration (resets state).
    void update_config(const Config& cfg);

private:
    mutable std::mutex mutex_;
    Config cfg_;

    double prev_bias_{0.0};
    bool   bias_seeded_{false};
    int    warmup_count_{0};

    double vel_ema_val_{0.0};
    double skew_ema_val_{0.0};
    bool   armed_{false};  // true when not in active divergence

    std::atomic<double>   vel_ema_{0.0};
    std::atomic<double>   skew_ema_{0.0};
    std::atomic<double>   div_score_{0.0};
    std::atomic<int>      last_kind_{0};
    std::atomic<uint64_t> div_count_{0};
};

} // namespace llmquant
