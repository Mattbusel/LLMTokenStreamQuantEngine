#pragma once

/// @file SignalShadowPortfolio.h
/// @brief Paper-trades every signal in a simulated portfolio running in parallel
///        with the live strategy, providing real-time counterfactual attribution.
///
/// ## Motivation
/// Position limits, risk gates, and cooldown windows all cause the live strategy
/// to deviate from the raw signal.  The shadow portfolio simulates what the
/// strategy *would* have made if every signal were executed at face value,
/// making it possible to attribute P&L drag to each specific risk constraint.
///
/// ## Algorithm
/// - `record_signal(signal_bias, confidence)` opens/adjusts a paper position:
///   `shadow_position = signal_bias × confidence × unit_size`.
/// - `record_price(price)` marks the shadow position to market:
///   `shadow_pnl += shadow_position × (price − prev_price)`.
/// - `record_live_signal(actual_bias)` tracks what the live strategy did.
/// - Attribution: `constraint_drag = shadow_pnl − live_pnl`.
/// - Fires `on_drag_threshold(drag, shadow_pnl, live_pnl)` when drag exceeds
///   `drag_alert_threshold` (absolute P&L units).
///
/// ## Feature flag
/// `LLMQUANT_ENABLE_SHADOW_PORTFOLIO` (default ON).
///
/// Thread-safe.

#include <atomic>
#include <cstdint>
#include <functional>
#include <mutex>
#include <string>

namespace llmquant {

class SignalShadowPortfolio {
public:
    struct Config {
        /// Notional unit size per unit of bias×confidence. Default 1000.0.
        double unit_size{1000.0};

        /// Maximum absolute shadow position (units). Default 10000.0.
        double max_position{10000.0};

        /// Drag threshold that triggers the alert callback (absolute P&L). Default 50.0.
        double drag_alert_threshold{50.0};

        /// Fired when constraint drag exceeds the alert threshold.
        /// Parameters: (drag, shadow_pnl, live_pnl).
        std::function<void(double drag, double shadow_pnl, double live_pnl)> on_drag_alert;
    };

    explicit SignalShadowPortfolio(Config cfg = Config{});

    // -----------------------------------------------------------------------
    // Data intake
    // -----------------------------------------------------------------------

    /// Record a raw signal (what the strategy *wants* to do).
    /// @param bias        Signal bias ∈ (−1, 1).
    /// @param confidence  Signal confidence ∈ [0, 1].
    void record_signal(double bias, double confidence);

    /// Record what the live strategy actually did (post risk-gate).
    /// @param actual_bias  Actual bias applied ∈ (−1, 1).
    void record_live_signal(double actual_bias);

    /// Update mark-to-market for both shadow and live portfolios.
    /// @param price  Current mid price.
    void record_price(double price);

    // -----------------------------------------------------------------------
    // Accessors (lock-free)
    // -----------------------------------------------------------------------

    /// Cumulative shadow P&L (unconstrained).
    [[nodiscard]] double shadow_pnl() const noexcept {
        return shadow_pnl_.load(std::memory_order_relaxed);
    }

    /// Cumulative live P&L (constrained by risk gates).
    [[nodiscard]] double live_pnl() const noexcept {
        return live_pnl_.load(std::memory_order_relaxed);
    }

    /// P&L drag attributable to risk constraints (shadow − live).
    [[nodiscard]] double constraint_drag() const noexcept {
        return shadow_pnl_.load(std::memory_order_relaxed)
             - live_pnl_.load(std::memory_order_relaxed);
    }

    /// Current shadow position.
    [[nodiscard]] double shadow_position() const noexcept {
        return shadow_pos_.load(std::memory_order_relaxed);
    }

    /// Current live position.
    [[nodiscard]] double live_position() const noexcept {
        return live_pos_.load(std::memory_order_relaxed);
    }

    /// Total price updates recorded.
    [[nodiscard]] uint64_t price_updates() const noexcept {
        return price_updates_.load(std::memory_order_relaxed);
    }

    /// Total drag alerts fired.
    [[nodiscard]] uint64_t drag_alert_count() const noexcept {
        return drag_alerts_.load(std::memory_order_relaxed);
    }

    /// JSON diagnostics snapshot.
    [[nodiscard]] std::string to_stats_json() const;

    /// Reset all P&L and positions.
    void reset();

    /// Update configuration (resets state).
    void update_config(const Config& cfg);

private:
    mutable std::mutex mutex_;
    Config cfg_;

    double shadow_pos_val_{0.0};
    double live_pos_val_{0.0};
    double shadow_pnl_val_{0.0};
    double live_pnl_val_{0.0};
    double prev_price_{0.0};
    bool   price_seeded_{false};
    bool   drag_armed_{true};

    std::atomic<double>   shadow_pnl_{0.0};
    std::atomic<double>   live_pnl_{0.0};
    std::atomic<double>   shadow_pos_{0.0};
    std::atomic<double>   live_pos_{0.0};
    std::atomic<uint64_t> price_updates_{0};
    std::atomic<uint64_t> drag_alerts_{0};
};

} // namespace llmquant
