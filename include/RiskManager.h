#pragma once

#include <atomic>
#include <chrono>
#include <cmath>
#include <functional>
#include <mutex>
#include <string>
#include <vector>
#include "TradeSignalEngine.h"

namespace llmquant {

/// Production risk management layer that gates TradeSignals before emission.
///
/// Enforces position limits, drawdown guards, signal magnitude caps, and a
/// per-second signal rate limit. Signals that breach any threshold are
/// suppressed and counted; breaches are surfaced via an optional alert callback.
///
/// Thread safety: all public methods are safe to call concurrently.
class RiskManager {
public:
    /// Construction-time risk parameters.
    struct Config {
        /// Maximum absolute value of delta_bias_shift in a single signal.
        double max_bias_magnitude{1.0};

        /// Maximum absolute value of volatility_adjustment in a single signal.
        double max_volatility_magnitude{1.0};

        /// Maximum absolute value of spread_modifier in a single signal.
        double max_spread_magnitude{0.5};

        /// Minimum confidence required to emit a signal (0.0–1.0).
        double min_confidence{0.1};

        /// Maximum number of signals allowed per second (rate limit).
        size_t max_signals_per_second{100};

        /// Cumulative bias drawdown limit: if |sum of bias shifts| exceeds
        /// this value within the drawdown window, signals are halted.
        double max_drawdown{5.0};

        /// Duration over which drawdown is measured before resetting.
        std::chrono::seconds drawdown_window{60};

        /// Fraction of position_limit at which a limit-approach warning is fired
        /// (e.g. 0.8 = fire callback when |projected_position| > 80% of limit).
        double position_warn_fraction{0.8};
    };

    /// Current position state reported to the risk manager by the OMS.
    struct PositionState {
        double net_position{0.0};    ///< Current net position (positive = long, negative = short).
        double position_limit{1.0};  ///< Maximum allowed absolute position.
        double pnl{0.0};             ///< Current unrealised PnL.
        double pnl_limit{-10.0};     ///< Maximum tolerated loss (negative number).
    };

    /// OMS notification callback: fired when position limits are approached or breached.
    using OmsCallback = std::function<void(const std::string& event,
                                           const PositionState& state,
                                           const TradeSignal& signal)>;

    /// Live statistics updated by the risk manager.
    struct Stats {
        std::atomic<uint64_t> signals_passed{0};
        std::atomic<uint64_t> signals_blocked_magnitude{0};
        std::atomic<uint64_t> signals_blocked_confidence{0};
        std::atomic<uint64_t> signals_blocked_rate{0};
        std::atomic<uint64_t> signals_blocked_drawdown{0};
        std::atomic<uint64_t> signals_blocked_position{0};
    };

    /// Alert callback type: invoked synchronously when a signal is blocked.
    using AlertCallback = std::function<void(const std::string& reason, const TradeSignal&)>;

    /// Construct a RiskManager with the given parameters.
    explicit RiskManager(const Config& config);

    /// Evaluate a signal against all risk rules.
    ///
    /// # Returns
    /// `true` if the signal passes all checks and should be emitted.
    /// `false` if the signal is blocked (stats updated, alert fired).
    bool evaluate(const TradeSignal& signal);

    /// Register a callback to be invoked when a signal is blocked.
    ///
    /// # Arguments
    /// * `cb` — Callable matching AlertCallback; stored by value.
    void set_alert_callback(AlertCallback cb);

    /// Update the current position state from the OMS.
    ///
    /// Thread-safe. Called by the OMS adapter on each fill or position update.
    ///
    /// # Arguments
    /// * `state` — Latest position snapshot from the order management system.
    void update_position(const PositionState& state);

    /// Register a callback for OMS events (limit-approach, limit-breach, pnl-alert).
    ///
    /// # Arguments
    /// * `cb` — Callable matching OmsCallback; stored by value.
    void set_oms_callback(OmsCallback cb);

    /// Return the most recently reported position state.
    ///
    /// Thread-safe (acquires mutex_).
    PositionState get_position() const;

    /// Reset the drawdown accumulator and rate-limit window.
    void reset();

    /// Return a read-only reference to live statistics.
    const Stats& get_stats() const { return stats_; }

private:
    bool check_magnitude(const TradeSignal& signal);
    bool check_confidence(const TradeSignal& signal);
    bool check_rate_limit();
    bool check_drawdown(const TradeSignal& signal);
    void update_drawdown(const TradeSignal& signal);
    void fire_alert(const std::string& reason, const TradeSignal& signal);

    /// Check position limits and fire OMS callbacks if thresholds are crossed.
    /// Must be called with mutex_ held.
    bool check_and_notify_position(const TradeSignal& signal);

    Config        config_;
    AlertCallback alert_cb_;
    OmsCallback   oms_cb_;
    PositionState position_;
    mutable std::mutex mutex_;

    // Rate limiting.
    std::chrono::high_resolution_clock::time_point rate_window_start_;
    size_t signals_in_window_{0};

    // Drawdown tracking.
    std::chrono::high_resolution_clock::time_point drawdown_window_start_;
    double cumulative_bias_{0.0};

    Stats stats_;
};

} // namespace llmquant
