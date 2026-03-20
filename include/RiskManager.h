#pragma once

#include <atomic>
#include <chrono>
#include <cmath>
#include <functional>
#include <mutex>
#include <string>
#include <vector>
#include "TradeSignalEngine.h"

namespace llmquant { class MetricsLogger; }

namespace llmquant {

/**
 * @brief Production risk management layer that gates TradeSignals before emission.
 *
 * Enforces position limits, drawdown guards, signal magnitude caps, and a
 * per-second signal rate limit. Signals that breach any threshold are
 * suppressed and counted; breaches are surfaced via an optional alert callback.
 *
 * Thread safety: all public methods are safe to call concurrently.
 */
class RiskManager {
public:
    /**
     * @brief Construction-time risk parameters.
     */
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

        /// When true, the magnitude gate is bypassed (for testing only).
        bool disable_magnitude_gate{false};
        /// When true, the confidence gate is bypassed (for testing only).
        bool disable_confidence_gate{false};
        /// When true, the rate-limit gate is bypassed (for testing only).
        bool disable_rate_gate{false};
        /// When true, the drawdown gate is bypassed (for testing only).
        bool disable_drawdown_gate{false};
        /// When true, the position/PnL gate is bypassed (for testing only).
        bool disable_position_gate{false};
    };

    /**
     * @brief Current position state reported to the risk manager by the OMS.
     */
    struct PositionState {
        double net_position{0.0};    ///< Current net position (positive = long, negative = short).
        double position_limit{1.0};  ///< Maximum allowed absolute position.
        double pnl{0.0};             ///< Current unrealised PnL.
        double pnl_limit{-10.0};     ///< Maximum tolerated loss (negative number).
    };

    /**
     * @brief OMS notification callback fired when position limits are approached or breached.
     *
     * @param event  Human-readable event name (e.g. "position_limit_approach").
     * @param state  Current position snapshot at the time of the event.
     * @param signal The TradeSignal that triggered the check.
     */
    using OmsCallback = std::function<void(const std::string& event,
                                           const PositionState& state,
                                           const TradeSignal& signal)>;

    /**
     * @brief Live statistics updated atomically by the risk manager.
     */
    struct Stats {
        std::atomic<uint64_t> signals_passed{0};
        std::atomic<uint64_t> signals_blocked_magnitude{0};
        std::atomic<uint64_t> signals_blocked_confidence{0};
        std::atomic<uint64_t> signals_blocked_rate{0};
        std::atomic<uint64_t> signals_blocked_drawdown{0};
        std::atomic<uint64_t> signals_blocked_position{0};
        std::atomic<uint64_t> signals_blocked_pnl{0};

        /**
         * @brief Sum of all blocked counters excluding the PnL sub-counter.
         *
         * signals_blocked_pnl is a strict subset of signals_blocked_position
         * (both are incremented on PnL breach).  Excluding it prevents
         * double-counting when a caller simply needs the total blocked signal
         * count for rate/ratio calculations.
         *
         * @return Total number of blocked signals since construction or last reset_stats().
         */
        uint64_t blocked_total() const noexcept {
            return signals_blocked_magnitude.load(std::memory_order_relaxed)
                 + signals_blocked_confidence.load(std::memory_order_relaxed)
                 + signals_blocked_rate.load(std::memory_order_relaxed)
                 + signals_blocked_drawdown.load(std::memory_order_relaxed)
                 + signals_blocked_position.load(std::memory_order_relaxed);
        }
    };

    /**
     * @brief Alert callback type: invoked synchronously when a signal is blocked.
     *
     * @param reason Human-readable rejection reason.
     * @param signal The rejected TradeSignal.
     */
    using AlertCallback = std::function<void(const std::string& reason, const TradeSignal&)>;

    /**
     * @brief Construct a RiskManager with the given parameters.
     *
     * @param config Risk thresholds and window configuration.
     */
    explicit RiskManager(const Config& config);

    /**
     * @brief Evaluate a signal against all risk rules.
     *
     * NOTE: evaluate() holds an internal mutex for its entire execution,
     * including any alert_cb_ invocations on rejected signal paths.
     * Alert callbacks MUST NOT call evaluate(), get_position(), or any other
     * RiskManager method, or a deadlock will result.  For complex alert handling
     * (e.g. network I/O), dispatch alerts to a separate queue from the callback.
     *
     * @param signal The TradeSignal to evaluate.
     * @return true if the signal passes all checks and should be emitted;
     *         false if the signal is blocked (stats updated, alert fired).
     */
    bool evaluate(const TradeSignal& signal);

    /**
     * @brief Evaluate a signal and return the decision with a rejection reason.
     *
     * Identical logic to evaluate(), but also populates a human-readable
     * reason string on rejection.  The reason is empty on pass.
     *
     * @param signal The TradeSignal to evaluate.
     * @param reason Output parameter: populated with the rejection reason on failure,
     *               cleared on pass.
     * @return true if the signal passes all checks; false if blocked.
     */
    bool evaluate_with_reason(const TradeSignal& signal, std::string& reason);

    /**
     * @brief Register a callback to be invoked when a signal is blocked.
     *
     * @warning Callbacks are invoked OUTSIDE the evaluate() mutex. Callbacks
     *          MUST NOT call any RiskManager methods or a deadlock will occur.
     *          Callbacks MUST NOT block for extended periods.
     *
     * @param cb Callable matching AlertCallback; stored by value.
     */
    void set_alert_callback(AlertCallback cb);

    /**
     * @brief Update the current position state from the OMS.
     *
     * Thread-safe. Called by the OMS adapter on each fill or position update.
     *
     * @param state Latest position snapshot from the order management system.
     */
    void update_position(const PositionState& state);

    /**
     * @brief Register a callback for OMS events (limit-approach, limit-breach, pnl-alert).
     *
     * @warning Callbacks are invoked OUTSIDE the evaluate() mutex. Callbacks
     *          MUST NOT call any RiskManager methods or a deadlock will occur.
     *          Callbacks MUST NOT block for extended periods.
     *
     * @param cb Callable matching OmsCallback; stored by value.
     */
    void set_oms_callback(OmsCallback cb);

    /**
     * @brief Return the most recently reported position state.
     *
     * Thread-safe (acquires mutex_).
     *
     * @return Copy of the most recently reported PositionState.
     */
    PositionState get_position() const;

    /**
     * @brief Return the current cumulative drawdown bias accumulator value.
     *
     * Thread-safe (acquires mutex_).  The value resets to 0 when the drawdown
     * window elapses or when reset() is called.
     *
     * @return Signed cumulative bias since the last window reset.
     */
    double get_cumulative_bias() const;

    /**
     * @brief Return the remaining drawdown budget in the current window.
     *
     * Computes max_drawdown - |cumulative_bias_|, clamped to [0, max_drawdown].
     * A value of 0.0 means the drawdown gate will block the next signal.
     *
     * Thread-safe (acquires mutex_).
     *
     * @return Remaining drawdown headroom in the same units as max_drawdown.
     */
    double get_drawdown_budget_remaining() const;

    /**
     * @brief Return the fraction of the per-second rate limit consumed in the
     *        current window, in [0.0, 1.0].
     *
     * 0.0 = no signals in this window; 1.0 = at or above the limit.
     * Thread-safe (acquires mutex_).
     *
     * @return Rate-limit utilization fraction.
     */
    double get_rate_limit_utilization() const;

    /**
     * @brief Return the fraction of evaluated signals that were blocked by any
     *        gate, in [0.0, 1.0].
     *
     * Computed as total_blocked / (total_blocked + signals_passed).
     * Returns 0.0 if no signals have been evaluated yet.
     * Thread-safe (reads atomic stats without acquiring mutex_).
     *
     * @return Block rate in [0.0, 1.0].
     */
    double get_blocked_rate() const noexcept;

    /**
     * @brief Attach a MetricsLogger for structured rejection logging.
     *
     * @param logger Pointer to an active MetricsLogger; must outlive this
     *               RiskManager. Pass nullptr to disable (default).
     */
    void set_metrics_logger(MetricsLogger* logger);

    /**
     * @brief Reset the drawdown accumulator and rate-limit window.
     */
    void reset();

    /**
     * @brief Reset only the drawdown accumulator and its window start time.
     *
     * Lighter-weight alternative to reset() that does not affect the rate-limit
     * window or any signal counters.  Useful at session boundaries when you want
     * a fresh drawdown budget without discarding rate-limiting history.
     *
     * Thread-safe (acquires mutex_).
     */
    void reset_drawdown();

    /**
     * @brief Reset all signal-counter statistics to zero.
     *
     * Thread-safe (each counter is an atomic).  Does NOT reset the drawdown
     * or rate-limit accumulators — call reset() for those.  Useful between
     * trading sessions when fresh counters are needed without a full restart.
     */
    void reset_stats() noexcept;

    /**
     * @brief Disable all risk gate checks simultaneously (for testing / circuit-break bypass).
     *
     * Sets all disable_*_gate flags in config_ to true in a single mutex-held
     * operation.  Equivalent to calling update_config() with every gate flag set,
     * but without needing to construct a full Config.
     *
     * Thread-safe (acquires mutex_).
     */
    void disable_all_gates();

    /**
     * @brief Re-enable all risk gate checks simultaneously.
     *
     * Clears all disable_*_gate flags in a single mutex-held operation.
     *
     * Thread-safe (acquires mutex_).
     */
    void enable_all_gates();

    /**
     * @brief Atomically replace the risk threshold configuration.
     *
     * Safe to call from any thread; takes the internal mutex.  Gate disable
     * flags (disable_*_gate) are also updated.  Does not reset existing
     * drawdown/rate-limit accumulators — call reset() first if that is desired.
     *
     * @param config New threshold configuration to apply.
     */
    void update_config(const Config& config);

    /**
     * @brief Return a copy of the current risk configuration.
     *
     * Thread-safe (acquires mutex_).
     *
     * @return Copy of the current Config struct.
     */
    Config get_config() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return config_;
    }

    /**
     * @brief Return a read-only reference to live statistics.
     *
     * @return Const reference to the internal Stats struct.
     */
    const Stats& get_stats() const noexcept { return stats_; }

private:
    bool check_magnitude(const TradeSignal& signal);
    bool check_confidence(const TradeSignal& signal);
    bool check_rate_limit();
    bool check_drawdown(const TradeSignal& signal);
    void update_drawdown(const TradeSignal& signal);

    Config        config_;
    AlertCallback alert_cb_;
    OmsCallback   oms_cb_;
    MetricsLogger* logger_{nullptr};
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
