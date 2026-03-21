#pragma once

#include <atomic>
#include <chrono>
#include <cinttypes>
#include <cmath>
#include <cstdio>
#include <functional>
#include <mutex>
#include <optional>
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
     * @brief Callback invoked when a gate transitions from passing to blocking.
     *
     * @param gate_name Identifies the gate that tripped (e.g. "magnitude").
     * @param signal    The TradeSignal that caused the trip.
     */
    using GateTripCallback = std::function<void(const std::string& gate_name, const TradeSignal& signal)>;

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
    [[nodiscard]] bool evaluate(const TradeSignal& signal);

    /**
     * @brief Evaluate a signal and return the decision with a rejection reason.
     *
     * Identical logic to evaluate(), but also populates a human-readable
     * reason string on rejection.  The reason is empty on pass.
     *
     * Thread safety: individual calls are thread-safe.  Concurrent calls from
     * multiple threads are serialized internally so that the temporary callback
     * swap used to capture the rejection reason is never visible to another
     * caller.  Concurrent calls to evaluate() (the non-reason variant) proceed
     * without being blocked by this serialization.
     *
     * @param signal The TradeSignal to evaluate.
     * @param reason Output parameter: populated with the rejection reason on failure,
     *               cleared on pass.
     * @return true if the signal passes all checks; false if blocked.
     */
    [[nodiscard]] bool evaluate_with_reason(const TradeSignal& signal, std::string& reason);

    /**
     * @brief Evaluate a signal; return nullopt on pass or the rejection reason string on block.
     *
     * Convenience wrapper around evaluate_with_reason().  Callers that prefer
     * a single return value over an output parameter can use this overload:
     *
     * @code
     *   if (auto reason = rm.try_evaluate(sig)) {
     *       log_rejection(*reason);
     *   }
     * @endcode
     *
     * @param signal The TradeSignal to evaluate.
     * @return std::nullopt if the signal passes; a non-empty reason string if blocked.
     */
    [[nodiscard]] std::optional<std::string> try_evaluate(const TradeSignal& signal);

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
     * @brief Return the current net position value directly.
     *
     * Equivalent to get_position().net_position but avoids copying the entire
     * PositionState struct.  Thread-safe (acquires mutex_).
     *
     * @return Current net_position from the most recent update_position() call.
     */
    double get_net_exposure() const;

    /**
     * @brief Return true if the next signal would be blocked by the rate gate.
     *
     * Checks whether signals_in_window_ has reached max_signals_per_second
     * without advancing the window.  Thread-safe (acquires mutex_).
     *
     * @return true if the rate limit has been hit in the current window.
     */
    bool is_rate_limited() const;

    /**
     * @brief Return elapsed milliseconds since the start of the current rate-limit window.
     *
     * Thread-safe (acquires mutex_).
     *
     * @return Elapsed time in the current rate window in milliseconds.
     */
    double get_window_time_elapsed_ms() const;

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
     * @brief Return the fraction of the drawdown budget currently consumed, in [0.0, 1.0].
     *
     * Computed as |cumulative_bias| / max_drawdown.  Returns 0.0 if max_drawdown
     * is zero (to avoid division by zero).  Clamped to [0.0, 1.0] so values
     * are always safe to display on a dashboard gauge.
     *
     * Thread-safe (acquires mutex_).
     *
     * @return Drawdown utilization in [0.0, 1.0].
     */
    double get_drawdown_utilization() const;

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
     * @brief Return the total number of signals evaluated (passed + all blocked).
     *
     * Thread-safe (reads atomic stats).
     *
     * @return Total signals evaluated since construction or last reset_stats().
     */
    uint64_t get_total_evaluated() const noexcept;

    /**
     * @brief Return a single-line human-readable summary of risk statistics.
     *
     * Format: "evaluated=<n> passed=<n> blocked=<n> blocked_rate=<rate>
     *          mag=<n> conf=<n> rate=<n> dd=<n> pos=<n> healthy=<true|false>"
     *
     * Thread-safe (delegates to existing thread-safe accessors).
     *
     * @return Single-line stats summary string.
     */
    std::string format_stats() const;

    /**
     * @brief Return true if all risk gates are nominally healthy.
     *
     * Checks:
     *   - All gates are enabled (none are in bypass mode).
     *   - The drawdown budget remaining is > 0.
     *   - The net position is within limits.
     *   - The current PnL is above the PnL floor.
     *
     * Thread-safe (acquires mutex_).
     *
     * @return true if no gates are bypassed and all thresholds are met.
     */
    bool is_healthy() const;

    /**
     * @brief Convenience setter to update the position and PnL limits without
     *        replacing the entire PositionState.
     *
     * Equivalent to: get_position() + modify fields + update_position(), but
     * atomic and without the extra copy.  The net_position and pnl fields of the
     * stored PositionState are left unchanged.
     *
     * Thread-safe (acquires mutex_).
     *
     * @param position_limit Maximum allowed absolute position.
     * @param pnl_limit      Maximum tolerated loss (should be negative or zero).
     */
    void set_position_limit(double position_limit, double pnl_limit = -10.0);

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
     * @brief Snapshot of which risk gates are currently enabled.
     *
     * A gate is enabled when its corresponding disable_*_gate flag is false.
     * Disabled gates are bypassed and let every signal through.
     */
    struct GateStatus {
        bool magnitude_gate{true};   ///< true = gate is active (not bypassed).
        bool confidence_gate{true};
        bool rate_gate{true};
        bool drawdown_gate{true};
        bool position_gate{true};
    };

    /**
     * @brief Return the current enabled/disabled state of every risk gate.
     *
     * Useful for health dashboards, observability endpoints, and tests that
     * verify gate configuration changes have taken effect.
     *
     * Thread-safe (acquires mutex_).
     *
     * @return GateStatus snapshot.
     */
    GateStatus get_gate_status() const;

    /**
     * @brief Per-gate signal block counts since construction or last reset_stats().
     *
     * Each field holds the number of signals rejected by that specific gate.
     * Note: pnl_count is a subset of position_count (both are incremented on PnL breach).
     */
    struct BlockedByGate {
        uint64_t magnitude{0};
        uint64_t confidence{0};
        uint64_t rate{0};
        uint64_t drawdown{0};
        uint64_t position{0};
        uint64_t pnl{0};
    };

    /**
     * @brief Return the number of signals blocked by each individual risk gate.
     *
     * Provides finer-grained visibility than get_blocked_rate(), which only
     * returns an aggregate.  Reads atomic counters with relaxed ordering.
     *
     * @return BlockedByGate snapshot.
     */
    BlockedByGate get_blocked_by_gate() const noexcept;

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

    /**
     * @brief Immutable state snapshot for dashboards and health checks.
     */
    struct Snapshot {
        Config        config;
        PositionState position;
        double        cumulative_bias{0.0};
        double        drawdown_budget_remaining{0.0};
        double        rate_limit_utilization{0.0};
        uint64_t      signals_passed{0};
        uint64_t      signals_blocked_total{0};
    };

    /**
     * @brief Capture all current state atomically in a single mutex-held call.
     *
     * Returns a value-copy of config, position, and drawdown state alongside
     * the current stat counters.  Useful for structured health-check endpoints
     * and Prometheus gauge export where you need a consistent cross-field view.
     *
     * Thread-safe (acquires mutex_).
     *
     * @return Snapshot of all current RiskManager state.
     */
    Snapshot snapshot() const;

    /**
     * @brief Return the approved-signal rate in the current rate-limit window.
     *
     * Computes signals_in_window_ / elapsed_seconds since the window last reset.
     * Returns 0.0 if elapsed time is less than 1 millisecond (avoids
     * divide-by-near-zero on the very first call).
     *
     * Thread-safe (acquires mutex_).
     *
     * @return Approved signals per second in the current rate-limit window.
     */
    double get_signals_per_second() const;

    /**
     * @brief Evaluate a batch of signals in order, returning one result per signal.
     *
     * Equivalent to calling evaluate() for each element but cheaper when all
     * signals share the same lock acquisition pattern.  The returned vector has
     * the same size as @p signals; element i is true iff signal i passed all
     * enabled gates.
     *
     * Thread-safe (each evaluate() call acquires mutex_ independently).
     *
     * @param signals Ordered sequence of signals to evaluate.
     * @return Vector of bool; true = passed, false = blocked.
     */
    [[nodiscard]] std::vector<bool> evaluate_batch(const std::vector<TradeSignal>& signals);

    /**
     * @brief Return the fraction of the absolute position limit currently consumed.
     *
     * Computed as |net_position| / position_limit, clamped to [0.0, 1.0].
     * Returns 0.0 if position_limit is zero.
     *
     * Thread-safe (acquires mutex_).
     *
     * @return Position utilization in [0.0, 1.0].
     */
    double get_position_utilization() const {
        std::lock_guard<std::mutex> lock(mutex_);
        if (position_.position_limit <= 0.0) return 0.0;
        double util = std::fabs(position_.net_position) / position_.position_limit;
        return util > 1.0 ? 1.0 : util;
    }

    /**
     * @brief Return the fraction of evaluated signals that were blocked, in [0.0, 1.0].
     *
     * Computed as total_blocked / (signals_passed + total_blocked).
     * Returns 0.0 if no signals have been evaluated.
     *
     * Thread-safe (reads atomic counters).
     *
     * @return Rejection rate in [0.0, 1.0].
     */
    double get_rejection_rate() const noexcept;

    /**
     * @brief Return the total number of signals evaluated (passed + all blocked).
     *
     * Thread-safe (reads atomic counters with relaxed ordering).
     *
     * @return Total evaluation count since construction or last reset_stats().
     */
    uint64_t get_total_signals_evaluated() const noexcept;

    /**
     * @brief Return the fraction of evaluated signals that passed all gates, in [0.0, 1.0].
     *
     * Computed as 1.0 - get_rejection_rate(). Returns 1.0 if no signals have
     * been evaluated.
     *
     * Thread-safe (reads atomic counters with relaxed ordering).
     *
     * @return Pass rate in [0.0, 1.0].
     */
    double get_pass_rate() const noexcept {
        return 1.0 - get_rejection_rate();
    }

    /**
     * @brief Return the fraction of total evaluations blocked by the magnitude gate.
     *
     * Computed as signals_blocked_magnitude / total_evaluated.
     * Returns 0.0 if no signals have been evaluated.
     *
     * Thread-safe (reads atomic counters with relaxed ordering).
     *
     * @return Magnitude block rate in [0.0, 1.0].
     */
    double get_magnitude_block_rate() const noexcept {
        uint64_t total = get_total_signals_evaluated();
        if (total == 0) return 0.0;
        return static_cast<double>(stats_.signals_blocked_magnitude.load(std::memory_order_relaxed))
             / static_cast<double>(total);
    }

    /**
     * @brief Return the fraction of total evaluations blocked by the confidence gate.
     *
     * Computed as signals_blocked_confidence / total_evaluated.
     * Returns 0.0 if no signals have been evaluated.
     *
     * Thread-safe (reads atomic counters with relaxed ordering).
     *
     * @return Confidence block rate in [0.0, 1.0].
     */
    double get_confidence_block_rate() const noexcept {
        uint64_t total = get_total_signals_evaluated();
        if (total == 0) return 0.0;
        return static_cast<double>(stats_.signals_blocked_confidence.load(std::memory_order_relaxed))
             / static_cast<double>(total);
    }

    /**
     * @brief Return the name of the gate that has blocked the most signals.
     *
     * Compares per-gate block counters atomically and returns the name of the
     * gate with the highest count.  If no signals have been blocked yet, or if
     * all gates have equal zero counts, returns "none".
     *
     * Useful for operator dashboards and automated alerts: a single call
     * identifies which constraint is the most binding bottleneck.
     *
     * Thread-safe (reads atomic counters with relaxed ordering).
     *
     * @return One of "magnitude", "confidence", "rate", "drawdown",
     *         "position", or "none".
     */
    std::string get_most_blocked_gate() const noexcept {
        uint64_t mag  = stats_.signals_blocked_magnitude.load(std::memory_order_relaxed);
        uint64_t conf = stats_.signals_blocked_confidence.load(std::memory_order_relaxed);
        uint64_t rate = stats_.signals_blocked_rate.load(std::memory_order_relaxed);
        uint64_t dd   = stats_.signals_blocked_drawdown.load(std::memory_order_relaxed);
        uint64_t pos  = stats_.signals_blocked_position.load(std::memory_order_relaxed);

        uint64_t best = 0;
        std::string name = "none";
        auto update = [&](uint64_t v, const char* n) {
            if (v > best) { best = v; name = n; }
        };
        update(mag,  "magnitude");
        update(conf, "confidence");
        update(rate, "rate");
        update(dd,   "drawdown");
        update(pos,  "position");
        return name;
    }

    /**
     * @brief Serialise the current risk statistics snapshot to a JSON string.
     *
     * Returns a single JSON object with all blocked-gate counters,
     * signals_passed, aggregate blocked_rate_frac, is_healthy flag, and
     * most_blocked_gate label. Intended for structured logging and diagnostics.
     * Thread-safe (reads atomic counters with relaxed ordering).
     *
     * @return JSON object as std::string.
     */
    std::string to_stats_json() const noexcept {
        uint64_t mag    = stats_.signals_blocked_magnitude.load(std::memory_order_relaxed);
        uint64_t conf   = stats_.signals_blocked_confidence.load(std::memory_order_relaxed);
        uint64_t ratel  = stats_.signals_blocked_rate.load(std::memory_order_relaxed);
        uint64_t dd     = stats_.signals_blocked_drawdown.load(std::memory_order_relaxed);
        uint64_t pos    = stats_.signals_blocked_position.load(std::memory_order_relaxed);
        uint64_t pnl    = stats_.signals_blocked_pnl.load(std::memory_order_relaxed);
        uint64_t passed = stats_.signals_passed.load(std::memory_order_relaxed);
        uint64_t total_blocked = mag + conf + ratel + dd + pos;
        uint64_t total = passed + total_blocked;
        double bfrac = (total > 0) ? (static_cast<double>(total_blocked) / static_cast<double>(total)) : 0.0;
        char buf[512];
        std::snprintf(buf, sizeof(buf),
            "{\"signals_passed\":%" PRIu64
            ",\"blocked_magnitude\":%" PRIu64
            ",\"blocked_confidence\":%" PRIu64
            ",\"blocked_rate\":%" PRIu64
            ",\"blocked_drawdown\":%" PRIu64
            ",\"blocked_position\":%" PRIu64
            ",\"blocked_pnl\":%" PRIu64
            ",\"blocked_rate_frac\":%.6f"
            ",\"is_healthy\":%s"
            ",\"most_blocked_gate\":\"%s\"}",
            passed, mag, conf, ratel, dd, pos, pnl,
            bfrac,
            is_healthy() ? "true" : "false",
            get_most_blocked_gate().c_str());
        return buf;
    }

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
    /// Serialises concurrent calls to evaluate_with_reason() to prevent
    /// the temporary callback-swap from racing between callers.
    mutable std::mutex ewr_mutex_;

    // Rate limiting.
    std::chrono::high_resolution_clock::time_point rate_window_start_;
    size_t signals_in_window_{0};

    // Drawdown tracking.
    std::chrono::high_resolution_clock::time_point drawdown_window_start_;
    double cumulative_bias_{0.0};

    // Per-gate trip-wire callbacks (pass→block edge trigger). Access via mutex_.
    GateTripCallback gate_trip_magnitude_cb_;
    GateTripCallback gate_trip_confidence_cb_;
    GateTripCallback gate_trip_rate_cb_;
    GateTripCallback gate_trip_drawdown_cb_;
    GateTripCallback gate_trip_position_cb_;
    bool gate_magnitude_last_blocked_{false};
    bool gate_confidence_last_blocked_{false};
    bool gate_rate_last_blocked_{false};
    bool gate_drawdown_last_blocked_{false};
    bool gate_position_last_blocked_{false};

    Stats stats_;
};

} // namespace llmquant
