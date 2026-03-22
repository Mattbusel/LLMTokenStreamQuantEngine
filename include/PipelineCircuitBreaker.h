#pragma once

#include <atomic>
#include <chrono>
#include <cstdint>
#include <functional>
#include <mutex>
#include <string>

namespace llmquant {

/**
 * @brief Three-state circuit breaker that monitors the signal block rate.
 *
 * Implements a standard CLOSED → OPEN → HALF-OPEN → CLOSED state machine:
 *
 *   CLOSED:    Normal operation.  Every signal evaluation is tracked.
 *              Transitions to OPEN when the rolling block_rate exceeds
 *              open_threshold for open_duration_s consecutive seconds.
 *
 *   OPEN:      The pipeline is "tripped".  is_open() returns true so callers
 *              can suppress further signal emission.  The circuit remains
 *              open for at least reset_timeout_s, then transitions to
 *              HALF-OPEN.
 *
 *   HALF-OPEN: One probe window is permitted.  If the block_rate falls
 *              below close_threshold during the probe window the circuit
 *              resets to CLOSED; otherwise it reopens.
 *
 * Thread safety: record_signal() and all query methods are safe to call
 * concurrently from multiple threads.
 *
 * Compile-time feature flag: LLMQUANT_CIRCUIT_BREAKER_ENABLED
 * (controlled by CMake option LLMQUANT_ENABLE_CIRCUIT_BREAKER).
 */
class PipelineCircuitBreaker {
public:
    /**
     * @brief Circuit-breaker tuning parameters.
     */
    struct Config {
        /**
         * @brief Block rate [0,1] above which the circuit trips to OPEN.
         *
         * Measured as a rolling EMA over the sliding window.
         * Default 0.90 = trip when 90% of signals are blocked.
         */
        double open_threshold{0.90};

        /**
         * @brief Block rate [0,1] below which the circuit resets to CLOSED.
         *
         * Must be <= open_threshold.  Hysteresis prevents rapid on/off cycling.
         * Default 0.50.
         */
        double close_threshold{0.50};

        /**
         * @brief Seconds the block rate must stay above open_threshold before tripping.
         *
         * Prevents transient spikes from false-tripping the circuit.
         * Default 30 s.
         */
        int open_duration_s{30};

        /**
         * @brief Minimum seconds the circuit stays open before probing recovery.
         *
         * Default 60 s.
         */
        int reset_timeout_s{60};

        /**
         * @brief EMA smoothing factor α for the rolling block rate.
         *
         * α = 1.0 means no smoothing (last window only); 0.1 means heavy smoothing.
         * Default 0.2.
         */
        double ema_alpha{0.2};

        /**
         * @brief When true, entering OPEN state suppresses signal emission.
         *
         * When false, the circuit still transitions states and fires the callback,
         * but callers are responsible for checking is_open().
         * Default true.
         */
        bool suppress_when_open{true};
    };

    /**
     * @brief Circuit-breaker state.
     */
    enum class State { Closed, Open, HalfOpen };

    /**
     * @brief Callback invoked on state transitions.
     *
     * @param new_state  The state being entered.
     * @param block_rate The current rolling block rate that triggered the change.
     */
    using StateChangeCallback =
        std::function<void(State new_state, double block_rate)>;

    /**
     * @brief Construct the circuit breaker with the given configuration.
     *
     * @param config Threshold, timeout, and EMA parameters.
     */
    explicit PipelineCircuitBreaker(Config config = Config{});

    /**
     * @brief Record one signal evaluation outcome.
     *
     * Updates the internal EMA and possibly transitions state.
     * Called once per signal from the hot path.
     *
     * @param blocked true if the signal was blocked by a risk gate; false if passed.
     */
    void record_signal(bool blocked);

    /**
     * @brief Returns true when the circuit is in OPEN state.
     *
     * Callers on the hot path should gate signal emission on this flag.
     *
     * @return true if signals should be suppressed.
     */
    [[nodiscard]] bool is_open() const noexcept {
        return state_.load(std::memory_order_acquire) == State::Open;
    }

    /**
     * @brief Returns true when the circuit is in HALF-OPEN (probe) state.
     *
     * @return true if the circuit is in HALF-OPEN state.
     */
    [[nodiscard]] bool is_half_open() const noexcept {
        return state_.load(std::memory_order_acquire) == State::HalfOpen;
    }

    /**
     * @brief Returns the current circuit state.
     *
     * @return CLOSED, OPEN, or HALF-OPEN.
     */
    [[nodiscard]] State state() const noexcept {
        return state_.load(std::memory_order_acquire);
    }

    /**
     * @brief Returns a human-readable state name.
     *
     * @return "closed", "open", or "half_open".
     */
    [[nodiscard]] std::string state_name() const noexcept;

    /**
     * @brief Current rolling EMA block rate [0,1].
     *
     * @return EMA-smoothed fraction of signals blocked.
     */
    [[nodiscard]] double block_rate() const noexcept {
        return ema_block_rate_.load(std::memory_order_relaxed);
    }

    /**
     * @brief Number of times the circuit has tripped to OPEN since construction.
     *
     * @return Trip count.
     */
    [[nodiscard]] uint64_t trips() const noexcept {
        return trips_.load(std::memory_order_relaxed);
    }

    /**
     * @brief Number of times the circuit has recovered to CLOSED since construction.
     *
     * @return Recovery count.
     */
    [[nodiscard]] uint64_t recoveries() const noexcept {
        return recoveries_.load(std::memory_order_relaxed);
    }

    /**
     * @brief Register a callback invoked on every state transition.
     *
     * @param cb Callable matching StateChangeCallback; stored by value.
     */
    void set_state_change_callback(StateChangeCallback cb);

    /**
     * @brief Manually force the circuit to CLOSED state.
     *
     * Useful for operator recovery without waiting for reset_timeout_s.
     */
    void force_close();

    /**
     * @brief Manually force the circuit to OPEN state.
     *
     * Useful for external triggers (e.g. LatencyBudgetEnforcer Breaker tier)
     * that need to trip the pipeline independently of the block-rate heuristic.
     */
    void force_open();

    /**
     * @brief Reset all statistics (trips, recoveries, EMA) to zero.
     *
     * Does NOT change the current state.
     */
    void reset_stats();

private:
    void transition_to(State new_state);
    void maybe_transition();

    Config config_;

    // Hot-path state (all atomics, no mutex on the read path).
    std::atomic<State>  state_{State::Closed};
    std::atomic<double> ema_block_rate_{0.0};
    std::atomic<uint64_t> trips_{0};
    std::atomic<uint64_t> recoveries_{0};

    // Timing for sustain and reset windows.  Protected by mutex_.
    mutable std::mutex mutex_;
    std::chrono::steady_clock::time_point high_rate_since_{};
    std::chrono::steady_clock::time_point opened_at_{};
    bool high_rate_timer_running_{false};

    StateChangeCallback state_change_cb_;
};

} // namespace llmquant
