#pragma once

#include <atomic>
#include <cstdint>
#include <functional>
#include <string>

namespace llmquant {

/**
 * @brief Tiered SLA enforcement based on measured p99 latency.
 *
 * Unlike a single-threshold circuit breaker, the LatencyBudgetEnforcer
 * implements a four-tier escalation ladder:
 *
 *   Normal → Warn → Throttle → Drop → Breaker
 *
 * Each tier fires a distinct callback and returns a distinct Action.
 * The caller decides what to do with the action (log, shed load, open
 * a circuit breaker, etc.).
 *
 * ## Hysteresis
 * Recovery requires latency to drop below `tier * recovery_factor`
 * (default 0.8) for at least one check before de-escalating.
 *
 * ## Usage
 * ```cpp
 * LatencyBudgetEnforcer enforcer;
 * enforcer.set_warn_callback([](int64_t p99) {
 *     spdlog::warn("[sla] p99={}µs — warn budget exceeded", p99);
 * });
 * // In monitoring loop:
 * auto action = enforcer.check(latency_ctrl.p99_us());
 * if (action == LatencyBudgetEnforcer::Action::Drop)
 *     skip_signal = true;
 * ```
 *
 * ## Thread safety
 * `check()` is thread-safe.  Callbacks are invoked from the calling thread.
 *
 * ## Feature flag
 * Controlled by `LLMQUANT_ENABLE_LATENCY_ENFORCER` (default ON).
 */
class LatencyBudgetEnforcer {
public:
    /** @brief Escalation tier returned by check(). */
    enum class Action {
        Normal,   ///< Latency within budget.
        Warn,     ///< Approaching budget — log.
        Throttle, ///< Over soft budget — slow down token intake.
        Drop,     ///< Over hard budget — drop this signal.
        Breaker   ///< Over critical budget — open circuit breaker.
    };

    /** @brief Tier transition callback type. */
    using TierCallback = std::function<void(int64_t p99_us)>;

    /**
     * @brief Configuration.
     */
    struct Config {
        /** @brief p99 µs at which to enter Warn tier.    Default  5 µs. */
        int64_t warn_us{5};
        /** @brief p99 µs at which to enter Throttle tier. Default  8 µs. */
        int64_t throttle_us{8};
        /** @brief p99 µs at which to enter Drop tier.     Default 12 µs. */
        int64_t drop_us{12};
        /** @brief p99 µs at which to enter Breaker tier.  Default 20 µs. */
        int64_t breaker_us{20};

        /**
         * @brief Fraction of the tier threshold below which the state
         * de-escalates.  E.g. 0.8 means latency must drop to 80% of
         * the threshold before recovering.  Default 0.8.
         */
        double recovery_factor{0.8};
    };

    explicit LatencyBudgetEnforcer(Config cfg = Config{}) : cfg_(cfg) {}

    // Non-copyable.
    LatencyBudgetEnforcer(const LatencyBudgetEnforcer&)            = delete;
    LatencyBudgetEnforcer& operator=(const LatencyBudgetEnforcer&) = delete;

    /**
     * @brief Evaluate the current p99 latency and return the action tier.
     *
     * Fires the appropriate callback if the tier has changed.
     * Thread-safe.
     *
     * @param p99_us  Current p99 latency in microseconds.
     * @return Recommended action for this check interval.
     */
    Action check(int64_t p99_us);

    /**
     * @brief Force a reset to Normal regardless of current latency.
     */
    void reset();

    /** @brief Current action tier. */
    [[nodiscard]] Action current_action() const noexcept {
        return static_cast<Action>(current_tier_.load(std::memory_order_relaxed));
    }

    /** @brief Number of times the tier escalated upward. */
    [[nodiscard]] uint64_t escalation_count() const noexcept {
        return escalation_count_.load(std::memory_order_relaxed);
    }

    /** @brief Number of times the tier de-escalated (recovered). */
    [[nodiscard]] uint64_t recovery_count() const noexcept {
        return recovery_count_.load(std::memory_order_relaxed);
    }

    /** @brief Human-readable name for an Action tier. */
    [[nodiscard]] static const char* action_name(Action a) noexcept;

    // ----------------------------------------------------------------
    // Callback registration
    // ----------------------------------------------------------------

    /** @brief Called once when entering Warn tier. */
    void set_warn_callback(TierCallback cb)     { warn_cb_     = std::move(cb); }
    /** @brief Called once when entering Throttle tier. */
    void set_throttle_callback(TierCallback cb) { throttle_cb_ = std::move(cb); }
    /** @brief Called once when entering Drop tier. */
    void set_drop_callback(TierCallback cb)     { drop_cb_     = std::move(cb); }
    /** @brief Called once when entering Breaker tier. */
    void set_breaker_callback(TierCallback cb)  { breaker_cb_  = std::move(cb); }
    /** @brief Called when recovering to Normal. */
    void set_recovery_callback(TierCallback cb) { recovery_cb_ = std::move(cb); }

    /**
     * @brief Update configuration at runtime.
     *
     * Resets state to Normal.
     */
    void update_config(const Config& cfg) {
        cfg_ = cfg;
        reset();
    }

    /**
     * @brief Serialise current state to a JSON object string.
     */
    [[nodiscard]] std::string to_stats_json() const;

private:
    Config cfg_;

    std::atomic<int>      current_tier_{0};  // Action enum value
    std::atomic<uint64_t> escalation_count_{0};
    std::atomic<uint64_t> recovery_count_{0};

    TierCallback warn_cb_;
    TierCallback throttle_cb_;
    TierCallback drop_cb_;
    TierCallback breaker_cb_;
    TierCallback recovery_cb_;

    mutable std::mutex mtx_;  // protects cfg_ and callback dispatch
};

} // namespace llmquant
