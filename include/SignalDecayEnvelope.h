#pragma once

#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <functional>
#include <mutex>
#include <string>

namespace llmquant {

/**
 * @brief Exponential time-decay envelope on accumulated trade signal bias.
 *
 * In a token-stream trading engine, new bullish tokens continuously reinforce
 * a positive bias.  But if the stream goes quiet — or shifts topic — the
 * accumulated bias should fade rather than persist indefinitely.
 *
 * SignalDecayEnvelope wraps the raw bias accumulator and applies:
 *
 *   decayed_bias(t) = raw_bias * exp(-λ * Δt)
 *
 * where:
 *   λ = ln(2) / half_life_ms   (so the bias halves every `half_life_ms`)
 *   Δt = milliseconds since last reinforcement
 *
 * ## Reinforcement
 * `reinforce(delta)` adds `delta` to the raw bias and resets the decay clock,
 * representing a new token that strengthens the current view.
 *
 * ## Usage
 * ```cpp
 * SignalDecayEnvelope envelope;
 * // In process_token lambda:
 * envelope.reinforce(signal.delta_bias_shift);
 * // At signal emit time:
 * double effective_bias = envelope.decayed_bias();
 * signal.delta_bias_shift = effective_bias;
 * ```
 *
 * ## Thread safety
 * All methods are thread-safe (atomic clock + mutex for bias).
 *
 * ## Feature flag
 * Controlled by `LLMQUANT_ENABLE_SIGNAL_DECAY` (default ON).
 */
class SignalDecayEnvelope {
public:
    /**
     * @brief Configuration.
     */
    struct Config {
        /**
         * @brief Half-life of the bias decay in milliseconds.
         *
         * After this many milliseconds of silence, the bias halves.
         * Default: 30 000 ms (30 seconds).
         */
        double half_life_ms{30'000.0};

        /**
         * @brief If true, clamp the decayed bias to [min_bias, max_bias].
         */
        bool clamp{true};

        /**
         * @brief Minimum decayed bias (after decay).  Default -1.0.
         */
        double min_bias{-1.0};

        /**
         * @brief Maximum decayed bias (after decay).  Default +1.0.
         */
        double max_bias{ 1.0};
    };

    using DecayCallback = std::function<void(double old_bias, double new_bias)>;

    explicit SignalDecayEnvelope(Config cfg = Config{});

    // Non-copyable.
    SignalDecayEnvelope(const SignalDecayEnvelope&)            = delete;
    SignalDecayEnvelope& operator=(const SignalDecayEnvelope&) = delete;

    /**
     * @brief Add `delta` to the raw bias and restart the decay clock.
     *
     * Thread-safe.
     *
     * @param delta Signed bias delta from the most recent token.
     */
    void reinforce(double delta);

    /**
     * @brief Return the current bias after applying exponential decay.
     *
     * Thread-safe.
     */
    [[nodiscard]] double decayed_bias() const;

    /**
     * @brief Return the raw (un-decayed) bias accumulator value.
     *
     * Thread-safe.
     */
    [[nodiscard]] double raw_bias() const;

    /**
     * @brief Return milliseconds elapsed since the last reinforce() call.
     *
     * Thread-safe.
     */
    [[nodiscard]] int64_t ms_since_last_reinforce() const;

    /**
     * @brief Total reinforcements (reinforce() calls).
     */
    [[nodiscard]] uint64_t total_reinforcements() const noexcept {
        return total_reinforcements_.load(std::memory_order_relaxed);
    }

    /**
     * @brief Reset bias and clock to zero.
     *
     * Thread-safe.
     */
    void reset();

    /**
     * @brief Register a callback fired when the decayed bias crosses zero
     *        (sign change), indicating a regime flip in perceived direction.
     *
     * Fired from the thread calling decayed_bias() or reinforce().
     * Thread-safe (callback registered under mutex).
     */
    void set_zero_cross_callback(DecayCallback cb);

    /**
     * @brief Update configuration at runtime.
     *
     * Resets the envelope.
     */
    void update_config(const Config& cfg);

    /**
     * @brief Serialise current state to a JSON object string.
     */
    [[nodiscard]] std::string to_stats_json() const;

private:
    [[nodiscard]] double compute_decayed(double raw, int64_t age_ms) const;

    Config cfg_;
    mutable std::mutex mtx_;

    double          raw_bias_{0.0};
    mutable double  last_reported_sign_{0.0};  // tracks zero-cross; mutable because updated in const decayed_bias()

    using Clock = std::chrono::steady_clock;
    Clock::time_point last_reinforce_time_{Clock::now()};

    std::atomic<uint64_t> total_reinforcements_{0};

    DecayCallback zero_cross_cb_;
};

} // namespace llmquant
