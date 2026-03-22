#pragma once

#include <atomic>
#include <chrono>
#include <cstdint>
#include <functional>
#include <mutex>
#include <string>

namespace llmquant {

/**
 * @brief Token-bucket rate limiter for trade signal emission.
 *
 * Prevents OMS flooding during rapid LLM regime shifts by enforcing a
 * configurable maximum throughput with burst tolerance.
 *
 * ## Algorithm
 * A virtual bucket holds up to `capacity` tokens.  Tokens refill at
 * `refill_rate` per second.  Each call to `try_acquire()` consumes one
 * token.  If the bucket is empty, the signal is dropped (caller should
 * discard it) and `dropped_count` increments.
 *
 * ## Typical configuration
 * | `capacity` | `refill_rate` | Meaning                           |
 * |------------|---------------|-----------------------------------|
 * | 10         | 2.0           | Burst up to 10 signals; 2/s steady|
 * | 1          | 0.1           | At most 1 signal every 10 s       |
 *
 * ## Usage
 * ```cpp
 * SignalThrottle throttle;
 * // In signal callback:
 * if (!throttle.try_acquire()) return;  // drop over-rate signal
 * oms.submit(signal);
 * ```
 *
 * ## Thread safety
 * `try_acquire()` is lock-free for the common fast path.
 *
 * ## Feature flag
 * Controlled by `LLMQUANT_ENABLE_SIGNAL_THROTTLE` (default ON).
 */
class SignalThrottle {
public:
    /**
     * @brief Configuration.
     */
    struct Config {
        /**
         * @brief Maximum burst capacity (number of signals that can be
         *        emitted immediately after a quiet period).  Default 10.
         */
        double capacity{10.0};

        /**
         * @brief Token refill rate (signals per second).  Default 2.0.
         */
        double refill_rate{2.0};

        /**
         * @brief Initial token count.  Defaults to full capacity.
         * Set to 0 to start with an empty bucket (cold start).
         */
        double initial_tokens{-1.0};  // -1 = use capacity
    };

    using DropCallback = std::function<void()>;

    explicit SignalThrottle(Config cfg = Config{});

    // Non-copyable.
    SignalThrottle(const SignalThrottle&)            = delete;
    SignalThrottle& operator=(const SignalThrottle&) = delete;

    /**
     * @brief Attempt to consume one token.
     *
     * Returns true if a token was available (signal may proceed).
     * Returns false if the bucket is empty (signal should be dropped).
     *
     * Thread-safe.
     */
    [[nodiscard]] bool try_acquire();

    /**
     * @brief Total calls to try_acquire().
     */
    [[nodiscard]] uint64_t total_attempts() const noexcept {
        return total_attempts_.load(std::memory_order_relaxed);
    }

    /**
     * @brief Signals passed (try_acquire() returned true).
     */
    [[nodiscard]] uint64_t passed_count() const noexcept {
        return passed_count_.load(std::memory_order_relaxed);
    }

    /**
     * @brief Signals dropped (try_acquire() returned false).
     */
    [[nodiscard]] uint64_t dropped_count() const noexcept {
        return dropped_count_.load(std::memory_order_relaxed);
    }

    /**
     * @brief Drop rate [0.0, 1.0] = dropped / total_attempts.
     */
    [[nodiscard]] double drop_rate() const noexcept;

    /**
     * @brief Current token level (approximate, may be stale by 1 refill cycle).
     */
    [[nodiscard]] double current_tokens() const;

    /**
     * @brief Reset token bucket to full capacity and clear counters.
     *
     * Thread-safe.
     */
    void reset();

    /**
     * @brief Register a callback fired each time a signal is dropped.
     */
    void set_drop_callback(DropCallback cb) {
        std::lock_guard<std::mutex> lk(mtx_);
        drop_cb_ = std::move(cb);
    }

    /**
     * @brief Update configuration (resets bucket).
     */
    void update_config(const Config& cfg);

    /**
     * @brief Serialise current state to a JSON object string.
     */
    [[nodiscard]] std::string to_stats_json() const;

private:
    using Clock = std::chrono::steady_clock;

    void refill_locked();  // called under mtx_

    Config cfg_;
    mutable std::mutex mtx_;

    double tokens_{0.0};
    Clock::time_point last_refill_{Clock::now()};

    std::atomic<uint64_t> total_attempts_{0};
    std::atomic<uint64_t> passed_count_{0};
    std::atomic<uint64_t> dropped_count_{0};

    DropCallback drop_cb_;
};

} // namespace llmquant
