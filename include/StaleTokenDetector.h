#pragma once

#include <atomic>
#include <chrono>
#include <cstdint>
#include <functional>
#include <string>

namespace llmquant {

/**
 * @brief Watchdog that detects when the LLM token stream has gone silent.
 *
 * In production, the LLM API can silently stop sending tokens due to network
 * issues, rate-limiting, or upstream outages.  Without a watchdog, the engine
 * continues running and reporting healthy while no signals are generated.
 *
 * This module tracks the wall-clock time since the last token arrived and
 * fires a callback when the gap exceeds a configurable threshold.
 *
 * ## Integration
 * Call `record_token()` from the token-processing path (hot path — lock-free).
 * Call `check()` from the monitoring loop (once per stats tick).
 *
 * The stale state is reflected in the health endpoint if the health callback
 * queries `is_stale()`.
 *
 * ## States
 * - **Fresh**: Last token arrived within `stale_threshold_ms`.
 * - **Stale**: No token for longer than `stale_threshold_ms`.
 *   The `StaleCallback` is fired once on the 0→1 transition.
 * - **Recovered**: Token arrived after a stale period.
 *   The `RecoveryCallback` is fired once on the 1→0 transition.
 *
 * ## Thread safety
 * `record_token()` is lock-free and safe to call from the hot path.
 * `check()` and all accessors are also thread-safe.
 *
 * ## Feature flag
 * Controlled by `LLMQUANT_ENABLE_STALE_DETECTOR` (default ON).
 */
class StaleTokenDetector {
public:
    /**
     * @brief Configuration for the watchdog.
     */
    struct Config {
        /**
         * @brief Milliseconds without a token before the stream is considered stale.
         * Default 30 000 ms (30 seconds).
         */
        int stale_threshold_ms{30000};

        /**
         * @brief How many ms `check()` may be skipped before forcing a fresh check.
         * This is an internal guard to avoid missing state transitions when
         * `check()` is called infrequently.  Default 5 000 ms (5 s).
         */
        int max_check_interval_ms{5000};
    };

    /**
     * @brief Callback invoked when the token stream becomes stale.
     *
     * Fires once on transition Fresh → Stale.
     * Signature: void(int64_t gap_ms)
     * where gap_ms is the milliseconds since the last token.
     */
    using StaleCallback    = std::function<void(int64_t gap_ms)>;

    /**
     * @brief Callback invoked when the token stream recovers from stale state.
     *
     * Fires once on transition Stale → Fresh.
     */
    using RecoveryCallback = std::function<void()>;

    /**
     * @brief Construct with default configuration.
     *
     * The clock starts at construction time; call `record_token()` or
     * `reset()` if the first token is expected to arrive much later.
     */
    explicit StaleTokenDetector(Config config = Config{});

    /**
     * @brief Record that a token just arrived.
     *
     * Updates the internal timestamp.  Lock-free; safe to call from the hot path.
     */
    void record_token() noexcept;

    /**
     * @brief Probe the stale state and fire callbacks if the state changed.
     *
     * Call from the monitoring loop (not the hot path).  Internally compares
     * the wall-clock against the last token timestamp and updates `is_stale()`.
     *
     * @return True if the stream is currently stale after this check.
     */
    bool check() noexcept;

    /**
     * @brief True if the stream is currently considered stale.
     *
     * Updated by `check()`.  Thread-safe (atomic read).
     */
    [[nodiscard]] bool is_stale() const noexcept {
        return stale_.load(std::memory_order_relaxed);
    }

    /**
     * @brief Milliseconds since the last `record_token()` call.
     *
     * Thread-safe.
     */
    [[nodiscard]] int64_t ms_since_last_token() const noexcept;

    /**
     * @brief Total number of times the stream transitioned from fresh → stale.
     *
     * Thread-safe (atomic read).
     */
    [[nodiscard]] uint64_t stale_events() const noexcept {
        return stale_events_.load(std::memory_order_relaxed);
    }

    /**
     * @brief Total number of times the stream recovered from stale → fresh.
     *
     * Thread-safe (atomic read).
     */
    [[nodiscard]] uint64_t recovery_events() const noexcept {
        return recovery_events_.load(std::memory_order_relaxed);
    }

    /**
     * @brief Register the stale transition callback.
     *
     * Replaces any previously registered callback.
     * Not safe to call concurrently with `check()`.
     *
     * @param cb Callable invoked with the gap in milliseconds.
     */
    void set_stale_callback(StaleCallback cb);

    /**
     * @brief Register the recovery transition callback.
     *
     * Replaces any previously registered callback.
     * Not safe to call concurrently with `check()`.
     *
     * @param cb Callable invoked on recovery.
     */
    void set_recovery_callback(RecoveryCallback cb);

    /**
     * @brief Update configuration at runtime.
     *
     * Thread-safe.
     *
     * @param config New thresholds.
     */
    void update_config(const Config& config);

    /**
     * @brief Reset the last-token timestamp to now and clear the stale flag.
     *
     * Use at startup to prevent a false-positive stale event if the
     * monitoring loop starts before the first token arrives.
     * Thread-safe.
     */
    void reset() noexcept;

    /**
     * @brief Serialise current state to a JSON object string.
     *
     * Fields: stale, ms_since_last_token, stale_events, recovery_events,
     * stale_threshold_ms.
     * Thread-safe.
     */
    [[nodiscard]] std::string to_stats_json() const;

private:
    Config config_;
    mutable std::atomic<int64_t> last_token_epoch_ms_;  // epoch ms, atomic for lock-free update
    std::atomic<bool>     stale_{false};
    std::atomic<uint64_t> stale_events_{0};
    std::atomic<uint64_t> recovery_events_{0};

    StaleCallback    stale_cb_;
    RecoveryCallback recovery_cb_;
};

} // namespace llmquant
