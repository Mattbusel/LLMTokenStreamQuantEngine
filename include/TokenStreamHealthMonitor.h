#pragma once

#include <atomic>
#include <chrono>
#include <functional>
#include <mutex>
#include <string>

namespace llmquant {

/**
 * @brief Watchdog that monitors token-stream inter-arrival times to detect
 *        feed stalls and token floods in real time.
 *
 * Low-latency token streams should arrive within a predictable window.
 * Two pathological patterns break production assumptions:
 *
 *  - **Stall**: no tokens arrive for longer than `stall_timeout_ms`.
 *    Causes: LLM generation paused, network partition, upstream queue drain.
 *    Response: suppress new signals; optionally close open positions.
 *
 *  - **Flood**: tokens arrive faster than `max_tokens_per_sec`.
 *    Causes: loop/retry bug upstream, LLM streaming max-tokens hit, replay.
 *    Response: rate-limit ingestion; fire alert so operator can investigate.
 *
 * ## Usage
 * @code
 * TokenStreamHealthMonitor mon;
 * // On every incoming token:
 * mon.ping();
 * // Before using the signal:
 * if (!mon.is_healthy()) skip_signal();
 * @endcode
 *
 * ## Feature flag
 * Controlled by `LLMQUANT_ENABLE_STREAM_HEALTH` (default ON).
 */
class TokenStreamHealthMonitor {
public:
    /** @brief Health status of the token stream. */
    enum class Status {
        Healthy,   ///< Tokens arriving at normal rate.
        Stalled,   ///< No tokens for > stall_timeout_ms.
        Flooded,   ///< Token rate exceeds max_tokens_per_sec.
    };

    struct Config {
        /**
         * @brief Milliseconds of silence before the stream is declared stalled.
         * Default 5000 ms (5 s).
         */
        int64_t stall_timeout_ms{5000};

        /**
         * @brief Maximum acceptable token arrival rate (tokens / second).
         * Default 500 tokens/s.  Set to 0 to disable flood detection.
         */
        double max_tokens_per_sec{500.0};

        /**
         * @brief Window over which token rate is averaged (milliseconds).
         * Default 1000 ms.
         */
        int64_t rate_window_ms{1000};

        /**
         * @brief Callback fired when status transitions to Stalled.
         * Parameter: milliseconds elapsed since last token.
         */
        std::function<void(int64_t elapsed_ms)> on_stall;

        /**
         * @brief Callback fired when status transitions to Flooded.
         * Parameter: current tokens-per-second rate.
         */
        std::function<void(double rate)> on_flood;

        /**
         * @brief Callback fired when status recovers from Stalled or Flooded
         *        back to Healthy.
         */
        std::function<void()> on_recovery;
    };

    // -----------------------------------------------------------------------
    // Construction
    // -----------------------------------------------------------------------

    explicit TokenStreamHealthMonitor(Config config = Config{});

    // -----------------------------------------------------------------------
    // Update
    // -----------------------------------------------------------------------

    /**
     * @brief Record arrival of one token.
     *
     * Thread-safe.  Updates internal arrival times, recomputes rate, and
     * fires status-change callbacks when status transitions occur.
     */
    void ping() noexcept;

    /**
     * @brief Poll the monitor without a token arrival.
     *
     * Call periodically (e.g. every 100 ms) to allow stall detection to
     * fire even when the processing thread is blocked waiting for tokens.
     * Thread-safe.
     */
    void poll() noexcept;

    // -----------------------------------------------------------------------
    // Query
    // -----------------------------------------------------------------------

    /**
     * @brief True when status is Healthy.
     *
     * Thread-safe.
     */
    [[nodiscard]] bool is_healthy() const noexcept {
        return status_.load(std::memory_order_relaxed) == Status::Healthy;
    }

    /** @brief Current health status.  Thread-safe. */
    [[nodiscard]] Status status() const noexcept {
        return status_.load(std::memory_order_relaxed);
    }

    /**
     * @brief Estimated token arrival rate over the last rate_window_ms.
     *
     * Returns 0.0 if no tokens have been received yet.  Thread-safe.
     */
    [[nodiscard]] double current_rate() const noexcept {
        return rate_.load(std::memory_order_relaxed);
    }

    /**
     * @brief Milliseconds since the last token arrived.
     *
     * Returns INT64_MAX if no token has ever been received.  Thread-safe.
     */
    [[nodiscard]] int64_t ms_since_last_token() const noexcept;

    /** @brief Total tokens received.  Thread-safe. */
    [[nodiscard]] uint64_t total_tokens() const noexcept {
        return total_tokens_.load(std::memory_order_relaxed);
    }

    /** @brief Total stall events detected.  Thread-safe. */
    [[nodiscard]] uint64_t stall_count() const noexcept {
        return stall_count_.load(std::memory_order_relaxed);
    }

    /** @brief Total flood events detected.  Thread-safe. */
    [[nodiscard]] uint64_t flood_count() const noexcept {
        return flood_count_.load(std::memory_order_relaxed);
    }

    // -----------------------------------------------------------------------
    // Config / reset
    // -----------------------------------------------------------------------

    /** @brief Replace configuration; resets all counters and timestamps. */
    void update_config(Config config);

    /** @brief Reset counters and timestamps; status → Healthy. */
    void reset();

    /** @brief JSON snapshot of current state. */
    [[nodiscard]] std::string to_stats_json() const;

private:
    using Clock    = std::chrono::steady_clock;
    using TimePoint = Clock::time_point;

    /// Evaluate rate + stall; caller must hold mutex_.
    /// Returns new status; fires callbacks outside the lock if changed.
    Status evaluate_locked(TimePoint now);

    mutable std::mutex mutex_;
    Config config_;

    TimePoint last_token_tp_{};
    bool      ever_received_{false};

    // Rolling token-count window: count tokens in [now - rate_window_ms, now].
    // We keep a ring of sub-bucket counts for efficiency.
    static constexpr int kBuckets = 10;
    int64_t  bucket_counts_[kBuckets]{};
    int      bucket_head_{0};
    TimePoint bucket_start_tp_{};

    std::atomic<Status>  status_{Status::Healthy};
    std::atomic<double>  rate_{0.0};
    std::atomic<uint64_t> total_tokens_{0};
    std::atomic<uint64_t> stall_count_{0};
    std::atomic<uint64_t> flood_count_{0};

    Status prev_status_{Status::Healthy};
};

} // namespace llmquant
