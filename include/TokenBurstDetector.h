#pragma once

#include <atomic>
#include <chrono>
#include <cstdint>
#include <functional>
#include <mutex>
#include <string>

namespace llmquant {

/**
 * @brief Sliding-window token arrival rate monitor with burst detection.
 *
 * Tracks how many tokens have arrived in the most recent `window_ms`
 * milliseconds.  When the instantaneous rate exceeds `burst_threshold`
 * tokens/second the detector transitions to BURST state and fires
 * `on_burst_start`.  When the rate drops back below the threshold it
 * transitions to NORMAL and fires `on_burst_end`.
 *
 * ## Why this matters
 * High-speed token bursts (e.g. model generating at 500 tok/s vs usual 50)
 * indicate a flush/catch-up period where the sentiment signal may be
 * backlogged and stale.  The engine can use burst state to widen spread
 * cost, reduce position size, or skip non-critical signals.
 *
 * ## Thread safety
 * All public methods are thread-safe.
 *
 * ## Feature flag
 * Controlled by `LLMQUANT_ENABLE_BURST_DETECTOR` (default ON).
 */
class TokenBurstDetector {
public:
    /**
     * @brief Configuration.
     */
    struct Config {
        /**
         * @brief Sliding window length in milliseconds.
         *
         * Tokens older than this are evicted from the window.  Default 500 ms.
         */
        int window_ms{500};

        /**
         * @brief Burst threshold in tokens per second.
         *
         * When the instantaneous rate (tokens in window / window_s) exceeds
         * this value the detector enters BURST state.  Default 100.0.
         */
        double burst_threshold{100.0};

        /**
         * @brief Hysteresis factor applied to the threshold on the exit edge.
         *
         * Exit threshold = burst_threshold * (1 - hysteresis).
         * Prevents rapid oscillation around the threshold.  Default 0.1.
         */
        double hysteresis{0.1};

        /**
         * @brief Callback fired when the detector enters BURST state.
         *
         * Parameter: current instantaneous rate (tok/s).
         * Called from the thread invoking record().
         */
        std::function<void(double)> on_burst_start;

        /**
         * @brief Callback fired when the detector exits BURST state.
         *
         * Parameter: current instantaneous rate at exit.
         */
        std::function<void(double)> on_burst_end;
    };

    // -----------------------------------------------------------------------
    // Construction
    // -----------------------------------------------------------------------

    explicit TokenBurstDetector(Config config = Config{});

    // -----------------------------------------------------------------------
    // Update
    // -----------------------------------------------------------------------

    /**
     * @brief Record one token arrival at the current wall-clock time.
     *
     * Evicts expired arrivals, updates the current rate, and fires
     * burst/end callbacks on state transitions.  Thread-safe.
     *
     * @return Instantaneous rate (tokens/second) after recording.
     */
    double record();

    /**
     * @brief Record one token arrival at an explicit timestamp.
     *
     * Useful for testing or deterministic replay.  Thread-safe.
     *
     * @param tp  Arrival timestamp.
     * @return    Instantaneous rate (tokens/second).
     */
    double record_at(std::chrono::steady_clock::time_point tp);

    /**
     * @brief Reset state (clear the window and exit burst mode).
     *
     * Thread-safe.
     */
    void reset();

    // -----------------------------------------------------------------------
    // Query
    // -----------------------------------------------------------------------

    /**
     * @brief Instantaneous rate (tokens/second) over the current window.
     *
     * Thread-safe.
     */
    [[nodiscard]] double current_rate() const noexcept;

    /**
     * @brief True if currently in BURST state.
     *
     * Thread-safe.
     */
    [[nodiscard]] bool is_burst() const noexcept {
        return in_burst_.load(std::memory_order_relaxed);
    }

    // -----------------------------------------------------------------------
    // Statistics
    // -----------------------------------------------------------------------

    /** @brief Total token arrivals recorded. */
    [[nodiscard]] uint64_t total_tokens() const noexcept {
        return total_tokens_.load(std::memory_order_relaxed);
    }

    /** @brief Number of times the detector entered BURST state. */
    [[nodiscard]] uint64_t burst_events() const noexcept {
        return burst_events_.load(std::memory_order_relaxed);
    }

    /**
     * @brief Update configuration; resets the window.
     *
     * Thread-safe.
     */
    void update_config(const Config& config);

    /**
     * @brief Serialise state to a JSON object string.
     */
    [[nodiscard]] std::string to_stats_json() const;

private:
    mutable std::mutex mutex_;
    Config config_;

    // Ring of arrival timestamps; stored as nanoseconds since epoch for
    // efficient eviction without per-element allocation.
    std::vector<int64_t> ring_;  // circular buffer of arrival ns timestamps
    int                  head_{0};
    int                  count_{0};

    double last_rate_{0.0};

    std::atomic<bool>     in_burst_{false};
    std::atomic<uint64_t> total_tokens_{0};
    std::atomic<uint64_t> burst_events_{0};

    // Evict arrivals older than window_ms from the deque; returns new count.
    // Must be called under mutex_.
    double evict_and_rate(int64_t now_ns);
};

} // namespace llmquant
