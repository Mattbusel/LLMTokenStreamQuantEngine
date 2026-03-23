#pragma once

#include <atomic>
#include <chrono>
#include <cstdint>
#include <functional>
#include <mutex>
#include <string>

namespace llmquant {

/**
 * @brief Per-stage latency budget enforcer for the signal pipeline hot path.
 *
 * Each signal pipeline stage registers a latency budget (microseconds). When a
 * stage exceeds its budget, `check_deadline()` returns false, signalling the
 * caller to emit a partial / approximate result rather than waiting for full
 * computation. This ensures the hot path is never blocked by a slow stage.
 *
 * ## Design
 * ```
 * auto token = budget.start();          // capture start timestamp
 * bool ok = budget.check_deadline(token); // true if still within budget
 * if (!ok) { emit_approximate(); return; }
 * // ... full computation ...
 * budget.record_completion(token);      // record actual latency for stats
 * ```
 *
 * ## Statistics
 * Tracks per-stage: total calls, deadline hits, total latency, p99 latency
 * (via a simple count-min approximation stored as a sorted ring buffer).
 *
 * ## Thread safety
 * All methods are thread-safe.
 *
 * ## Feature flag
 * Controlled by `LLMQUANT_ENABLE_LATENCY_BUDGET` (default ON).
 */
class LatencyBudget {
public:
    using Clock     = std::chrono::steady_clock;
    using TimePoint = Clock::time_point;

    /**
     * @brief Configuration for a single pipeline stage budget.
     */
    struct Config {
        /**
         * @brief Stage name for logging and JSON output.
         * Default: "unnamed".
         */
        std::string stage_name{"unnamed"};

        /**
         * @brief Latency budget in microseconds.
         *
         * If processing takes longer than this, `check_deadline()` returns false
         * and the stage should emit a partial result.
         * Default: 500 µs.
         */
        int64_t budget_us{500};

        /**
         * @brief Size of the ring buffer used for rolling latency percentile tracking.
         * Default: 1024.
         */
        size_t ring_size{1024};
    };

    /** @brief Opaque token representing a single stage invocation. */
    struct Token {
        TimePoint start;
    };

    explicit LatencyBudget(Config cfg = Config{});

    // Non-copyable.
    LatencyBudget(const LatencyBudget&)            = delete;
    LatencyBudget& operator=(const LatencyBudget&) = delete;

    /**
     * @brief Begin timing a stage invocation.
     *
     * Call this at the entry point of the pipeline stage.
     * Thread-safe.
     *
     * @return An opaque Token to pass to check_deadline() and record_completion().
     */
    [[nodiscard]] Token start() const noexcept;

    /**
     * @brief Check whether the stage is still within its latency budget.
     *
     * Returns true if elapsed time since `token.start` is less than `budget_us`.
     * Increments deadline-exceeded counter when false.
     *
     * Thread-safe.
     *
     * @param token  Token returned by start().
     * @return true  if within budget, false if budget exceeded.
     */
    [[nodiscard]] bool check_deadline(const Token& token);

    /**
     * @brief Record the actual latency for a completed stage invocation.
     *
     * Updates rolling statistics (mean, p99 ring buffer).
     * Should be called once the stage finishes, regardless of whether the
     * deadline was met.
     *
     * Thread-safe.
     *
     * @param token   Token from start().
     */
    void record_completion(const Token& token);

    /**
     * @brief Return the configured budget in microseconds.
     */
    [[nodiscard]] int64_t budget_us() const noexcept { return cfg_.budget_us; }

    /**
     * @brief Total calls to check_deadline() since construction or reset().
     *
     * Thread-safe.
     */
    [[nodiscard]] uint64_t total_calls() const noexcept {
        return total_calls_.load(std::memory_order_relaxed);
    }

    /**
     * @brief Number of times check_deadline() returned false (budget exceeded).
     *
     * Thread-safe.
     */
    [[nodiscard]] uint64_t deadline_exceeded_count() const noexcept {
        return deadline_exceeded_.load(std::memory_order_relaxed);
    }

    /**
     * @brief Approximate p99 latency in microseconds from the ring buffer.
     *
     * Thread-safe.
     */
    [[nodiscard]] int64_t p99_us() const;

    /**
     * @brief Mean latency in microseconds across all recorded completions.
     *
     * Thread-safe.
     */
    [[nodiscard]] double mean_us() const;

    /**
     * @brief Reset all statistics and ring buffer.
     *
     * Thread-safe.
     */
    void reset();

    /**
     * @brief Update configuration; resets statistics.
     *
     * Thread-safe.
     */
    void update_config(const Config& cfg);

    /**
     * @brief Serialise state to a JSON object string.
     *
     * Fields: stage_name, budget_us, total_calls, deadline_exceeded,
     * deadline_exceeded_rate, mean_us, p99_us.
     */
    [[nodiscard]] std::string to_stats_json() const;

private:
    mutable std::mutex mtx_;
    Config cfg_;

    std::atomic<uint64_t> total_calls_{0};
    std::atomic<uint64_t> deadline_exceeded_{0};

    // Ring buffer of recent latency samples (microseconds) for percentile estimation.
    std::vector<int64_t> ring_;
    size_t ring_pos_{0};
    size_t ring_count_{0};

    // Running sum for mean computation.
    int64_t latency_sum_us_{0};
    uint64_t completion_count_{0};
};

} // namespace llmquant
