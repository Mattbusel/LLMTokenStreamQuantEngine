#pragma once

#include <atomic>
#include <cstdint>
#include <functional>
#include <mutex>
#include <string>
#include <vector>

namespace llmquant {

/**
 * @brief Tracks how quickly signal confidence decays after a sentiment burst.
 *
 * After a high-confidence sentiment event, confidence typically decays as
 * the LLM token stream returns to baseline.  Understanding this decay rate
 * helps determine how long a trade signal remains actionable:
 *
 *  - Slow decay (long half-life) → news-driven event; hold position longer.
 *  - Fast decay (short half-life) → noise spike; close quickly.
 *
 * ## Method
 * The tracker fits an online exponential model C(t) = C₀ × exp(-λt) to
 * a rolling window of (age_ms, confidence) pairs using the log-linear least-
 * squares estimator:
 *
 *   log C(t) = log C₀ - λt
 *
 * The decay constant λ = ln(2) / half_life_ms.  A half_life_ms of 0 means
 * confidence is flat (no meaningful decay detected).
 *
 * ## Usage
 * @code
 * ConfidenceDecayTracker tracker;
 * // After each signal evaluation:
 * tracker.record(signal.confidence);
 *
 * // Before sizing a position:
 * double half_life_s = tracker.half_life_ms() / 1000.0;
 * double hold_ms     = tracker.is_fast_decay() ? 200 : 1000;
 * @endcode
 *
 * ## Feature flag
 * Controlled by `LLMQUANT_ENABLE_CONFIDENCE_DECAY` (default ON).
 */
class ConfidenceDecayTracker {
public:
    struct Config {
        /**
         * @brief Number of observations in the rolling window used for fitting.
         * Default 32.
         */
        int window_size{32};

        /**
         * @brief Minimum samples before decay estimates are meaningful.
         * Default 8.
         */
        int min_samples{8};

        /**
         * @brief Half-life below which is_fast_decay() returns true (ms).
         * Default 500 ms.
         */
        double fast_decay_threshold_ms{500.0};

        /**
         * @brief Callback fired when the half-life estimate changes by more
         *        than change_fraction (relative).
         *
         * Parameters: (new_half_life_ms, old_half_life_ms).
         */
        std::function<void(double new_hl, double old_hl)> on_decay_change;

        /** @brief Minimum relative change to trigger on_decay_change. Default 0.2. */
        double change_fraction{0.20};
    };

    // -----------------------------------------------------------------------
    // Construction
    // -----------------------------------------------------------------------

    explicit ConfidenceDecayTracker(Config config = Config{});

    // -----------------------------------------------------------------------
    // Update
    // -----------------------------------------------------------------------

    /**
     * @brief Record a new confidence observation.
     *
     * Uses a monotonic timestamp (std::chrono::steady_clock) internally.
     * Thread-safe.
     *
     * @param confidence  Confidence value in [0, 1].
     */
    void record(double confidence) noexcept;

    // -----------------------------------------------------------------------
    // Query
    // -----------------------------------------------------------------------

    /**
     * @brief Estimated half-life of signal confidence (milliseconds).
     *
     * Returns 0.0 when fewer than min_samples have been collected or when
     * the fitted decay is zero or negative (flat / increasing confidence).
     * Thread-safe.
     */
    [[nodiscard]] double half_life_ms() const noexcept {
        return half_life_ms_.load(std::memory_order_relaxed);
    }

    /**
     * @brief Fitted decay constant λ (1/ms).  Thread-safe.
     */
    [[nodiscard]] double lambda() const noexcept {
        return lambda_.load(std::memory_order_relaxed);
    }

    /**
     * @brief True when half_life_ms() < fast_decay_threshold_ms.
     * Thread-safe.
     */
    [[nodiscard]] bool is_fast_decay() const noexcept;

    /**
     * @brief Total observations recorded.  Thread-safe.
     */
    [[nodiscard]] uint64_t total_records() const noexcept {
        return total_.load(std::memory_order_relaxed);
    }

    // -----------------------------------------------------------------------
    // Config / reset
    // -----------------------------------------------------------------------

    /** @brief Replace configuration; clears window. */
    void update_config(Config config);

    /** @brief Reset window and fitted parameters. */
    void reset();

    /** @brief JSON snapshot of current estimates. */
    [[nodiscard]] std::string to_stats_json() const;

private:
    using Clock     = std::chrono::steady_clock;
    using TimePoint = Clock::time_point;

    /// Refit the decay model from the current window.  Caller holds mutex_.
    void refit_locked();

    mutable std::mutex mutex_;
    Config config_;

    // Ring buffer of (age_ms, log_confidence) pairs.
    struct Sample {
        double age_ms{0.0};
        double log_conf{0.0};
    };
    std::vector<Sample> ring_;
    int    ring_head_{0};
    int    ring_fill_{0};
    TimePoint epoch_;      // timestamp of oldest sample in window

    std::atomic<double>   half_life_ms_{0.0};
    std::atomic<double>   lambda_{0.0};
    std::atomic<double>   fast_threshold_{500.0};
    std::atomic<uint64_t> total_{0};

    double prev_half_life_{0.0};
};

} // namespace llmquant
