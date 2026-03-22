#pragma once

#include <atomic>
#include <cstdint>
#include <functional>
#include <mutex>
#include <string>

namespace llmquant {

/**
 * @brief Tracks consecutive same-direction signals for conviction scoring.
 *
 * Each time a signal passes risk evaluation, the caller records its
 * direction (bullish/bearish/neutral).  Consecutive signals in the same
 * direction increment a streak counter; a reversal resets it and fires
 * `on_reversal`.  When the streak exceeds `conviction_threshold`, the
 * detector fires `on_conviction` and returns a conviction multiplier > 1.0
 * for use in position sizing.
 *
 * ## Rationale
 * Persistent directional signals are a proxy for a genuine trend in the
 * LLM sentiment stream.  A single-token spike is noise; 10 consecutive
 * bullish signals with no reversal is conviction.  The Kelly sizer can
 * multiply its output by `conviction_scale()` to lean harder into trends.
 *
 * ## Feature flag
 * Controlled by `LLMQUANT_ENABLE_SIGNAL_PERSISTENCE` (default ON).
 */
class SignalPersistenceTracker {
public:
    /**
     * @brief Signal direction enum.
     */
    enum class Direction { Bullish, Bearish, Neutral };

    /**
     * @brief Configuration.
     */
    struct Config {
        /**
         * @brief Streak length that triggers on_conviction.
         *
         * Default 5 (5 consecutive same-direction signals = high conviction).
         */
        int conviction_threshold{5};

        /**
         * @brief Maximum conviction multiplier returned by conviction_scale().
         *
         * Capped so a very long streak does not produce unreasonably large
         * sizes.  Scale = min(max_scale, 1 + streak / conviction_threshold).
         * Default 2.0.
         */
        double max_scale{2.0};

        /**
         * @brief Callback fired when the streak first crosses conviction_threshold.
         *
         * Parameter: current streak length.
         * Not re-fired for every additional same-direction signal above the
         * threshold (edge-triggered on the first crossing only).
         */
        std::function<void(int)> on_conviction;

        /**
         * @brief Callback fired when the direction reverses.
         *
         * Parameters: (previous streak, previous direction).
         */
        std::function<void(int, Direction)> on_reversal;
    };

    // -----------------------------------------------------------------------
    // Construction
    // -----------------------------------------------------------------------

    explicit SignalPersistenceTracker(Config config = Config{});

    // -----------------------------------------------------------------------
    // Update
    // -----------------------------------------------------------------------

    /**
     * @brief Record a new signal direction.
     *
     * Neutral signals do not break a streak but do not extend it either.
     * Thread-safe.
     *
     * @param dir  Direction of the most recent signal.
     * @return     Current streak length after recording.
     */
    int record(Direction dir);

    /**
     * @brief Convenience overload: positive bias = Bullish, negative = Bearish.
     *
     * Values within [-deadband, +deadband] are treated as Neutral.
     * Thread-safe.
     *
     * @param bias_value  Raw delta_bias_shift.
     * @param deadband    Minimum absolute value for a directional signal.
     * @return            Current streak length.
     */
    int record_bias(double bias_value, double deadband = 0.001);

    /**
     * @brief Reset streak and direction history.
     *
     * Thread-safe.
     */
    void reset();

    // -----------------------------------------------------------------------
    // Query
    // -----------------------------------------------------------------------

    /**
     * @brief Current streak length (0 if no directional signals recorded).
     *
     * Thread-safe.
     */
    [[nodiscard]] int current_streak() const noexcept;

    /**
     * @brief Current direction of the streak.
     *
     * Thread-safe.
     */
    [[nodiscard]] Direction current_direction() const noexcept;

    /**
     * @brief Position size multiplier based on streak: in [1.0, max_scale].
     *
     * Formula: min(max_scale, 1 + streak / conviction_threshold).
     * Thread-safe.
     */
    [[nodiscard]] double conviction_scale() const noexcept;

    // -----------------------------------------------------------------------
    // Statistics
    // -----------------------------------------------------------------------

    /** @brief Total directional signals recorded. */
    [[nodiscard]] uint64_t total_signals() const noexcept {
        return total_signals_.load(std::memory_order_relaxed);
    }

    /** @brief Number of direction reversals. */
    [[nodiscard]] uint64_t total_reversals() const noexcept {
        return total_reversals_.load(std::memory_order_relaxed);
    }

    /** @brief Number of times conviction threshold was crossed. */
    [[nodiscard]] uint64_t conviction_events() const noexcept {
        return conviction_events_.load(std::memory_order_relaxed);
    }

    /**
     * @brief Update configuration (does not reset the streak).
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
    Config             config_;

    int       streak_{0};
    Direction direction_{Direction::Neutral};
    bool      conviction_fired_{false}; // edge-trigger: only fire once per streak

    std::atomic<uint64_t> total_signals_{0};
    std::atomic<uint64_t> total_reversals_{0};
    std::atomic<uint64_t> conviction_events_{0};
};

} // namespace llmquant
