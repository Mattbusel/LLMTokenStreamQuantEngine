#pragma once

#include <atomic>
#include <cstdint>
#include <mutex>
#include <string>
#include <vector>

#include "TradeSignalEngine.h"  // TradeSignal

namespace llmquant {

/**
 * @brief Adaptive position sizer using the Kelly Criterion.
 *
 * The Kelly Criterion computes the theoretically optimal fraction of capital
 * to allocate to a bet given an estimated edge (win rate and average win/loss
 * ratio).  For a binary outcome with win probability p, average win w, and
 * average loss l, the Kelly fraction is:
 *
 *   f* = p/l - (1-p)/w  = (p*w - (1-p)*l) / (w*l)
 *
 * In practice the raw Kelly fraction is too aggressive; a common risk-management
 * convention is "fractional Kelly" — scaling the raw fraction by a constant in
 * (0.0, 1.0] (e.g. 0.25 = "quarter Kelly").
 *
 * ## Signal outcome recording
 * Call `record_outcome(win, return_pct)` after each completed signal trade.
 * The win rate and avg win/loss are estimated as exponential moving averages
 * that weight recent outcomes more heavily.
 *
 * ## Signal sizing
 * Call `size_signal(signal)` to scale a TradeSignal's `delta_bias_shift`
 * by the current Kelly fraction.  Returns a copy with the scaled field.
 *
 * ## Safety limits
 * - `max_fraction` caps the Kelly output (default 0.25 = quarter Kelly).
 * - Returns 0.0 when fewer than `min_history` outcomes have been recorded
 *   (avoids acting on an empty history).
 * - Returns 0.0 when estimated edge (p*w - (1-p)*l) is <= 0 (negative edge).
 *
 * ## Feature flag
 * Controlled by `LLMQUANT_ENABLE_KELLY_SIZER` (default ON).
 *
 * ## Thread safety
 * `record_outcome()` and all accessors are thread-safe via a shared mutex.
 * `size_signal()` is thread-safe (reads atomics only).
 *
 * ## Typical usage
 * @code
 * KellyPositionSizer sizer;
 * engine.set_signal_callback([&](const TradeSignal& raw) {
 *     TradeSignal sized = sizer.size_signal(raw);
 *     oms->submit(sized);
 * });
 * // later, when trade result is known:
 * sizer.record_outcome(true,  0.015);  // won 1.5%
 * sizer.record_outcome(false, 0.008);  // lost 0.8%
 * @endcode
 */
class KellyPositionSizer {
public:
    /**
     * @brief Configuration for the Kelly sizer.
     */
    struct Config {
        /**
         * @brief Maximum allowed Kelly fraction (fractional Kelly cap).
         *
         * The raw Kelly fraction is clamped to [0.0, max_fraction].
         * A value of 0.25 implements "quarter Kelly", which is a common
         * risk-management practice.  Set to 1.0 for full Kelly (not recommended).
         */
        double max_fraction{0.25};

        /**
         * @brief EMA smoothing factor for win-rate and win/loss estimates.
         *
         * alpha in (0.0, 1.0]: higher = faster adaptation, lower = more stable.
         * 0.1 weights the most recent outcome at 10% per update.
         */
        double ema_alpha{0.1};

        /**
         * @brief Minimum number of recorded outcomes before sizing is applied.
         *
         * Below this count, size_signal() returns the signal unchanged
         * (Kelly fraction treated as 1.0) to avoid acting on insufficient data.
         */
        int min_history{10};

        /**
         * @brief Minimum absolute win return to count as a non-trivial win.
         * Outcomes below this threshold are not recorded.
         * Set to 0.0 to record all outcomes.
         */
        double min_outcome_return{0.0};

        /**
         * @brief Default assumed win rate before any outcomes are recorded.
         *
         * Used to initialise the EMA; 0.5 = 50/50 prior.
         */
        double prior_win_rate{0.5};

        /**
         * @brief Default assumed average win return before any outcomes are recorded.
         * Initialises the avg_win EMA.
         */
        double prior_avg_win{0.01};

        /**
         * @brief Default assumed average loss return before any outcomes are recorded.
         * Initialises the avg_loss EMA.
         */
        double prior_avg_loss{0.01};
    };

    /**
     * @brief Snapshot of the current sizer statistics.
     */
    struct Stats {
        uint64_t total_wins{0};          ///< Total recorded winning outcomes.
        uint64_t total_losses{0};        ///< Total recorded losing outcomes.
        double   win_rate_ema{0.0};      ///< EMA-smoothed win rate in [0, 1].
        double   avg_win_ema{0.0};       ///< EMA-smoothed average win return.
        double   avg_loss_ema{0.0};      ///< EMA-smoothed average loss magnitude.
        double   kelly_fraction{0.0};    ///< Current computed Kelly fraction.
        double   edge{0.0};             ///< Current estimated edge (p*w - (1-p)*l).
        uint64_t total_sized{0};         ///< Total calls to size_signal().
        uint64_t sized_full{0};          ///< Calls that used full Kelly fraction.
        uint64_t sized_zero{0};          ///< Calls that returned zero (negative edge or insufficient history).
    };

    /**
     * @brief Construct the sizer with default configuration.
     */
    KellyPositionSizer();

    /**
     * @brief Construct the sizer with the given configuration.
     * @param config Kelly parameters.
     */
    explicit KellyPositionSizer(Config config);

    /**
     * @brief Record the outcome of a completed signal-driven trade.
     *
     * Updates the EMA win-rate and avg-win/loss estimates.  Should be called
     * once per completed trade with the actual realised return.
     *
     * @param win        true if the trade was profitable; false otherwise.
     * @param return_pct Absolute return magnitude (e.g. 0.015 = 1.5%).
     *                   Must be >= 0.  The sign is determined by @p win.
     */
    void record_outcome(bool win, double return_pct);

    /**
     * @brief Scale a TradeSignal's delta_bias_shift by the current Kelly fraction.
     *
     * Returns a copy of @p signal with `delta_bias_shift` multiplied by the
     * current Kelly fraction.  All other fields are unchanged.
     *
     * Returns the signal unchanged (Kelly = 1.0) if fewer than min_history
     * outcomes have been recorded; returns with delta_bias_shift=0 if the
     * estimated edge is negative.
     *
     * Thread-safe.
     *
     * @param signal Raw TradeSignal from TradeSignalEngine.
     * @return Sized copy of the signal.
     */
    [[nodiscard]] TradeSignal size_signal(const TradeSignal& signal) const;

    /**
     * @brief Return the current Kelly fraction in [0.0, max_fraction].
     *
     * Returns 0.0 if edge <= 0 or fewer than min_history outcomes recorded.
     * Thread-safe.
     *
     * @return Kelly fraction.
     */
    [[nodiscard]] double get_kelly_fraction() const;

    /**
     * @brief Return the current estimated edge (p*avg_win - (1-p)*avg_loss).
     *
     * Positive edge → Kelly fraction > 0; negative edge → Kelly = 0 (sit out).
     * Thread-safe.
     *
     * @return Estimated edge in the same units as avg_win and avg_loss.
     */
    [[nodiscard]] double get_edge() const;

    /**
     * @brief Return the current EMA-smoothed win rate.
     *
     * Thread-safe.
     *
     * @return Win rate in [0.0, 1.0].
     */
    [[nodiscard]] double get_win_rate() const;

    /**
     * @brief Return a snapshot of all sizer statistics.
     *
     * Thread-safe.
     *
     * @return Stats snapshot.
     */
    [[nodiscard]] Stats get_stats() const;

    /**
     * @brief Return the total number of outcomes recorded since construction or reset().
     *
     * Thread-safe (atomic read).
     *
     * @return Total outcome count.
     */
    [[nodiscard]] uint64_t total_outcomes() const noexcept {
        return total_wins_.load(std::memory_order_relaxed)
             + total_losses_.load(std::memory_order_relaxed);
    }

    /**
     * @brief Reset all outcome history and statistics to initial values.
     *
     * Resets EMA estimates to prior values; min_history counter resets.
     * Thread-safe.
     */
    void reset();

    /**
     * @brief Serialise the current state to a JSON object string.
     *
     * Fields: win_rate, avg_win, avg_loss, kelly_fraction, edge,
     * total_wins, total_losses, total_sized.
     * Thread-safe.
     *
     * @return JSON object string.
     */
    [[nodiscard]] std::string to_stats_json() const;

    /**
     * @brief Update the sizer configuration at runtime.
     *
     * The EMA state is preserved; only the config parameters change.
     * Thread-safe.
     *
     * @param config New configuration.
     */
    void update_config(const Config& config);

    /**
     * @brief Return a copy of the current configuration.
     * Thread-safe.
     *
     * @return Current Config.
     */
    [[nodiscard]] Config get_config() const;

private:
    [[nodiscard]] double compute_kelly_locked() const;

    mutable std::mutex mutex_;
    Config config_;

    double win_rate_ema_;   ///< EMA win-rate estimate.
    double avg_win_ema_;    ///< EMA average win return.
    double avg_loss_ema_;   ///< EMA average loss magnitude.

    std::atomic<uint64_t> total_wins_{0};
    std::atomic<uint64_t> total_losses_{0};
    std::atomic<uint64_t> total_sized_{0};
    std::atomic<uint64_t> sized_full_{0};
    std::atomic<uint64_t> sized_zero_{0};
};

} // namespace llmquant
