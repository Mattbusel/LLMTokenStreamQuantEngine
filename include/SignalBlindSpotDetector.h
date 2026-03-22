#pragma once

#include <atomic>
#include <array>
#include <cstdint>
#include <functional>
#include <mutex>
#include <string>

namespace llmquant {

/**
 * @brief Detects calendar blind spots where signals consistently underperform.
 *
 * LLM sentiment models are calibrated on general text corpora and may
 * systematically misread market-specific events that recur at fixed calendar
 * intervals:
 *  - Pre-market opening (9:00–9:30 ET): thin liquidity, noisy price action.
 *  - End of month / quarter: window dressing, rebalancing flows.
 *  - FOMC announcement hours (2:00–2:30 PM ET, 8 times/year).
 *  - Earnings season (peaks in Jan, Apr, Jul, Oct).
 *
 * This detector accumulates per-slot win-rate statistics (where a "slot" is
 * a configurable time bucket — e.g. hour-of-day) and returns a suppression
 * flag for slots with historically poor performance.
 *
 * ## Usage
 * @code
 * SignalBlindSpotDetector bsd;
 * // After each trade resolves:
 * bsd.record_outcome(hour_of_day, won);
 *
 * // Before emitting a signal:
 * if (bsd.is_blind_spot(current_hour)) skip_signal();
 * @endcode
 *
 * ## Feature flag
 * Controlled by `LLMQUANT_ENABLE_BLIND_SPOT` (default ON).
 */
class SignalBlindSpotDetector {
public:
    /**
     * @brief Number of time slots.
     *
     * Default: 24 (one per hour of the day).
     */
    static constexpr int kSlots = 24;

    /**
     * @brief Configuration.
     */
    struct Config {
        /**
         * @brief Minimum outcomes per slot before it can be marked a blind spot.
         *
         * Prevents a single early loss from suppressing a slot.
         * Default 10.
         */
        int min_samples{10};

        /**
         * @brief Win-rate threshold below which a slot is a blind spot.
         *
         * Default 0.4 (40%).
         */
        double blind_spot_threshold{0.4};

        /**
         * @brief Forgetting factor per new observation (exponential moving average).
         *
         * 1.0 = full history, 0.95 = gentle forgetting.
         * Default 0.98 (slow adaptation).
         */
        double forgetting{0.98};

        /**
         * @brief Callback fired the first time a slot is identified as a blind spot.
         *
         * Parameters: (slot_index, win_rate).
         */
        std::function<void(int slot, double win_rate)> on_blind_spot_found;
    };

    // -----------------------------------------------------------------------
    // Construction
    // -----------------------------------------------------------------------

    explicit SignalBlindSpotDetector(Config config = Config{});

    // -----------------------------------------------------------------------
    // Update
    // -----------------------------------------------------------------------

    /**
     * @brief Record the outcome of a trade that occurred in the given slot.
     *
     * @param slot  Time slot index [0, kSlots).  Values outside this range
     *              are silently ignored.
     * @param won   True if the trade was profitable.
     * Thread-safe.
     */
    void record_outcome(int slot, bool won);

    /**
     * @brief Reset all slot statistics.
     *
     * Thread-safe.
     */
    void reset();

    // -----------------------------------------------------------------------
    // Query
    // -----------------------------------------------------------------------

    /**
     * @brief True if the given slot is a blind spot.
     *
     * Returns false if the slot has fewer than min_samples.  Thread-safe.
     *
     * @param slot  Time slot [0, kSlots).
     */
    [[nodiscard]] bool is_blind_spot(int slot) const noexcept;

    /**
     * @brief Estimated win rate for the given slot.
     *
     * Returns 0.5 if fewer than min_samples recorded.  Thread-safe.
     *
     * @param slot  Time slot [0, kSlots).
     */
    [[nodiscard]] double win_rate(int slot) const noexcept;

    /**
     * @brief Number of outcomes recorded for a slot.
     *
     * Thread-safe.
     */
    [[nodiscard]] int sample_count(int slot) const noexcept;

    /**
     * @brief Number of slots currently marked as blind spots.
     *
     * Thread-safe.
     */
    [[nodiscard]] int blind_spot_count() const noexcept;

    // -----------------------------------------------------------------------
    // Statistics
    // -----------------------------------------------------------------------

    /** @brief Total outcomes recorded across all slots. */
    [[nodiscard]] uint64_t total_outcomes() const noexcept {
        return total_outcomes_.load(std::memory_order_relaxed);
    }

    /** @brief Total blind-spot detection events (first-time per slot). */
    [[nodiscard]] uint64_t detection_events() const noexcept {
        return detection_events_.load(std::memory_order_relaxed);
    }

    /**
     * @brief Update configuration; clears all slot statistics.
     *
     * Thread-safe.
     */
    void update_config(const Config& config);

    /**
     * @brief Serialise all slot statistics to a JSON object string.
     */
    [[nodiscard]] std::string to_stats_json() const;

private:
    mutable std::mutex mutex_;
    Config config_;

    struct SlotStats {
        double ema_win_rate{0.5};
        double ema_count{0.0};     // effective sample count under forgetting
        bool   is_blind{false};    // once set, remains until reset
        bool   alerted{false};     // have we fired the on_blind_spot_found callback?
    };

    std::array<SlotStats, kSlots> slots_{};

    std::atomic<uint64_t> total_outcomes_{0};
    std::atomic<uint64_t> detection_events_{0};
};

} // namespace llmquant
