#pragma once

#include <atomic>
#include <chrono>
#include <cstdint>
#include <functional>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

#include "KellyPositionSizer.h"
#include "TradeSignalEngine.h"

namespace llmquant {

/**
 * @brief Real-time P&L tracker that automatically feeds trade outcomes back
 *        into a KellyPositionSizer, closing the adaptive sizing feedback loop.
 *
 * ## Problem solved
 * KellyPositionSizer is adaptive — but it only adapts when outcomes are
 * explicitly recorded via `record_outcome()`.  In production, outcomes arrive
 * from the OMS (fill price vs. entry price).  This module bridges that gap:
 * record entries with `open_trade()`, close them with `close_trade()`, and the
 * realised P&L is automatically forwarded to the Kelly sizer.
 *
 * ## Lifecycle
 * @code
 * PositionTracker tracker(kelly_sizer);
 * // When a signal fires and the OMS acknowledges entry:
 * uint64_t id = tracker.open_trade(signal, entry_price);
 * // When the OMS confirms the exit:
 * tracker.close_trade(id, exit_price);
 * // Kelly sizer now has an updated win/loss EMA.
 * @endcode
 *
 * ## Statistics
 * Tracks total trades, win rate, average win/loss, cumulative P&L, max
 * drawdown, and best/worst single trade return.
 *
 * ## Thread safety
 * All public methods are thread-safe via a shared mutex.
 *
 * ## Feature flag
 * Controlled by `LLMQUANT_ENABLE_POSITION_TRACKER` (default ON).
 */
class PositionTracker {
public:
    /**
     * @brief Configuration for the position tracker.
     */
    struct Config {
        /**
         * @brief Maximum number of concurrent open trades tracked.
         *
         * If this limit is reached, `open_trade()` returns 0 (rejected).
         * Prevents unbounded memory growth under runaway signal emission.
         */
        int max_open_trades{64};

        /**
         * @brief If true, automatically call `kelly_sizer.record_outcome()`
         *        when a trade is closed.  Set false to disable auto-feedback
         *        while still tracking P&L statistics.
         */
        bool auto_kelly_feedback{true};

        /**
         * @brief Minimum absolute return to count as a non-trivial trade for
         *        Kelly feedback.  Trades with |return| below this are tracked
         *        but not forwarded to the Kelly sizer.
         *        Default 0.0 (all trades forwarded).
         */
        double min_return_for_kelly{0.0};
    };

    /**
     * @brief Snapshot of a single open or recently closed trade.
     */
    struct Trade {
        uint64_t    id{0};
        double      entry_price{0.0};
        double      signal_bias{0.0};   ///< delta_bias_shift at entry.
        double      size_fraction{0.0}; ///< Kelly fraction at entry.
        std::chrono::steady_clock::time_point entry_time;
    };

    /**
     * @brief Aggregate performance statistics.
     */
    struct Stats {
        uint64_t total_trades{0};
        uint64_t total_wins{0};
        uint64_t total_losses{0};
        double   win_rate{0.0};         ///< total_wins / total_trades.
        double   avg_win_return{0.0};   ///< Average return on winning trades.
        double   avg_loss_return{0.0};  ///< Average |return| on losing trades.
        double   cumulative_pnl{0.0};   ///< Sum of all realised returns.
        double   max_drawdown{0.0};     ///< Peak-to-trough drawdown (positive = loss).
        double   best_trade{0.0};       ///< Best single-trade return.
        double   worst_trade{0.0};      ///< Worst single-trade return (negative).
        uint64_t open_trades{0};        ///< Currently open (not yet closed).
        uint64_t rejected_opens{0};     ///< Rejected due to max_open_trades limit.
    };

    /**
     * @brief Callback invoked when a trade is closed with its realised return.
     *
     * Signature: void(uint64_t trade_id, double return_pct, bool win)
     */
    using TradeCloseCallback =
        std::function<void(uint64_t trade_id, double return_pct, bool win)>;

    /**
     * @brief Construct with a reference to the Kelly sizer to feed outcomes into.
     *
     * The caller must ensure that @p kelly_sizer outlives this tracker.
     *
     * @param kelly_sizer Reference to the sizer to update on trade close.
     * @param config      Tracker configuration.
     */
    explicit PositionTracker(KellyPositionSizer& kelly_sizer,
                             Config config = Config{});

    /**
     * @brief Record a new open trade.
     *
     * @param signal       The sized TradeSignal that triggered this entry.
     * @param entry_price  Fill price at which the position was entered.
     * @return Unique trade ID (> 0).  Returns 0 if max_open_trades is reached.
     */
    [[nodiscard]] uint64_t open_trade(const TradeSignal& signal,
                                      double entry_price);

    /**
     * @brief Close an open trade and record the realised P&L.
     *
     * Computes `return_pct = (exit_price - entry_price) / entry_price`.
     * If the signal had a negative bias (short), the return is negated.
     * Forwards the outcome to the Kelly sizer if auto_kelly_feedback is true.
     *
     * @param trade_id    ID returned by open_trade().
     * @param exit_price  Fill price at which the position was exited.
     * @return true if the trade ID was found and closed; false otherwise.
     */
    bool close_trade(uint64_t trade_id, double exit_price);

    /**
     * @brief Return the current aggregate statistics snapshot.
     *
     * Thread-safe.
     */
    [[nodiscard]] Stats get_stats() const;

    /**
     * @brief Return the number of currently open (un-closed) trades.
     *
     * Thread-safe (reads lock).
     */
    [[nodiscard]] uint64_t open_trade_count() const;

    /**
     * @brief Register a callback invoked on each trade close.
     *
     * Replaces any previously registered callback.
     * Not safe to call concurrently with close_trade().
     *
     * @param cb Callable invoked with trade_id, return_pct, and win flag.
     */
    void set_trade_close_callback(TradeCloseCallback cb);

    /**
     * @brief Serialise aggregate statistics to a compact JSON string.
     *
     * Fields: total_trades, win_rate, avg_win_return, avg_loss_return,
     * cumulative_pnl, max_drawdown, best_trade, worst_trade, open_trades.
     * Thread-safe.
     */
    [[nodiscard]] std::string to_stats_json() const;

    /**
     * @brief Reset all statistics and clear open trades.
     *
     * Thread-safe.
     */
    void reset();

    /**
     * @brief Update configuration at runtime.
     *
     * Thread-safe.
     */
    void update_config(const Config& config);

    /**
     * @brief Return the current configuration.
     *
     * Thread-safe.
     */
    [[nodiscard]] Config get_config() const;

private:
    mutable std::mutex mutex_;
    KellyPositionSizer& kelly_sizer_;
    Config config_;

    std::unordered_map<uint64_t, Trade> open_trades_;
    std::atomic<uint64_t> next_id_{1};

    // Cumulative statistics (updated under mutex_).
    uint64_t total_trades_{0};
    uint64_t total_wins_{0};
    uint64_t total_losses_{0};
    double   sum_win_returns_{0.0};
    double   sum_loss_returns_{0.0};
    double   cumulative_pnl_{0.0};
    double   peak_pnl_{0.0};
    double   max_drawdown_{0.0};
    double   best_trade_{0.0};
    double   worst_trade_{0.0};
    uint64_t rejected_opens_{0};

    TradeCloseCallback close_cb_;
};

} // namespace llmquant
