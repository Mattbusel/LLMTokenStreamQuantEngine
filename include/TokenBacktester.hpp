#pragma once

/// @file TokenBacktester.hpp
/// @brief Token-sentiment backtester that replays historical signals against
///        OHLCV price bars and produces a full BacktestResult.
///
/// ## Overview
/// `TokenBacktester` bridges the gap between raw token-sentiment signals and
/// a realistic historical simulation by:
///
///   1. **Loading OHLCV price bars** (`load_price_history`) from a CSV file
///      with columns: `timestamp_ms,open,high,low,close,volume`.
///
///   2. **Replaying a signal log** (`replay_signals`) from a tab/comma
///      separated file where each row is a historical token-sentiment signal
///      with nanosecond timestamps: `signal_time_ns,direction,bias,confidence`.
///
///   3. **Matching signals to bars** within a ±100 ms tolerance window
///      using the bar whose timestamp is closest to the signal time.
///
///   4. **Computing PnL** per matched signal using:
///         simulated_return = direction * (close - open) / open
///                          - spread_cost * |bias|
///      (A positive direction with a positive close-open move is profitable.)
///
///   5. **Returning a `BacktestResult`** with total_return, Sharpe ratio,
///      max drawdown, win rate, and average hold time.
///
/// ## CSV Formats
///
/// ### Price history CSV
/// ```
/// timestamp_ms,open,high,low,close,volume
/// 1700000000000,100.0,101.5,99.5,101.0,50000
/// ```
/// Lines starting with `#` and the header line are skipped.
///
/// ### Signal log CSV/TSV
/// ```
/// signal_time_ns,direction,bias,confidence
/// 1700000000000000000,1,0.72,0.85
/// ```
/// `direction` is +1 (buy) or -1 (sell).
/// Lines starting with `#` and the header line are skipped.
///
/// ## Signal matching
/// The matcher finds the bar whose `timestamp_ms * 1_000_000` is within
/// ±100 ms (i.e. ±100_000_000 ns) of `signal_time_ns`.  When multiple bars
/// are within the window the closest one is used.  Unmatched signals are
/// counted in `BacktestResult::unmatched_signals`.
///
/// ## Thread safety
/// Not thread-safe; use one instance per thread.
///
/// ## Feature flag
/// `LLMQUANT_ENABLE_TOKEN_BACKTESTER` (new; independent of BacktestRunner).

#include <cstdint>
#include <string>
#include <vector>

namespace llmquant {

// ---------------------------------------------------------------------------
// HistoricalBar
// ---------------------------------------------------------------------------

/**
 * @brief A single OHLCV price bar.
 */
struct HistoricalBar {
    /// Bar opening timestamp in milliseconds since Unix epoch.
    int64_t timestamp_ms{0};

    double open{0.0};
    double high{0.0};
    double low{0.0};
    double close{0.0};
    double volume{0.0};
};

// ---------------------------------------------------------------------------
// SignalEntry (internal representation of a replayed signal)
// ---------------------------------------------------------------------------

/**
 * @brief A historical token-sentiment signal entry from the signal log.
 */
struct SignalEntry {
    /// Signal emission timestamp in nanoseconds since Unix epoch.
    int64_t signal_time_ns{0};

    /// Trade direction: +1 buy, -1 sell.
    int direction{0};

    /// Raw directional bias [-1, 1].
    double bias{0.0};

    /// Signal confidence [0, 1].
    double confidence{0.0};
};

// ---------------------------------------------------------------------------
// BacktestResult
// ---------------------------------------------------------------------------

/**
 * @brief Aggregate statistics from a completed token-sentiment backtest.
 */
struct BacktestResult {
    // -----------------------------------------------------------------------
    // Input summary
    // -----------------------------------------------------------------------

    /// Number of bars loaded.
    int  total_bars{0};

    /// Number of signals in the log.
    int  total_signals{0};

    /// Signals successfully matched to a bar within ±100 ms.
    int  matched_signals{0};

    /// Signals with no bar within the ±100 ms window.
    int  unmatched_signals{0};

    // -----------------------------------------------------------------------
    // PnL metrics
    // -----------------------------------------------------------------------

    /// Sum of all per-signal simulated returns.
    double total_return{0.0};

    /**
     * @brief Annualised Sharpe ratio of per-signal returns.
     *
     * Computed as mean(returns) / std(returns) * sqrt(252) using per-signal
     * returns (not daily).  Returns 0.0 if fewer than 2 matched signals.
     */
    double sharpe{0.0};

    /**
     * @brief Maximum peak-to-trough decline in cumulative return curve.
     *
     * Always >= 0.  Measured in the same units as total_return.
     */
    double max_drawdown{0.0};

    /**
     * @brief Fraction of matched signals with simulated_return > 0.
     *
     * Returns 0.0 if no matched signals.
     */
    double win_rate{0.0};

    /**
     * @brief Mean signal hold time in milliseconds.
     *
     * Defined as the time between consecutive matched-signal bar timestamps.
     * Returns 0.0 if fewer than 2 matched signals.
     */
    double avg_hold_time_ms{0.0};

    // -----------------------------------------------------------------------
    // Per-signal detail
    // -----------------------------------------------------------------------

    struct SignalResult {
        SignalEntry entry;
        HistoricalBar bar;          ///< Matched bar.
        double simulated_return{0.0};
        double cumulative_return{0.0};
        bool   matched{false};
    };

    /// Per-signal detail records in signal-log order.
    std::vector<SignalResult> records;

    // -----------------------------------------------------------------------
    // Formatting
    // -----------------------------------------------------------------------

    /// Multi-line human-readable summary.
    [[nodiscard]] std::string to_summary_string() const;

    /// JSON object summary (no per-record detail).
    [[nodiscard]] std::string to_json() const;
};

// ---------------------------------------------------------------------------
// TokenBacktester
// ---------------------------------------------------------------------------

/**
 * @brief Backtests token-sentiment signals against historical OHLCV price data.
 */
class TokenBacktester {
public:
    // -----------------------------------------------------------------------
    // Configuration
    // -----------------------------------------------------------------------

    struct Config {
        /**
         * @brief Maximum allowable timestamp difference (nanoseconds) between
         * a signal and its matched bar.  Default 100 ms = 100_000_000 ns.
         */
        int64_t match_tolerance_ns{100'000'000LL};

        /**
         * @brief Transaction cost per unit of |bias|.
         * Deducted from the gross return.  Default 0.0001 (1 bp per unit).
         */
        double spread_cost{0.0001};

        /**
         * @brief If true, skip signals whose confidence is below
         * `min_confidence`.  Default false.
         */
        bool   apply_confidence_filter{false};

        /**
         * @brief Minimum confidence threshold when `apply_confidence_filter`
         * is true.  Default 0.5.
         */
        double min_confidence{0.5};
    };

    // -----------------------------------------------------------------------
    // Construction
    // -----------------------------------------------------------------------

    explicit TokenBacktester(Config cfg = Config{});

    // Not copyable (owns price history and signal vectors).
    TokenBacktester(const TokenBacktester&)            = delete;
    TokenBacktester& operator=(const TokenBacktester&) = delete;

    // -----------------------------------------------------------------------
    // Data loading
    // -----------------------------------------------------------------------

    /**
     * @brief Load OHLCV price bars from a CSV file.
     *
     * Expected columns (header line or no header):
     *   `timestamp_ms,open,high,low,close,volume`
     *
     * Lines starting with `#` and blank lines are skipped.
     * The first non-comment line is treated as the header if it contains a
     * non-numeric first field.
     *
     * @param path Path to the CSV file.
     * @return true on success; false if the file cannot be opened or parsed.
     */
    [[nodiscard]] bool load_price_history(const std::string& path);

    /**
     * @brief Replay historical signals from a CSV/TSV log file.
     *
     * Expected columns:
     *   `signal_time_ns,direction,bias,confidence`
     *
     * @param signal_log Path to the signal log file.
     * @return true on success; false if the file cannot be opened or parsed.
     */
    [[nodiscard]] bool replay_signals(const std::string& signal_log);

    /**
     * @brief Load price bars from an in-memory vector (alternative to CSV).
     */
    void load_price_history(std::vector<HistoricalBar> bars);

    /**
     * @brief Load signal entries from an in-memory vector.
     */
    void replay_signals(std::vector<SignalEntry> signals);

    // -----------------------------------------------------------------------
    // Backtest execution
    // -----------------------------------------------------------------------

    /**
     * @brief Run the backtest and return the result.
     *
     * Matches each signal to the nearest bar (within ±`match_tolerance_ns`),
     * computes per-signal simulated returns, and aggregates statistics.
     *
     * Requires `load_price_history()` and `replay_signals()` to have been
     * called first.  Can be called multiple times; each call returns an
     * independent result.
     *
     * @return BacktestResult with full per-signal detail and aggregate stats.
     */
    [[nodiscard]] BacktestResult compute_pnl() const;

    // -----------------------------------------------------------------------
    // Convenience
    // -----------------------------------------------------------------------

    /**
     * @brief Load price bars and signal log from file paths, then run.
     *
     * @param price_csv  Path to the OHLCV CSV.
     * @param signal_log Path to the signal log.
     * @return BacktestResult.  `total_bars == -1` on price file error;
     *         `total_signals == -1` on signal file error.
     */
    [[nodiscard]] BacktestResult run(const std::string& price_csv,
                                     const std::string& signal_log);

    /// Number of loaded price bars.
    [[nodiscard]] int bar_count()    const noexcept {
        return static_cast<int>(bars_.size());
    }

    /// Number of loaded signal entries.
    [[nodiscard]] int signal_count() const noexcept {
        return static_cast<int>(signals_.size());
    }

    /// Current configuration.
    [[nodiscard]] const Config& config() const noexcept { return cfg_; }

    /// Update configuration.
    void update_config(const Config& cfg) { cfg_ = cfg; }

private:
    Config                   cfg_;
    std::vector<HistoricalBar> bars_;
    std::vector<SignalEntry>   signals_;

    // Find the bar index closest to `signal_ns` within tolerance.
    // Returns -1 if none found.
    [[nodiscard]] int match_bar(int64_t signal_ns) const noexcept;

    // Aggregate statistics from per-signal results.
    static BacktestResult aggregate(std::vector<BacktestResult::SignalResult> records,
                                    int total_bars,
                                    int total_signals);
};

} // namespace llmquant
