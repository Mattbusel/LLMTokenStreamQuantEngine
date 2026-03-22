#pragma once

#include "TradeSignalEngine.h"

#include <atomic>
#include <cstdint>
#include <functional>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

namespace llmquant {

/**
 * @brief Retrospective P&L attribution across sentiment drivers.
 *
 * Closes the signal→outcome feedback loop by recording realized P&L
 * alongside the state of each major signal driver at trade time.
 *
 * For every completed trade you call record_outcome(signal, pnl).
 * The engine extracts categorical driver labels from the signal fields:
 *
 *   - **Bias direction** (bullish / bearish / neutral)
 *   - **Confidence band** (low <0.4, medium 0.4–0.7, high >0.7)
 *   - **Volatility regime** (calm / elevated / extreme)
 *   - **Signal quality band** (poor <0.5, fair 0.5–0.75, good >0.75)
 *
 * Per-driver running totals are kept: trade count, winning trades, and
 * cumulative P&L.  From these, win-rate and average P&L per trade are
 * derived on demand.
 *
 * ## Usage
 * ```cpp
 * PnLAttributionEngine attr;
 * // After each fill:
 * attr.record_outcome(last_signal, realized_pnl);
 * // Periodically:
 * auto top = attr.top_drivers(5);
 * spdlog::info("[pnl-attr] top driver: {} win_rate={:.0f}%",
 *              top[0].label, top[0].win_rate * 100.0);
 * ```
 *
 * ## Thread safety
 * record_outcome() is thread-safe.
 *
 * ## Feature flag
 * Controlled by `LLMQUANT_ENABLE_PNL_ATTRIBUTION` (default ON).
 */
class PnLAttributionEngine {
public:
    /**
     * @brief Attribution stats for one driver category.
     */
    struct DriverStats {
        std::string label;          ///< Human-readable driver label.
        uint64_t    trade_count{0}; ///< Total trades attributed to this driver.
        uint64_t    win_count{0};   ///< Trades where pnl > 0.
        double      total_pnl{0.0}; ///< Cumulative P&L for this driver.

        /** @brief Win rate [0,1].  Returns 0 if no trades recorded. */
        [[nodiscard]] double win_rate() const noexcept {
            return (trade_count > 0)
                ? static_cast<double>(win_count) / static_cast<double>(trade_count)
                : 0.0;
        }

        /** @brief Average P&L per trade.  Returns 0 if no trades recorded. */
        [[nodiscard]] double avg_pnl() const noexcept {
            return (trade_count > 0)
                ? total_pnl / static_cast<double>(trade_count)
                : 0.0;
        }
    };

    using OutcomeCallback = std::function<void(const DriverStats&)>;

    /**
     * @brief Configuration.
     */
    struct Config {
        /**
         * @brief Bias threshold below which the signal is classified "neutral".
         *
         * |bias| < neutral_band → neutral; >= → bullish or bearish.
         */
        double neutral_band{0.05};

        /** @brief Confidence threshold separating low from medium band. */
        double conf_medium_threshold{0.40};
        /** @brief Confidence threshold separating medium from high band. */
        double conf_high_threshold{0.70};

        /** @brief Volatility adjustment separating calm from elevated. */
        double vol_elevated_threshold{0.015};
        /** @brief Volatility adjustment separating elevated from extreme. */
        double vol_extreme_threshold{0.040};

        /** @brief Signal quality separating poor from fair. */
        double sq_fair_threshold{0.50};
        /** @brief Signal quality separating fair from good. */
        double sq_good_threshold{0.75};

        /**
         * @brief Optional callback fired after each record_outcome() call.
         *
         * Receives the *combined* composite driver stats for this trade.
         */
        OutcomeCallback on_outcome;
    };

    explicit PnLAttributionEngine(Config cfg = Config{});

    // Non-copyable.
    PnLAttributionEngine(const PnLAttributionEngine&)            = delete;
    PnLAttributionEngine& operator=(const PnLAttributionEngine&) = delete;

    /**
     * @brief Record a completed trade outcome.
     *
     * Extracts driver labels from signal and attributes pnl to each.
     * Thread-safe.
     *
     * @param signal     The TradeSignal that triggered the trade.
     * @param realized_pnl Realized P&L for this trade (positive = profit).
     */
    void record_outcome(const TradeSignal& signal, double realized_pnl);

    /**
     * @brief Return all driver stats, sorted by total_pnl descending.
     */
    [[nodiscard]] std::vector<DriverStats> all_drivers() const;

    /**
     * @brief Return the top N drivers by cumulative P&L.
     *
     * If fewer than N drivers exist, all are returned.
     */
    [[nodiscard]] std::vector<DriverStats> top_drivers(size_t n) const;

    /**
     * @brief Return the bottom N drivers by cumulative P&L (worst performers).
     */
    [[nodiscard]] std::vector<DriverStats> bottom_drivers(size_t n) const;

    /** @brief Total number of trades recorded. */
    [[nodiscard]] uint64_t total_trades() const noexcept {
        return total_trades_.load(std::memory_order_relaxed);
    }

    /** @brief Cumulative P&L across all recorded trades. */
    [[nodiscard]] double total_pnl() const noexcept;

    /**
     * @brief Reset all attribution stats.
     */
    void reset();

    /**
     * @brief Update configuration at runtime.
     *
     * Resets accumulated stats.
     */
    void update_config(const Config& cfg);

    /**
     * @brief Return current config.
     */
    [[nodiscard]] const Config& get_config() const noexcept { return cfg_; }

    /**
     * @brief Serialise current state to a JSON object string.
     */
    [[nodiscard]] std::string to_stats_json() const;

private:
    [[nodiscard]] std::vector<std::string> classify(const TradeSignal& s) const;

    Config cfg_;
    mutable std::mutex mtx_;

    std::unordered_map<std::string, DriverStats> drivers_;
    std::atomic<uint64_t> total_trades_{0};
    double total_pnl_{0.0};  // protected by mtx_
};

} // namespace llmquant
