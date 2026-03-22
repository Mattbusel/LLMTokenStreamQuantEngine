#pragma once

#include <atomic>
#include <cstdint>
#include <functional>
#include <mutex>
#include <string>

namespace llmquant {

/**
 * @brief Bid-ask spread estimator that gates signals by predicted edge vs cost.
 *
 * In live trading the true cost of each order is not just commission — it
 * includes half the bid-ask spread (the cost of crossing the spread) plus any
 * price impact.  Signals with predicted edge smaller than this cost will lose
 * money in expectation even if the directional prediction is correct.
 *
 * This filter:
 * 1. Estimates the effective half-spread from OMS fill data using an
 *    exponentially-weighted moving average.
 * 2. Estimates price impact as a linear fraction of the trade value.
 * 3. Computes the total entry cost: cost = half_spread + impact_fraction * value.
 * 4. Returns a pass/block decision in should_trade() based on whether the
 *    caller's predicted edge exceeds cost * min_edge_multiplier.
 *
 * ## Fill-based spread estimation
 * For each OMS fill the half-spread is approximated as:
 *   |fill_price - mid_price|   (requires mid_price from the caller)
 *
 * If no mid_price is available, use the signed fill vs arrival price:
 *   half_spread = |fill_price - arrival_price|  (buy: fill > arrival, sell: fill < arrival)
 *
 * ## Feature flag
 * Controlled by `LLMQUANT_ENABLE_MICROSTRUCTURE_FILTER` (default ON).
 */
class MarketMicrostructureFilter {
public:
    /**
     * @brief Decision returned by should_trade().
     */
    enum class Decision { Pass, Block };

    /**
     * @brief Configuration.
     */
    struct Config {
        /**
         * @brief Decay factor for the EWMA spread estimate.
         *
         * Larger values (closer to 1) give more weight to historical data.
         * Default 0.95.
         */
        double ewma_decay{0.95};

        /**
         * @brief Initial half-spread estimate (absolute price units).
         *
         * Used before any fills are recorded.  Default 0.01 (1 cent).
         */
        double initial_half_spread{0.01};

        /**
         * @brief Linear price-impact fraction per unit of trade value.
         *
         * impact_cost = impact_fraction * abs(trade_value).
         * Default 0.0001 (1 bp).
         */
        double impact_fraction{0.0001};

        /**
         * @brief Minimum edge-to-cost ratio required to pass.
         *
         * edge ≥ total_cost * min_edge_multiplier → Pass.
         * Default 1.5 (must predict at least 1.5× the trading cost as edge).
         */
        double min_edge_multiplier{1.5};

        /**
         * @brief Callback fired each time a signal is blocked.
         *
         * Parameters: (predicted_edge, estimated_cost).
         */
        std::function<void(double edge, double cost)> on_blocked;
    };

    // -----------------------------------------------------------------------
    // Construction
    // -----------------------------------------------------------------------

    explicit MarketMicrostructureFilter(Config config = Config{});

    // -----------------------------------------------------------------------
    // Update
    // -----------------------------------------------------------------------

    /**
     * @brief Record an OMS fill to update the spread estimate.
     *
     * @param arrival_price  Mid/quote price at signal time (before order entry).
     * @param fill_price     Actual execution price.
     * @param is_buy         True for a buy fill, false for a sell.
     *
     * Thread-safe.
     */
    void record_fill(double arrival_price, double fill_price, bool is_buy);

    /**
     * @brief Reset spread estimate to initial_half_spread.
     *
     * Thread-safe.
     */
    void reset();

    // -----------------------------------------------------------------------
    // Gate
    // -----------------------------------------------------------------------

    /**
     * @brief Decide whether to proceed with a trade.
     *
     * @param predicted_edge  Estimated profit per unit (e.g. expected P&L per share).
     * @param trade_value     Absolute value of the planned order (price × quantity).
     * @return Pass if edge ≥ total_cost * min_edge_multiplier, else Block.
     *
     * Thread-safe.
     */
    [[nodiscard]] Decision should_trade(double predicted_edge, double trade_value) const;

    // -----------------------------------------------------------------------
    // Query
    // -----------------------------------------------------------------------

    /**
     * @brief Current EWMA half-spread estimate.
     *
     * Thread-safe.
     */
    [[nodiscard]] double estimated_half_spread() const noexcept;

    /**
     * @brief Estimated total cost for a given trade value.
     *
     * cost = half_spread + impact_fraction * trade_value.
     * Thread-safe.
     */
    [[nodiscard]] double estimated_cost(double trade_value) const noexcept;

    // -----------------------------------------------------------------------
    // Statistics
    // -----------------------------------------------------------------------

    /** @brief Total fills recorded. */
    [[nodiscard]] uint64_t total_fills() const noexcept {
        return total_fills_.load(std::memory_order_relaxed);
    }

    /** @brief Total times should_trade() returned Block. */
    [[nodiscard]] uint64_t total_blocked() const noexcept {
        return total_blocked_.load(std::memory_order_relaxed);
    }

    /** @brief Total times should_trade() returned Pass. */
    [[nodiscard]] uint64_t total_passed() const noexcept {
        return total_passed_.load(std::memory_order_relaxed);
    }

    /**
     * @brief Update configuration; resets spread estimate.
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
    double half_spread_{0.01};  // current EWMA estimate

    std::atomic<uint64_t> total_fills_{0};
    std::atomic<uint64_t> total_blocked_{0};
    std::atomic<uint64_t> total_passed_{0};
};

} // namespace llmquant
