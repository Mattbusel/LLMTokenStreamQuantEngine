#pragma once

#include <atomic>
#include <cstdint>
#include <functional>
#include <mutex>
#include <string>

namespace llmquant {

/**
 * @brief Synthetic limit order book driven by token sentiment.
 *
 * Models a simplified LOB where the mid-price and spread respond to
 * accumulated sentiment signal.  Provides estimated fill prices for
 * notional order sizes, enabling slippage-aware position sizing.
 *
 * ## Model
 * - Mid price tracks a reference price, shifted by accumulated bias:
 *     mid = ref_price * (1 + bias_sensitivity * cumulative_bias)
 * - Half-spread = base_spread + impact_coeff * sqrt(|order_size|)
 *     (Almgren-Chriss square-root market impact)
 * - Fill price for a buy:  mid + half_spread
 * - Fill price for a sell: mid - half_spread
 *
 * ## Usage
 * @code
 * OrderBookSimulator lob;
 * lob.update_bias(signal.delta_bias_shift);
 * double fill = lob.estimated_fill_price(side, notional_usd);
 * @endcode
 *
 * ## Feature flag
 * Controlled by `LLMQUANT_ENABLE_ORDER_BOOK_SIM` (default ON).
 */
class OrderBookSimulator {
public:
    /**
     * @brief Order side.
     */
    enum class Side { Buy, Sell };

    /**
     * @brief Configuration.
     */
    struct Config {
        /**
         * @brief Reference (fair-value) mid price.
         *
         * Default 100.0 (generic asset unit).
         */
        double ref_price{100.0};

        /**
         * @brief Sensitivity of mid price to accumulated bias.
         *
         * mid = ref_price * (1 + bias_sensitivity * cumulative_bias).
         * Default 0.01 (1% price move per unit of bias).
         */
        double bias_sensitivity{0.01};

        /**
         * @brief Base half-spread at zero order size (in price units).
         *
         * Default 0.05 (5 bps on a $100 instrument).
         */
        double base_spread{0.05};

        /**
         * @brief Market impact coefficient for sqrt-impact model.
         *
         * half_spread += impact_coeff * sqrt(|order_notional|).
         * Default 0.001.
         */
        double impact_coeff{0.001};

        /**
         * @brief EMA decay for cumulative bias (0 = no decay, 1 = full reset each update).
         *
         * cumulative_bias = (1 - bias_decay) * cumulative_bias + bias_value.
         * Default 0.05.
         */
        double bias_decay{0.05};

        /**
         * @brief Callback fired when estimated spread exceeds spread_alert_threshold.
         *
         * Parameter: current half-spread.
         */
        std::function<void(double)> on_wide_spread;

        /**
         * @brief Half-spread threshold that triggers on_wide_spread.
         *
         * 0 = disabled.  Default 0.5.
         */
        double spread_alert_threshold{0.5};
    };

    // -----------------------------------------------------------------------
    // Construction
    // -----------------------------------------------------------------------

    explicit OrderBookSimulator(Config config = Config{});

    // -----------------------------------------------------------------------
    // Update
    // -----------------------------------------------------------------------

    /**
     * @brief Update cumulative bias from the latest signal delta.
     *
     * Thread-safe.
     *
     * @param delta_bias  delta_bias_shift from TradeSignal.
     */
    void update_bias(double delta_bias);

    /**
     * @brief Reset cumulative bias to zero.
     *
     * Thread-safe.
     */
    void reset();

    // -----------------------------------------------------------------------
    // Query
    // -----------------------------------------------------------------------

    /**
     * @brief Current synthetic mid price.
     *
     * Thread-safe.
     */
    [[nodiscard]] double mid_price() const noexcept;

    /**
     * @brief Estimated half-spread for a given order notional.
     *
     * Thread-safe.
     */
    [[nodiscard]] double half_spread(double order_notional = 0.0) const noexcept;

    /**
     * @brief Estimated fill price for an order.
     *
     * Thread-safe.
     *
     * @param side            Buy or Sell.
     * @param order_notional  Notional size of the order.
     * @return                Estimated fill price (mid ± half_spread).
     */
    [[nodiscard]] double estimated_fill_price(Side side,
                                              double order_notional = 0.0) const noexcept;

    /**
     * @brief Current cumulative bias in the EMA accumulator.
     *
     * Thread-safe.
     */
    [[nodiscard]] double cumulative_bias() const noexcept;

    // -----------------------------------------------------------------------
    // Statistics
    // -----------------------------------------------------------------------

    /** @brief Total bias update calls. */
    [[nodiscard]] uint64_t total_updates() const noexcept {
        return total_updates_.load(std::memory_order_relaxed);
    }

    /** @brief Times on_wide_spread fired. */
    [[nodiscard]] uint64_t wide_spread_events() const noexcept {
        return wide_spread_events_.load(std::memory_order_relaxed);
    }

    /**
     * @brief Update configuration.
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

    double cumulative_bias_{0.0};

    std::atomic<uint64_t> total_updates_{0};
    std::atomic<uint64_t> wide_spread_events_{0};
};

} // namespace llmquant
