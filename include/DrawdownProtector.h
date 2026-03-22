#pragma once

#include <atomic>
#include <chrono>
#include <cstdint>
#include <functional>
#include <mutex>
#include <string>

namespace llmquant {

/**
 * @brief Dynamic risk tightening as cumulative losses grow.
 *
 * The DrawdownProtector monitors a running P&L series and progressively
 * tightens downstream risk parameters as the portfolio moves further from
 * its high-water mark.  It is the complement of KellyPositionSizer: Kelly
 * sizes the *upside* (grow when winning); DrawdownProtector protects the
 * *downside* (shrink when losing).
 *
 * ## Tiers
 * Four configurable drawdown tiers each carry a `scale_factor ∈ (0, 1]`
 * that is multiplied into the caller-supplied base confidence threshold and
 * max_bias_magnitude before they are applied to the pipeline:
 *
 *   Tier 0 (healthy): drawdown < tier1_pct  → scale = 1.0 (no change)
 *   Tier 1 (caution): drawdown ≥ tier1_pct  → scale = tier1_scale
 *   Tier 2 (warning): drawdown ≥ tier2_pct  → scale = tier2_scale
 *   Tier 3 (critical): drawdown ≥ tier3_pct → scale = tier3_scale (≈ 0)
 *
 * ## Usage
 * @code
 * DrawdownProtector protector;
 * // After each trade outcome:
 * protector.record_pnl(trade_pnl);
 * // Before risk evaluation:
 * auto [conf_scale, bias_scale] = protector.current_scale();
 * risk_cfg.min_confidence   = base_conf   / conf_scale;  // tighten
 * risk_cfg.max_bias_magnitude = base_mag  * bias_scale;  // shrink
 * @endcode
 *
 * ## Feature flag
 * Controlled by `LLMQUANT_ENABLE_DRAWDOWN_PROTECTOR` (default ON).
 */
class DrawdownProtector {
public:
    /**
     * @brief Tier configuration.
     */
    struct Tier {
        double drawdown_pct{0.0};  ///< Drawdown depth (fraction, 0–1) to enter this tier.
        double scale{1.0};         ///< Multiplier applied to risk parameters in this tier.
    };

    /**
     * @brief Configuration.
     */
    struct Config {
        /**
         * @brief Tier thresholds (applied in ascending drawdown_pct order).
         *
         * Defaults: 5%/0.7, 10%/0.4, 20%/0.1 drawdown depths.
         */
        Tier tier1{0.05, 0.7};   ///< Caution.
        Tier tier2{0.10, 0.4};   ///< Warning.
        Tier tier3{0.20, 0.1};   ///< Critical.

        /**
         * @brief Recovery threshold: fraction of the drawdown that must be
         *        recovered before stepping back up to the previous tier.
         *
         * Hysteresis prevents rapid oscillation at tier boundaries.
         * Default 0.5 (recover half the drawdown to step back up).
         */
        double recovery_fraction{0.5};

        /**
         * @brief Callback invoked when the active tier changes.
         *
         * Parameters: (new_tier_index 0–3, current_scale, current_drawdown_pct).
         * Called from the thread invoking record_pnl() or reset().
         */
        std::function<void(int, double, double)> on_tier_change;
    };

    /**
     * @brief Scales to apply to risk parameters in the current tier.
     */
    struct ScalePair {
        double confidence_scale{1.0};    ///< Multiplier for min_confidence threshold.
        double bias_magnitude_scale{1.0};///< Multiplier for max_bias_magnitude.
    };

    // -----------------------------------------------------------------------
    // Construction
    // -----------------------------------------------------------------------

    explicit DrawdownProtector(Config config = Config{});

    // -----------------------------------------------------------------------
    // P&L recording
    // -----------------------------------------------------------------------

    /**
     * @brief Record a new P&L observation (positive = profit, negative = loss).
     *
     * Updates the high-water mark, computes current drawdown, selects the
     * appropriate tier, and fires on_tier_change if the tier has changed.
     * Thread-safe.
     *
     * @param pnl Incremental P&L for this trade/signal.
     */
    void record_pnl(double pnl);

    /**
     * @brief Reset the P&L series to zero (e.g. start of new trading session).
     *
     * Resets cumulative_pnl, high_water_mark, and current tier to 0.
     * Thread-safe.
     */
    void reset();

    // -----------------------------------------------------------------------
    // Query
    // -----------------------------------------------------------------------

    /**
     * @brief Return the scale pair for the current drawdown tier.
     *
     * Thread-safe; non-blocking.
     */
    [[nodiscard]] ScalePair current_scale() const noexcept;

    /**
     * @brief Current drawdown as a fraction of the high-water mark (0 = no drawdown).
     *
     * Returns 0 if the high-water mark is ≤ 0 (no profits seen yet).
     */
    [[nodiscard]] double current_drawdown_pct() const noexcept;

    /**
     * @brief Active tier index: 0 = healthy, 1–3 = caution/warning/critical.
     */
    [[nodiscard]] int current_tier() const noexcept {
        return current_tier_.load(std::memory_order_relaxed);
    }

    /**
     * @brief Running cumulative P&L across all record_pnl() calls since last reset().
     */
    [[nodiscard]] double cumulative_pnl() const noexcept;

    /**
     * @brief Highest cumulative P&L seen since last reset().
     */
    [[nodiscard]] double high_water_mark() const noexcept;

    /**
     * @brief Number of tier transitions since construction or last reset().
     */
    [[nodiscard]] uint64_t tier_transitions() const noexcept {
        return tier_transitions_.load(std::memory_order_relaxed);
    }

    // -----------------------------------------------------------------------
    // Runtime configuration
    // -----------------------------------------------------------------------

    /**
     * @brief Update configuration at runtime.
     *
     * Thread-safe.  The current P&L series is preserved.
     */
    void update_config(const Config& config);

    /**
     * @brief Serialise current state to a JSON object string.
     */
    [[nodiscard]] std::string to_stats_json() const;

private:
    mutable std::mutex mutex_;
    Config config_;

    double cumulative_pnl_{0.0};
    double high_water_mark_{0.0};
    std::atomic<int>      current_tier_{0};
    std::atomic<uint64_t> tier_transitions_{0};

    [[nodiscard]] static double scale_for_tier(int tier,
                                                const Config& cfg) noexcept;
    [[nodiscard]] static int compute_tier(double drawdown_pct,
                                           const Config& cfg) noexcept;
};

} // namespace llmquant
