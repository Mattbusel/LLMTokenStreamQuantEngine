#pragma once

#include <atomic>
#include <chrono>
#include <cstdint>
#include <functional>
#include <mutex>
#include <string>
#include <vector>

namespace llmquant {

/**
 * @brief Multi-horizon signal fusion: aggregates bias values across time buckets.
 *
 * Real trading systems observe that sentiment signals on different timescales
 * carry complementary information.  A 1-second micro-signal captures immediacy
 * (breaking news token burst) while a 5-minute macro-signal captures sustained
 * directional pressure.  The MultiTimeframeAggregator maintains a set of named
 * "timeframes", each with a rolling EMA of incoming bias values.  It exposes a
 * *consensus* signal that is a configurable weighted blend of all timeframes.
 *
 * ## Typical usage
 * @code
 * MultiTimeframeAggregator::Config cfg;
 * cfg.timeframes = {
 *     {  "1s",  1.0, 1.0},   // 1s horizon, EMA alpha=1.0 (no smoothing), weight=1
 *     {  "5s",  0.2, 1.5},   // 5s horizon, heavier weight
 *     { "30s",  0.05, 2.0},  // 30s horizon
 *     {  "5m",  0.01, 1.0},  // 5-minute macro view
 * };
 * MultiTimeframeAggregator mta(cfg);
 *
 * // In signal callback:
 * mta.record(signal.delta_bias_shift);
 * double consensus = mta.consensus();
 * // Use consensus to gate or scale the signal.
 * @endcode
 *
 * ## EMA update
 * Each timeframe maintains: `ema = alpha * value + (1 - alpha) * ema`.
 *
 * ## Consensus
 * `consensus = Σ(weight_i * ema_i) / Σ(weight_i)`
 *
 * ## Divergence detection
 * The spread between the fastest and slowest EMAs is exposed as
 * `timeframe_spread()`.  Large spread → short- and long-term signals disagree.
 *
 * ## Feature flag
 * Controlled by `LLMQUANT_ENABLE_MULTI_TIMEFRAME` (default ON).
 */
class MultiTimeframeAggregator {
public:
    /**
     * @brief Configuration for a single timeframe.
     */
    struct Timeframe {
        std::string name;        ///< Human-readable label (e.g. "1s", "5m").
        double      ema_alpha;   ///< EMA smoothing factor ∈ (0, 1].  1.0 = no smoothing.
        double      weight{1.0}; ///< Blend weight in the consensus calculation.
    };

    /**
     * @brief Configuration.
     */
    struct Config {
        /**
         * @brief Ordered list of timeframes.
         *
         * At least one must be provided; if empty the aggregator is a no-op.
         * Default: four timeframes at 1s/5s/30s/5m (simulated by EMA alphas).
         */
        std::vector<Timeframe> timeframes{
            {"1s",  1.00, 1.0},
            {"5s",  0.20, 1.5},
            {"30s", 0.05, 2.0},
            {"5m",  0.01, 1.0},
        };

        /**
         * @brief Divergence callback: fires when |fastest_ema - slowest_ema|
         *        exceeds this threshold.  0 = disabled.
         * Default 0.3.
         */
        double diverge_threshold{0.3};

        /**
         * @brief Callback invoked when a divergence event is detected.
         *
         * Parameters: (spread, fastest_ema, slowest_ema).
         * Called from the thread invoking record().  Must not block.
         */
        std::function<void(double, double, double)> on_divergence;
    };

    // -----------------------------------------------------------------------
    // Construction
    // -----------------------------------------------------------------------

    explicit MultiTimeframeAggregator(Config config = Config{});

    // -----------------------------------------------------------------------
    // Recording
    // -----------------------------------------------------------------------

    /**
     * @brief Push a new bias value into all timeframe EMAs.
     *
     * Thread-safe.
     *
     * @param bias_value  Scalar bias (delta_bias_shift from TradeSignal).
     */
    void record(double bias_value);

    /**
     * @brief Reset all EMAs to zero (e.g. on session open).
     *
     * Thread-safe.
     */
    void reset();

    // -----------------------------------------------------------------------
    // Query
    // -----------------------------------------------------------------------

    /**
     * @brief Weighted consensus of all timeframe EMAs.
     *
     * Returns 0 if no data has been recorded or if no timeframes are defined.
     * Thread-safe.
     */
    [[nodiscard]] double consensus() const noexcept;

    /**
     * @brief EMA value for a named timeframe.
     *
     * Returns 0 if the name is not found.  Thread-safe.
     */
    [[nodiscard]] double ema(const std::string& name) const noexcept;

    /**
     * @brief |fastest_ema − slowest_ema|: indicates short/long disagreement.
     *
     * Thread-safe.
     */
    [[nodiscard]] double timeframe_spread() const noexcept;

    /**
     * @brief True if the most recent record() call triggered a divergence event.
     */
    [[nodiscard]] bool is_diverging() const noexcept {
        return diverging_.load(std::memory_order_relaxed);
    }

    // -----------------------------------------------------------------------
    // Statistics
    // -----------------------------------------------------------------------

    /** @brief Total calls to record(). */
    [[nodiscard]] uint64_t total_records() const noexcept {
        return total_records_.load(std::memory_order_relaxed);
    }

    /** @brief Number of divergence events fired. */
    [[nodiscard]] uint64_t divergence_events() const noexcept {
        return divergence_events_.load(std::memory_order_relaxed);
    }

    /**
     * @brief Serialise all EMA values and stats to a JSON string.
     */
    [[nodiscard]] std::string to_stats_json() const;

    // -----------------------------------------------------------------------
    // Runtime configuration
    // -----------------------------------------------------------------------

    /**
     * @brief Update configuration; resets all EMAs.
     *
     * Thread-safe.
     */
    void update_config(const Config& config);

private:
    mutable std::mutex mutex_;
    Config config_;

    // Per-timeframe EMA state (parallel to config_.timeframes).
    std::vector<double> emas_;

    std::atomic<bool>     diverging_{false};
    std::atomic<uint64_t> total_records_{0};
    std::atomic<uint64_t> divergence_events_{0};
};

} // namespace llmquant
