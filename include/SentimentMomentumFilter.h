#pragma once

#include <atomic>
#include <cstdint>
#include <mutex>
#include <string>

#include "SentimentTrajectoryAnalyzer.h"
#include "TradeSignalEngine.h"

namespace llmquant {

/**
 * @brief Cross-validates trade signals against the sentiment trajectory
 *        to reduce false positives from noisy single-token spikes.
 *
 * A bullish signal (positive delta_bias_shift) is only passed when the
 * SentimentTrajectoryAnalyzer reports Improving or Stable trend.
 * A bearish signal (negative delta_bias_shift) is only passed when the
 * trend is Declining or Stable.
 * Signals are suppressed when the trajectory is Volatile or Undefined.
 *
 * This directly couples the micro-level signal strength (single token)
 * with the macro-level sentiment momentum (sliding window regression),
 * cutting false positives caused by transient sentiment spikes that
 * immediately reverse.
 *
 * ## Modes
 * - `Strict`:   Bullish only on Improving; Bearish only on Declining.
 * - `Relaxed`:  Bullish on Improving|Stable; Bearish on Declining|Stable.
 * - `Disabled`: All signals pass (filter is a no-op — useful for A/B testing).
 *
 * ## Feature flag
 * Controlled by `LLMQUANT_ENABLE_SENTIMENT_MOMENTUM_FILTER` (default ON).
 *
 * ## Thread safety
 * `filter_signal()` is thread-safe (reads analyzer via its thread-safe API).
 * `record_sample()` delegates directly to the underlying analyzer.
 */
class SentimentMomentumFilter {
public:
    /**
     * @brief Filtering mode controlling how strictly trajectory must align
     *        with signal direction.
     */
    enum class Mode {
        Strict,   ///< Bullish only on Improving; Bearish only on Declining.
        Relaxed,  ///< Bullish on Improving|Stable; Bearish on Declining|Stable.
        Disabled, ///< Pass-through — all signals allowed regardless of trend.
    };

    /**
     * @brief Configuration for the momentum filter.
     */
    struct Config {
        /**
         * @brief Filtering mode.
         */
        Mode mode{Mode::Relaxed};

        /**
         * @brief If true, scale `delta_bias_shift` by the trajectory's
         *        normalized slope magnitude rather than blocking/passing
         *        the signal binary.  Only applies in Relaxed or Strict modes.
         *
         * When enabled the bias is multiplied by `|slope| / slope_scale`
         * (clamped to [0, 1]), producing a soft attenuation instead of a
         * hard gate.  `slope_scale` should be set to the "full strength"
         * slope value (e.g. 0.05).
         */
        bool scale_by_slope{false};

        /**
         * @brief Slope value that maps to a 1.0 (full-strength) scale factor.
         * Ignored when scale_by_slope is false.
         */
        double slope_scale{0.05};

        /**
         * @brief Analyzer config forwarded at construction.
         *
         * Allows single-point configuration of both the filter and the
         * underlying trajectory window.
         */
        SentimentTrajectoryAnalyzer::Config analyzer;
    };

    /**
     * @brief Statistics snapshot.
     */
    struct Stats {
        uint64_t signals_in{0};      ///< Total calls to filter_signal().
        uint64_t signals_passed{0};  ///< Signals that were not suppressed.
        uint64_t signals_blocked{0}; ///< Signals suppressed by the filter.
        uint64_t signals_scaled{0};  ///< Signals attenuated by slope scaling.
        double   pass_rate{0.0};     ///< signals_passed / signals_in.
    };

    /**
     * @brief Construct with the given configuration.
     *
     * @param config Filter and analyzer parameters.
     */
    explicit SentimentMomentumFilter(Config config = Config{});

    /**
     * @brief Feed a new sentiment sample into the underlying trajectory window.
     *
     * Call this from the same token path that feeds TradeSignalEngine.
     * Thread-safe (delegates to SentimentTrajectoryAnalyzer).
     *
     * @param sentiment_score Raw sentiment score for the current token.
     */
    void record_sample(double sentiment_score);

    /**
     * @brief Gate or scale a signal based on the current sentiment trend.
     *
     * Returns the signal unchanged if it aligns with the trend (or mode
     * is Disabled).  Returns a signal with `delta_bias_shift = 0.0` if
     * the signal is suppressed.  In scale_by_slope mode, attenuates the
     * bias proportionally to the slope magnitude.
     *
     * Thread-safe.
     *
     * @param signal Raw TradeSignal from the engine callback.
     * @return Filtered (or pass-through) copy of the signal.
     */
    [[nodiscard]] TradeSignal filter_signal(const TradeSignal& signal);

    /**
     * @brief Return the current trend classification from the underlying analyzer.
     *
     * Thread-safe.
     */
    [[nodiscard]] SentimentTrend current_trend() const;

    /**
     * @brief Return the current regression slope from the underlying analyzer.
     *
     * Thread-safe.
     */
    [[nodiscard]] double current_slope() const;

    /**
     * @brief Return accumulated filter statistics.
     *
     * Thread-safe.
     */
    [[nodiscard]] Stats get_stats() const;

    /**
     * @brief Serialise current state to a compact JSON string.
     *
     * Fields: mode, trend, slope, signals_in, signals_passed, signals_blocked,
     * pass_rate.
     * Thread-safe.
     */
    [[nodiscard]] std::string to_stats_json() const;

    /**
     * @brief Reset the trajectory window and all filter statistics.
     *
     * Thread-safe.
     */
    void reset();

    /**
     * @brief Update the filter configuration at runtime.
     *
     * Does not reset the trajectory window.
     * Thread-safe.
     *
     * @param config New configuration.
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
    Config config_;
    SentimentTrajectoryAnalyzer analyzer_;

    std::atomic<uint64_t> signals_in_{0};
    std::atomic<uint64_t> signals_passed_{0};
    std::atomic<uint64_t> signals_blocked_{0};
    std::atomic<uint64_t> signals_scaled_{0};
};

} // namespace llmquant
