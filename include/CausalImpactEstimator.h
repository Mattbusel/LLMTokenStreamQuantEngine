#pragma once

#include <atomic>
#include <cstdint>
#include <functional>
#include <mutex>
#include <string>
#include <vector>

namespace llmquant {

/**
 * @brief Real-time CUSUM-based causal impact detector for sentiment events.
 *
 * Standard correlation between sentiment and returns is spurious: it does not
 * tell us *when* a narrative event actually caused a return shift.  This class
 * uses a Page-Hinkley / CUSUM detector to identify statistically significant
 * *structural breaks* in a returns series, then attributes them to the nearest
 * preceding sentiment event.
 *
 * ## Algorithm
 * 1. `record_return(r)` — append a return observation and update running statistics.
 * 2. CUSUM test statistic: S_t = max(0, S_{t-1} + (r - mu − delta))
 *    where mu = baseline mean return, delta = detection sensitivity.
 * 3. When S_t > h (detection threshold), a structural break is declared.
 * 4. If a sentinel event was recorded within the `event_lookback` window before
 *    the break, it is attributed as the probable cause.
 *
 * The class does NOT need historical price data or a causal inference library;
 * it uses only online statistics and is O(1) per observation.
 *
 * ## Feature flag
 * Controlled by `LLMQUANT_ENABLE_CAUSAL_IMPACT` (default ON).
 *
 * Thread safety: all public methods are thread-safe.
 */
class CausalImpactEstimator {
public:
    // -----------------------------------------------------------------------
    // Configuration
    // -----------------------------------------------------------------------

    struct Config {
        /**
         * @brief Baseline mean return (μ₀).
         *
         * Estimated online if set to 0 (default): the CUSUM uses the rolling
         * mean for the first `warmup_window` observations.
         */
        double baseline_mean{0.0};

        /**
         * @brief CUSUM sensitivity (δ).
         *
         * Smaller δ = more sensitive to small deviations.  Default 0.001.
         */
        double sensitivity{0.001};

        /**
         * @brief CUSUM decision threshold (h).
         *
         * A break is declared when S_t > h.  Default 0.05.
         */
        double threshold{0.05};

        /**
         * @brief Observations window for computing the rolling baseline.
         *
         * Default 50.
         */
        int warmup_window{50};

        /**
         * @brief Maximum lookback (in return observations) to attribute a
         * structural break to a preceding sentiment event.
         *
         * Default 20.
         */
        int event_lookback{20};

        /**
         * @brief Fired when a structural break is detected.
         *
         * Parameters: (cusum_stat, attributed_event_label, impact_estimate).
         * `attributed_event_label` is empty if no event falls in the lookback.
         * `impact_estimate` = (post-break mean − baseline) over the break window.
         * Called outside the internal lock.
         */
        std::function<void(double cusum_stat,
                           const std::string& event_label,
                           double impact_estimate)> on_break;
    };

    // -----------------------------------------------------------------------
    // Construction
    // -----------------------------------------------------------------------

    explicit CausalImpactEstimator(Config cfg = Config{});

    // -----------------------------------------------------------------------
    // Data intake
    // -----------------------------------------------------------------------

    /**
     * @brief Record a return observation.
     *
     * @param ret  Period return (e.g. 5-second mark-to-market P&L change).
     * @return Current CUSUM statistic (≥ 0).
     */
    double record_return(double ret);

    /**
     * @brief Record a sentiment event sentinel.
     *
     * Events are timestamped by observation index (not wall clock) and matched
     * against subsequent structural breaks within `event_lookback` steps.
     *
     * @param label  Human-readable event description (e.g. "earnings miss").
     */
    void record_event(const std::string& label);

    // -----------------------------------------------------------------------
    // Accessors
    // -----------------------------------------------------------------------

    /**
     * @brief Current CUSUM statistic S_t ∈ [0, ∞).
     */
    [[nodiscard]] double cusum_stat() const noexcept {
        return cusum_.load(std::memory_order_relaxed);
    }

    /**
     * @brief True when CUSUM stat exceeds the detection threshold.
     */
    [[nodiscard]] bool break_detected() const noexcept {
        return break_detected_.load(std::memory_order_relaxed);
    }

    /**
     * @brief Total structural breaks detected.
     */
    [[nodiscard]] uint64_t break_count() const noexcept {
        return break_count_.load(std::memory_order_relaxed);
    }

    /**
     * @brief Total return observations recorded.
     */
    [[nodiscard]] uint64_t observation_count() const noexcept {
        return obs_count_.load(std::memory_order_relaxed);
    }

    /**
     * @brief Online baseline mean estimate.
     */
    [[nodiscard]] double baseline_mean_estimate() const noexcept;

    // -----------------------------------------------------------------------
    // Control
    // -----------------------------------------------------------------------

    /**
     * @brief Reset CUSUM state; retain configuration and baseline estimate.
     */
    void reset_cusum();

    /**
     * @brief Full reset: clear all statistics and event history.
     */
    void reset();

    /**
     * @brief Update configuration without resetting.
     */
    void update_config(const Config& cfg);

    /**
     * @brief JSON diagnostics snapshot.
     */
    [[nodiscard]] std::string to_stats_json() const;

private:
    struct EventEntry {
        std::string label;
        uint64_t    obs_index;  // observation index when event was recorded
    };

    Config cfg_;

    // CUSUM state.
    double   cusum_s_{0.0};
    double   mu_hat_{0.0};     // online mean estimate
    int      n_obs_{0};         // observations seen for baseline
    bool     baseline_locked_{false};  // true once warmup_window passed

    // Rolling returns for post-break impact estimation.
    std::vector<double> return_window_;
    int  return_head_{0};
    int  return_fill_{0};

    // Event sentinel history.
    std::vector<EventEntry> events_;

    std::atomic<double>   cusum_{0.0};
    std::atomic<bool>     break_detected_{false};
    std::atomic<uint64_t> break_count_{0};
    std::atomic<uint64_t> obs_count_{0};

    mutable std::mutex mutex_;

    [[nodiscard]] std::string find_recent_event_locked(uint64_t obs_index) const;
    [[nodiscard]] double post_break_impact_locked() const;
};

} // namespace llmquant
