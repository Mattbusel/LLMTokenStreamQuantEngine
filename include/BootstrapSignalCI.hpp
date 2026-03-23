#pragma once

#include <cstddef>
#include <cstdint>
#include <functional>
#include <mutex>
#include <string>
#include <vector>

namespace llmquant {

/**
 * @brief Bootstrap confidence interval for signal reliability.
 *
 * Maintains a rolling reservoir of the most recent `window_size` signal
 * observations.  When `compute()` is called it draws `n_bootstrap` re-samples
 * (with replacement) from the reservoir, computes the mean of each re-sample,
 * sorts all re-sample means, and returns the lower percentile, median, and
 * upper percentile cut-points as a `ConfidenceInterval`.
 *
 * This is the non-parametric bootstrap estimate of the distribution of the
 * signal mean.  It makes no Gaussian assumption — useful when signal
 * distributions are skewed or heavy-tailed.
 *
 * ## Usage
 * ```cpp
 * BootstrapSignalCI ci(500, 0.95f);
 * ci.add_observation(signal.delta_bias_shift);
 * if (ci.is_ready()) {
 *     auto interval = ci.compute();
 *     if (interval.lower > 0.0f) // statistically bullish
 *         ...;
 * }
 * ```
 *
 * ## Feature flag
 * Controlled by `LLMQUANT_ENABLE_BOOTSTRAP_SIGNAL_CI` (default ON).
 *
 * Thread safety: all public methods are protected by an internal mutex.
 */
class BootstrapSignalCI {
public:
    // -----------------------------------------------------------------------
    // Result type
    // -----------------------------------------------------------------------

    /**
     * @brief Confidence interval returned by `compute()`.
     *
     * @param lower   Lower percentile cut-point, e.g. 2.5th percentile for 95% CI.
     * @param upper   Upper percentile cut-point, e.g. 97.5th percentile for 95% CI.
     * @param median  Median (50th percentile) of the bootstrap distribution.
     */
    struct ConfidenceInterval {
        float lower  {0.0f};
        float upper  {0.0f};
        float median {0.0f};
    };

    // -----------------------------------------------------------------------
    // Construction
    // -----------------------------------------------------------------------

    /**
     * @brief Construct a bootstrap CI estimator.
     *
     * @param n_bootstrap      Number of bootstrap re-samples drawn per
     *                         `compute()` call (≥ 2).  More re-samples give
     *                         more stable CI estimates at higher cost.
     *                         Default 200.
     * @param confidence_level Nominal coverage of the interval (0, 1).
     *                         E.g. 0.95 → [2.5%, 97.5%] percentile CI.
     *                         Default 0.95.
     * @param window_size      Maximum number of observations retained.
     *                         Older observations are evicted when capacity is
     *                         reached.  Default 128.
     * @param min_observations Minimum number of observations before
     *                         `is_ready()` returns true and `compute()` can
     *                         produce a meaningful interval.  Must be ≥ 2.
     *                         Default 8.
     */
    explicit BootstrapSignalCI(size_t n_bootstrap       = 200,
                               float  confidence_level  = 0.95f,
                               size_t window_size       = 128,
                               size_t min_observations  = 8);

    // Non-copyable.
    BootstrapSignalCI(const BootstrapSignalCI&)            = delete;
    BootstrapSignalCI& operator=(const BootstrapSignalCI&) = delete;

    // -----------------------------------------------------------------------
    // Data intake
    // -----------------------------------------------------------------------

    /**
     * @brief Add a signal observation to the reservoir.
     *
     * When the reservoir exceeds `window_size` the oldest observation is
     * evicted (sliding-window semantics).
     *
     * @param signal_value  Observed signal value (e.g. `delta_bias_shift`).
     */
    void add_observation(float signal_value);

    // -----------------------------------------------------------------------
    // Computation
    // -----------------------------------------------------------------------

    /**
     * @brief Compute the bootstrap confidence interval for the signal mean.
     *
     * Draws `n_bootstrap` re-samples with replacement from the current
     * reservoir, computes the per-re-sample mean, sorts the means, and
     * returns the lower/median/upper percentile cut-points.
     *
     * If fewer than `min_observations` are present the returned interval has
     * lower = upper = median = the simple mean of available observations
     * (or 0 if none).
     *
     * @return ConfidenceInterval { lower, upper, median }.
     */
    [[nodiscard]] ConfidenceInterval compute() const;

    // -----------------------------------------------------------------------
    // Accessors
    // -----------------------------------------------------------------------

    /**
     * @brief True when at least `min_observations` have been recorded.
     */
    [[nodiscard]] bool is_ready() const noexcept;

    /**
     * @brief Number of observations currently in the reservoir.
     */
    [[nodiscard]] size_t observation_count() const noexcept;

    /**
     * @brief Total observations ever recorded (including evicted ones).
     */
    [[nodiscard]] uint64_t total_recorded() const noexcept {
        return total_recorded_.load(std::memory_order_relaxed);
    }

    // -----------------------------------------------------------------------
    // Control
    // -----------------------------------------------------------------------

    /**
     * @brief Clear all observations.
     */
    void reset();

    /**
     * @brief Serialise diagnostics to a JSON object string.
     */
    [[nodiscard]] std::string to_stats_json() const;

private:
    size_t n_bootstrap_;
    float  confidence_level_;
    size_t window_size_;
    size_t min_observations_;

    // Circular buffer of observations (oldest-first).
    std::vector<float> reservoir_;
    size_t head_{0};    // index of next write slot
    size_t count_{0};   // number of valid entries (≤ window_size_)

    std::atomic<uint64_t> total_recorded_{0};

    mutable std::mutex mutex_;

    // Internal: compute re-sample means and sort them.
    // Called with mutex held.
    [[nodiscard]] std::vector<float> bootstrap_means_locked() const;
};

} // namespace llmquant
