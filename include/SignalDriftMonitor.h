#pragma once

#include <atomic>
#include <cstdint>
#include <functional>
#include <mutex>
#include <string>
#include <vector>

namespace llmquant {

/**
 * @brief Wasserstein-distance signal-distribution drift monitor.
 *
 * Detects structural shifts in the bias-shift distribution without requiring
 * hard-coded regime thresholds.  The monitor maintains two rolling windows:
 *
 * - **Baseline window** (size `baseline_size`): the oldest observations; used
 *   as the reference distribution.
 * - **Recent window** (size `recent_size`): the most recent observations;
 *   compared against the baseline.
 *
 * The 1-D Wasserstein-1 distance (Earth Mover's Distance) between the two
 * empirical distributions is computed as the L1 distance between their sorted
 * sample arrays — an O(N log N) computation done on-demand via accessors.
 *
 * ## Interpretation
 * - W1 ≈ 0 → recent distribution matches baseline → stable regime.
 * - W1 > `drift_threshold` → distribution has shifted → possible regime change.
 * - W1 rising monotonically → continuous directional trend → trending regime.
 *
 * ## Design
 * - `record()` is O(1) amortised (ring-buffer push).
 * - `wasserstein_distance()` sorts both windows each call — intended for low-
 *   frequency diagnostics, not the hot signal path.
 * - Thread-safe via mutex; atomic outputs for the most recent W1 and drift flag.
 */
class SignalDriftMonitor {
public:
    // -------------------------------------------------------------------------
    // Configuration
    // -------------------------------------------------------------------------
    struct Config {
        int    baseline_size    = 128;  ///< Reference window size.
        int    recent_size      = 32;   ///< Comparison window size.
        double drift_threshold  = 0.10; ///< W1 above which drift is declared.
        double clear_hysteresis = 0.7;  ///< Clear factor: W1 < threshold*hysteresis to de-assert.

        /// Fired when W1 crosses above drift_threshold (first cross only until cleared).
        std::function<void(double)> on_drift_detected;
        /// Fired when W1 falls back below the hysteresis level after a drift event.
        std::function<void(double)> on_drift_cleared;
    };

    // -------------------------------------------------------------------------
    // Construction
    // -------------------------------------------------------------------------
    SignalDriftMonitor();
    explicit SignalDriftMonitor(const Config& cfg);

    // -------------------------------------------------------------------------
    // Core
    // -------------------------------------------------------------------------

    /**
     * @brief Record a new bias-shift observation.
     * @param value  The delta_bias_shift of the most recent signal.
     */
    void record(double value);

    // -------------------------------------------------------------------------
    // Accessors (lock-free hot path)
    // -------------------------------------------------------------------------

    /// Most recently computed Wasserstein-1 distance (updated each record call).
    double last_w1() const noexcept;
    /// True if W1 currently exceeds drift_threshold.
    bool is_drifting() const noexcept;
    /// Total observations recorded.
    uint64_t total_records() const noexcept;
    /// Number of drift events detected.
    uint64_t drift_events() const noexcept;

    // -------------------------------------------------------------------------
    // Accessors (full computation, mutex-guarded)
    // -------------------------------------------------------------------------

    /// Compute and return the current W1 distance between baseline and recent windows.
    double wasserstein_distance() const noexcept;
    /// Mean of the baseline window.
    double baseline_mean() const noexcept;
    /// Mean of the recent window.
    double recent_mean() const noexcept;

    // -------------------------------------------------------------------------
    // Mutation
    // -------------------------------------------------------------------------

    /** Reset all observations and drift state. */
    void reset();
    /** Replace config and reset. */
    void update_config(const Config& cfg);

    // -------------------------------------------------------------------------
    // Diagnostics
    // -------------------------------------------------------------------------

    /** Compact JSON with drift monitor stats. */
    std::string to_stats_json() const;

private:
    Config cfg_;
    mutable std::mutex mutex_;

    // Circular buffers
    std::vector<double> baseline_;
    std::vector<double> recent_;
    int   baseline_head_ = 0;
    int   baseline_count_ = 0;
    int   recent_head_  = 0;
    int   recent_count_ = 0;

    bool   drifting_  = false;

    // Atomic outputs for hot-path reads
    std::atomic<double>   last_w1_{0.0};
    std::atomic<bool>     drift_flag_{false};
    std::atomic<uint64_t> total_{0};
    std::atomic<uint64_t> drift_events_{0};

    // Helper
    static double compute_w1(std::vector<double> a, std::vector<double> b) noexcept;
};

}  // namespace llmquant
