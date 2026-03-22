#pragma once

/// @file BiasBootstrapCI.h
/// @brief Computes bootstrap confidence intervals for the rolling mean bias,
///        providing uncertainty-aware signal reliability bounds.
///
/// ## Motivation
/// A mean bias of 0.3 is meaningless without uncertainty context.  If the
/// 95% CI is [0.25, 0.35] the signal is reliable; if it is [-0.5, 1.1] the
/// regime is too noisy to trade on.  Bootstrap CI is model-free (no
/// normality assumption) and robust to heavy tails — important for LLM bias
/// streams that exhibit fat-tailed bursts.
///
/// ## Algorithm
/// - Maintains a rolling window of `window` bias observations.
/// - Every `recompute_interval` records, performs `n_bootstrap` bootstrap
///   resamplings of the window:
///     For each resample: draw window samples with replacement, compute mean.
///   - Reports the 2.5th and 97.5th percentiles of bootstrap means as the
///     95% CI (BCa correction omitted for O(1) simplicity).
/// - Fires `on_ci_wide(lo, hi)` when CI width > `width_threshold` (unreliable).
/// - Fires `on_ci_narrow(lo, hi)` when CI width < `width_threshold * 0.3` (reliable).
///
/// ## Feature flag
/// `LLMQUANT_ENABLE_BOOTSTRAP_CI` (default ON).
///
/// Thread-safe.

#include <atomic>
#include <cstdint>
#include <functional>
#include <mutex>
#include <random>
#include <string>
#include <vector>

namespace llmquant {

class BiasBootstrapCI {
public:
    struct Config {
        /// Rolling window of observations. Default 64.
        int window{64};

        /// Number of bootstrap resamples per computation. Default 200.
        int n_bootstrap{200};

        /// Recompute every N records. Default 10.
        int recompute_interval{10};

        /// CI width > this fires on_ci_wide. Default 0.3.
        double width_threshold{0.3};

        /// Minimum fill before computing. Default 16.
        int min_samples{16};

        /// Fired when CI width > width_threshold (outside lock).
        std::function<void(double lo, double hi)> on_ci_wide;

        /// Fired when CI width < width_threshold * 0.3 (outside lock).
        std::function<void(double lo, double hi)> on_ci_narrow;
    };

    explicit BiasBootstrapCI(Config cfg = Config{});

    // -----------------------------------------------------------------------
    // Data intake
    // -----------------------------------------------------------------------

    void record(double bias);

    // -----------------------------------------------------------------------
    // Accessors (lock-free)
    // -----------------------------------------------------------------------

    /// Lower bound of 95% bootstrap CI for rolling mean.
    [[nodiscard]] double ci_lo() const noexcept { return lo_.load(std::memory_order_relaxed); }

    /// Upper bound of 95% bootstrap CI for rolling mean.
    [[nodiscard]] double ci_hi() const noexcept { return hi_.load(std::memory_order_relaxed); }

    /// CI width = ci_hi - ci_lo.
    [[nodiscard]] double ci_width() const noexcept {
        return hi_.load(std::memory_order_relaxed) -
               lo_.load(std::memory_order_relaxed);
    }

    /// Rolling window mean.
    [[nodiscard]] double rolling_mean() const noexcept {
        return mean_.load(std::memory_order_relaxed);
    }

    /// Total wide-CI events fired.
    [[nodiscard]] uint64_t wide_events() const noexcept {
        return wide_events_.load(std::memory_order_relaxed);
    }

    /// Total observations recorded.
    [[nodiscard]] uint64_t total_records() const noexcept {
        return total_.load(std::memory_order_relaxed);
    }

    [[nodiscard]] std::string to_stats_json() const;

    void reset();
    void update_config(const Config& cfg);

private:
    void run_bootstrap_locked();

    mutable std::mutex mutex_;
    Config cfg_;

    std::vector<double> win_buf_;
    int win_head_{0};
    int win_fill_{0};

    std::mt19937 rng_{42};

    std::atomic<double>   lo_{0.0};
    std::atomic<double>   hi_{0.0};
    std::atomic<double>   mean_{0.0};
    std::atomic<uint64_t> wide_events_{0};
    std::atomic<uint64_t> total_{0};
};

} // namespace llmquant
