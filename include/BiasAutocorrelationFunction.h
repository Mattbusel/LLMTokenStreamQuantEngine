#pragma once

/// @file BiasAutocorrelationFunction.h
/// @brief Rolling autocorrelation function (ACF) at lags 1..max_lag for bias streams.
///
/// ## Motivation
/// Autocorrelation tells us whether the current bias is predictive of the next
/// one.  Positive ACF at lag 1 → trending / momentum.  Negative ACF → mean-
/// reversion / noise.  Long-memory ACF → slow regime change.  Applied to LLM
/// sentiment, it answers: "does this narrative have momentum, or is it noise?"
///
/// ## Algorithm
/// Maintains a circular buffer of the last `window` bias observations.  On each
/// new observation the function:
///   1. Computes the sample mean μ over the window.
///   2. For each lag k in [1, max_lag], computes:
///        r_k = Σ(x_t − μ)(x_{t-k} − μ) / Σ(x_t − μ)²
///      Using the biased estimator (denominator = n, not n-k) for online stability.
///   3. Identifies the lag with the highest |r_k| as the dominant_lag.
///   4. Fires `on_periodic_pattern` when |r_{dominant_lag}| > periodic_threshold
///      for the first time, and `on_pattern_lost` when it drops back below.
///   5. Fires `on_lag1_sign_change` when the sign of r_1 flips (momentum ↔ mean-rev).
///
/// ## Feature flag
/// `LLMQUANT_ENABLE_BIAS_ACF` (default ON).
///
/// Thread-safe.

#include <atomic>
#include <cstdint>
#include <functional>
#include <mutex>
#include <string>
#include <vector>

namespace llmquant {

class BiasAutocorrelationFunction {
public:
    struct Config {
        /// Sliding window size (samples). Default 64.
        int window{64};

        /// Maximum lag to compute. Must be < window/2. Default 16.
        int max_lag{16};

        /// Threshold on |r_k| to detect a periodic pattern. Default 0.4.
        double periodic_threshold{0.4};

        /// Minimum samples before outputs are valid. Default == window.
        int min_samples{0};  // 0 → use window

        /// Fired when a dominant periodic lag exceeds periodic_threshold.
        /// Args: lag (int), acf_value (double).
        std::function<void(int lag, double acf)> on_periodic_pattern;

        /// Fired when the periodic pattern is no longer detected (hysteresis).
        std::function<void()> on_pattern_lost;

        /// Fired when r_1 changes sign (positive ↔ negative).
        /// Args: new_r1 value.
        std::function<void(double r1)> on_lag1_sign_change;
    };

    explicit BiasAutocorrelationFunction(Config cfg = Config{});

    // -----------------------------------------------------------------------
    // Data intake
    // -----------------------------------------------------------------------

    void record(double x);

    // -----------------------------------------------------------------------
    // Accessors (lock-free for scalars; vector copy for ACF)
    // -----------------------------------------------------------------------

    /// Lag-1 autocorrelation, or NaN if not enough data.
    [[nodiscard]] double r1() const noexcept {
        return r1_.load(std::memory_order_relaxed);
    }

    /// Dominant lag (1-based) with highest |r_k|, or 0 if not enough data.
    [[nodiscard]] int dominant_lag() const noexcept {
        return static_cast<int>(dom_lag_.load(std::memory_order_relaxed));
    }

    /// ACF value at the dominant lag, or NaN if not enough data.
    [[nodiscard]] double dominant_acf() const noexcept {
        return dom_acf_.load(std::memory_order_relaxed);
    }

    /// True if a periodic pattern is currently detected.
    [[nodiscard]] bool is_periodic() const noexcept {
        return periodic_.load(std::memory_order_relaxed);
    }

    [[nodiscard]] uint64_t total_records() const noexcept {
        return total_.load(std::memory_order_relaxed);
    }

    [[nodiscard]] uint64_t periodic_events() const noexcept {
        return periodic_events_.load(std::memory_order_relaxed);
    }

    /// Thread-safe snapshot of the full ACF vector [r_1, r_2, ..., r_max_lag].
    /// Returns an empty vector if not enough data.
    [[nodiscard]] std::vector<double> acf_snapshot() const;

    [[nodiscard]] std::string to_stats_json() const;

    void reset();
    void update_config(const Config& cfg);

private:
    void recompute_locked();

    mutable std::mutex mutex_;
    Config             cfg_;

    std::vector<double> buf_;   ///< circular buffer (window samples)
    int head_{0};
    int fill_{0};

    std::vector<double> acf_;   ///< current ACF [r_1..r_max_lag]

    bool prev_r1_positive_{false};
    bool prev_periodic_{false};
    int  prev_dom_lag_{0};

    std::atomic<double>   r1_{std::numeric_limits<double>::quiet_NaN()};
    std::atomic<double>   dom_acf_{std::numeric_limits<double>::quiet_NaN()};
    std::atomic<uint32_t> dom_lag_{0};
    std::atomic<bool>     periodic_{false};
    std::atomic<uint64_t> total_{0};
    std::atomic<uint64_t> periodic_events_{0};
};

} // namespace llmquant
