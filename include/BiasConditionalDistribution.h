#pragma once

/// @file BiasConditionalDistribution.h
/// @brief Tracks conditional bias distributions P(bias_t | prev_dir > 0) and
///        P(bias_t | prev_dir < 0) to reveal directional asymmetry.
///
/// ## Motivation
/// In momentum markets, P(bias_t > 0 | prev_bias > 0) is elevated.
/// In mean-reverting markets, P(bias_t > 0 | prev_bias < 0) is elevated.
/// By maintaining separate rolling histograms conditioned on the previous
/// bias direction, this module quantifies which regime currently applies
/// and fires `on_asymmetry_detected` when the two distributions diverge.
///
/// ## Algorithm
/// - Separates incoming biases by the previous bias sign:
///     pos_hist: histogram of bias values when prev_bias > 0.
///     neg_hist: histogram of bias values when prev_bias < 0.
/// - Computes total variation distance between the two histograms.
/// - TV distance = 0.5 * Σ|p_i - q_i| ∈ [0, 1].
/// - Fires `on_asymmetry_detected(tv_distance)` when TV > `asymmetry_threshold`.
///
/// ## Feature flag
/// `LLMQUANT_ENABLE_CONDITIONAL_DIST` (default ON).
///
/// Thread-safe.

#include <atomic>
#include <cstdint>
#include <functional>
#include <mutex>
#include <string>
#include <vector>

namespace llmquant {

class BiasConditionalDistribution {
public:
    struct Config {
        /// Rolling window per conditional histogram. Default 50.
        int window{50};

        /// Number of histogram bins. Default 8.
        int n_bins{8};

        /// Bias range for binning. Default -1..+1.
        double range_min{-1.0};
        double range_max{1.0};

        /// TV distance above this fires on_asymmetry_detected. Default 0.3.
        double asymmetry_threshold{0.3};

        /// Fired when TV distance crosses above asymmetry_threshold (outside lock).
        std::function<void(double tv_distance)> on_asymmetry_detected;

        /// Fired when TV distance drops below asymmetry_threshold * 0.7 (recovery).
        std::function<void(double tv_distance)> on_symmetry_restored;
    };

    explicit BiasConditionalDistribution(Config cfg = Config{});

    // -----------------------------------------------------------------------
    // Data intake
    // -----------------------------------------------------------------------

    void record(double bias);

    // -----------------------------------------------------------------------
    // Accessors (lock-free)
    // -----------------------------------------------------------------------

    /// Current total variation distance between pos/neg conditional distributions.
    [[nodiscard]] double tv_distance() const noexcept {
        return tv_.load(std::memory_order_relaxed);
    }

    /// True when TV > asymmetry_threshold.
    [[nodiscard]] bool is_asymmetric() const noexcept {
        return asymmetric_.load(std::memory_order_relaxed);
    }

    /// Total observations recorded.
    [[nodiscard]] uint64_t total_records() const noexcept {
        return total_.load(std::memory_order_relaxed);
    }

    [[nodiscard]] std::string to_stats_json() const;

    void reset();
    void update_config(const Config& cfg);

private:
    int bin_for(double bias) const;
    double compute_tv_locked() const;

    mutable std::mutex mutex_;
    Config cfg_;

    // Two circular histograms (one per conditioning event)
    std::vector<int> pos_win_;   // ring buffer of bin indices (when prev > 0)
    std::vector<int> neg_win_;   // ring buffer of bin indices (when prev < 0)
    std::vector<int> pos_hist_;
    std::vector<int> neg_hist_;
    int pos_head_{0}, pos_fill_{0};
    int neg_head_{0}, neg_fill_{0};

    double prev_bias_{0.0};
    bool   seeded_{false};
    bool   prev_asymmetric_{false};

    std::atomic<double>   tv_{0.0};
    std::atomic<bool>     asymmetric_{false};
    std::atomic<uint64_t> total_{0};
};

} // namespace llmquant
