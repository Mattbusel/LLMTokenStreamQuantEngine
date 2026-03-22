#pragma once

/// @file CrossTokenCorrelationMatrix.h
/// @brief Online rolling Pearson correlation between multiple named bias channels
///        — detects when sentiment signals from different sources agree or diverge.
///
/// ## Motivation
/// In a multi-source sentiment pipeline (Reddit, Twitter, news), each source
/// produces its own bias series.  When the sources agree (high pairwise correlation),
/// signal confidence is high; when they diverge (negative correlation), the signal
/// is conflicted and position sizes should be reduced.  This module maintains
/// pairwise rolling correlations across N registered channels.
///
/// ## Algorithm
/// For each channel pair (i, j), maintains:
///   Σ(x_i), Σ(x_i²), Σ(x_i * x_j), n  over a rolling window.
/// Pearson r = [n*Σ(xy) - Σx*Σy] / sqrt([n*Σx²-(Σx)²][n*Σy²-(Σy)²]).
/// Uses a sliding-window approach via ring buffers per channel.
///
/// Fires `on_divergence(ch_i, ch_j, r)` when |r| drops below `divergence_threshold`.
/// Fires `on_convergence(ch_i, ch_j, r)` when |r| rises above `convergence_threshold`.
///
/// ## Feature flag
/// `LLMQUANT_ENABLE_CROSS_TOKEN_CORR` (default ON).
///
/// Thread-safe.

#include <atomic>
#include <cstdint>
#include <functional>
#include <mutex>
#include <string>
#include <vector>

namespace llmquant {

class CrossTokenCorrelationMatrix {
public:
    /// Maximum number of channels supported.
    static constexpr int kMaxChannels = 8;

    struct Config {
        /// Rolling window length for each channel. Default 30.
        int window{30};

        /// Number of registered channels (1..kMaxChannels). Default 2.
        int n_channels{2};

        /// |r| below this fires on_divergence (once per crossing). Default 0.3.
        double divergence_threshold{0.3};

        /// |r| above this fires on_convergence (once per crossing). Default 0.7.
        double convergence_threshold{0.7};

        /// Fired when correlation between channel i and j drops below divergence_threshold.
        std::function<void(int ch_i, int ch_j, double r)> on_divergence;

        /// Fired when correlation between channel i and j rises above convergence_threshold.
        std::function<void(int ch_i, int ch_j, double r)> on_convergence;
    };

    explicit CrossTokenCorrelationMatrix(Config cfg = Config{});

    // -----------------------------------------------------------------------
    // Data intake
    // -----------------------------------------------------------------------

    /// Record a bias value for the given channel index [0, n_channels).
    void record(int channel, double bias);

    // -----------------------------------------------------------------------
    // Accessors
    // -----------------------------------------------------------------------

    /// Pearson correlation between channels i and j. Returns 0 if insufficient data.
    [[nodiscard]] double correlation(int ch_i, int ch_j) const;

    /// Total observations recorded across all channels.
    [[nodiscard]] uint64_t total_records() const noexcept {
        return total_.load(std::memory_order_relaxed);
    }

    [[nodiscard]] std::string to_stats_json() const;

    void reset();
    void update_config(const Config& cfg);

private:
    void recompute_pair_locked(int i, int j);

    mutable std::mutex mutex_;
    Config cfg_;

    // Per-channel ring buffers
    std::vector<std::vector<double>> bufs_;  // [n_channels][window]
    std::vector<int>                 heads_;
    std::vector<int>                 fills_;

    // Pairwise correlation cache: [n_channels * n_channels]
    std::vector<double>  corr_cache_;  // symmetric, corr_cache_[i*n + j]

    // Previous state for edge-detection per pair
    std::vector<bool>    prev_diverged_;
    std::vector<bool>    prev_converged_;

    std::atomic<uint64_t> total_{0};
};

} // namespace llmquant
