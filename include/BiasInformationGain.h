#pragma once

/// @file BiasInformationGain.h
/// @brief Computes the information gain (mutual information proxy) between
///        consecutive bias buckets — how much does knowing the current bucket
///        reduce uncertainty about the next one?
///
/// ## Motivation
/// Mutual information I(X;Y) = H(X) + H(Y) - H(X,Y) quantifies statistical
/// dependency between the current and next bias bin.  High MI → knowing today's
/// sentiment tells you something about tomorrow's (momentum / mean-reversion
/// detectable).  Low MI → near-independent steps (noise).  Unlike correlation,
/// MI captures nonlinear dependence.
///
/// ## Algorithm
/// - Discretises bias into `n_bins` bins.
/// - Maintains a rolling joint count matrix C[i][j] = count(bin_t=i, bin_{t+1}=j)
///   and marginals P[i] and Q[j] via a ring buffer of (prev_bin, curr_bin) pairs.
/// - Computes MI = Σ_{i,j} p(i,j) log2(p(i,j) / (p(i) p(j))) each record.
/// - Normalised MI (NMI) = MI / H(X) ∈ [0,1] so it is scale-free.
/// - Fires `on_mi_change(old_nmi, new_nmi)` when |ΔNMI| > change_threshold.
///
/// ## Feature flag
/// `LLMQUANT_ENABLE_INFORMATION_GAIN` (default ON).
///
/// Thread-safe.

#include <atomic>
#include <cstdint>
#include <functional>
#include <mutex>
#include <string>
#include <vector>

namespace llmquant {

class BiasInformationGain {
public:
    struct Config {
        /// Number of bias bins. Default 6.
        int n_bins{6};

        /// Rolling window of (prev_bin, curr_bin) pairs. Default 100.
        int window{100};

        /// Bias range for binning. Default -1..+1.
        double range_min{-1.0};
        double range_max{1.0};

        /// |ΔNMI| above which on_mi_change fires. Default 0.05.
        double change_threshold{0.05};

        /// Minimum fill before computing. Default 16.
        int min_samples{16};

        /// Fired when normalised MI changes significantly (outside lock).
        std::function<void(double old_nmi, double new_nmi)> on_mi_change;
    };

    explicit BiasInformationGain(Config cfg = Config{});

    // -----------------------------------------------------------------------
    // Data intake
    // -----------------------------------------------------------------------

    void record(double bias);

    // -----------------------------------------------------------------------
    // Accessors (lock-free)
    // -----------------------------------------------------------------------

    /// Raw mutual information I(X;Y) in bits.
    [[nodiscard]] double mutual_information() const noexcept {
        return mi_.load(std::memory_order_relaxed);
    }

    /// Normalised MI = I(X;Y) / H(X) ∈ [0,1]. 0 = independent, 1 = fully predictable.
    [[nodiscard]] double normalised_mi() const noexcept {
        return nmi_.load(std::memory_order_relaxed);
    }

    /// Total MI-change events fired.
    [[nodiscard]] uint64_t mi_change_events() const noexcept {
        return change_events_.load(std::memory_order_relaxed);
    }

    /// Total observations recorded.
    [[nodiscard]] uint64_t total_records() const noexcept {
        return total_.load(std::memory_order_relaxed);
    }

    [[nodiscard]] std::string to_stats_json() const;

    void reset();
    void update_config(const Config& cfg);

private:
    int  bin_for(double v) const;
    void compute_mi_locked();

    mutable std::mutex mutex_;
    Config cfg_;

    // Joint ring buffer: each entry is prev_bin * n_bins + curr_bin
    std::vector<int> ring_;
    int ring_head_{0};
    int ring_fill_{0};

    // Joint counts (n_bins × n_bins)
    std::vector<int> joint_;

    int prev_bin_{-1};
    bool seeded_{false};
    double prev_nmi_{-1.0};

    std::atomic<double>   mi_{0.0};
    std::atomic<double>   nmi_{0.0};
    std::atomic<uint64_t> change_events_{0};
    std::atomic<uint64_t> total_{0};
};

} // namespace llmquant
