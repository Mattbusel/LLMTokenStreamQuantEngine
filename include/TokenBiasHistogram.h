#pragma once
#include <atomic>
#include <functional>
#include <mutex>
#include <string>
#include <vector>

namespace llmquant {

/**
 * @brief Fixed-bin histogram of bias values with mode-change detection.
 *
 * Partitions [lo, hi] into n_bins equal-width bins and accumulates counts.
 * The peak (mode) bin index and distribution entropy are tracked.  A callback
 * fires when the mode bin changes.
 */
class TokenBiasHistogram {
public:
    struct Config {
        int    n_bins{10};    ///< Number of histogram bins
        double lo{-1.0};      ///< Lower edge of histogram range
        double hi{1.0};       ///< Upper edge of histogram range
        int    min_samples{5}; ///< Minimum samples before mode detection

        /// Fired when the peak bin index changes.
        std::function<void(int /*new_mode_bin*/, int /*prev_mode_bin*/)> on_mode_change;
    };

    explicit TokenBiasHistogram(Config cfg = {});

    /// Feed a new bias sample (clamped to [lo, hi]).
    void record(double bias);

    /// Count in a specific bin (0-indexed).
    uint64_t    bin_count(int bin) const noexcept;
    int         peak_bin()         const noexcept;
    double      peak_center()      const noexcept; ///< Centre value of mode bin
    double      entropy()          const noexcept; ///< Shannon entropy of distribution
    std::size_t mode_changes()     const noexcept;
    std::size_t total_records()    const noexcept;

    std::string to_stats_json() const;
    void reset();
    void update_config(const Config& cfg);

private:
    mutable std::mutex         mutex_;
    Config                     cfg_;
    std::vector<uint64_t>      counts_;    ///< Per-bin counts

    std::atomic<int>      peak_bin_{0};
    std::atomic<double>   entropy_{0.0};
    std::atomic<uint64_t> mode_events_{0};
    std::atomic<uint64_t> total_{0};
    int prev_mode_{-1};

    void recompute_locked();
};

} // namespace llmquant
