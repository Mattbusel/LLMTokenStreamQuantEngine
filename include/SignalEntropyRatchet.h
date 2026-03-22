#pragma once

/// @file SignalEntropyRatchet.h
/// @brief One-way ratchet on the Shannon entropy of the discretised bias
///        distribution — asymmetrically tracks rising uncertainty vs. genuine
///        regime stabilisation.
///
/// ## Motivation
/// Shannon entropy of the bias distribution rises quickly when the market is
/// confused (many equally-probable bias buckets).  A symmetric tracker would
/// declare "low entropy" after just a few stable ticks — too quick.  The
/// ratchet floor prevents the floor from dropping until entropy has remained
/// below the current floor for `floor_hold_ticks` consecutive ticks, enforcing
/// that stabilisation must be *sustained* before position sizes are increased.
///
/// ## Algorithm
/// - Bias values are discretised into `n_buckets` equal-width bins over
///   [range_min, range_max].
/// - Maintains a rolling histogram of the last `window` observations.
/// - Shannon entropy: H = -Σ p_i * log2(p_i)  (zero terms skipped).
/// - `entropy_floor`: the lowest entropy seen, updated (raised) eagerly but
///   lowered only after `floor_hold_ticks` consecutive sub-floor observations.
/// - Fires `on_entropy_spike(entropy)` when H rises above `spike_threshold`.
/// - Fires `on_floor_drop(new_floor)` when the floor genuinely drops.
///
/// ## Feature flag
/// `LLMQUANT_ENABLE_ENTROPY_RATCHET` (default ON).
///
/// Thread-safe.

#include <atomic>
#include <cstdint>
#include <functional>
#include <mutex>
#include <string>
#include <vector>

namespace llmquant {

class SignalEntropyRatchet {
public:
    struct Config {
        /// Rolling window size. Default 64.
        int window{64};

        /// Number of equal-width histogram bins. Default 10.
        int n_buckets{10};

        /// Bias range minimum (values clipped). Default -1.0.
        double range_min{-1.0};

        /// Bias range maximum (values clipped). Default 1.0.
        double range_max{1.0};

        /// H above this fires on_entropy_spike. Default log2(n_buckets) * 0.8.
        double spike_threshold{2.64};  // ≈ 0.8 * log2(10)

        /// Consecutive sub-floor ticks required before floor drops. Default 5.
        int floor_hold_ticks{5};

        /// Fired when H crosses above spike_threshold (outside lock).
        std::function<void(double entropy)> on_entropy_spike;

        /// Fired when the floor genuinely drops (outside lock).
        std::function<void(double new_floor)> on_floor_drop;
    };

    explicit SignalEntropyRatchet(Config cfg = Config{});

    // -----------------------------------------------------------------------
    // Data intake
    // -----------------------------------------------------------------------

    /// Record a new bias value. Updates histogram, entropy, and ratchet floor.
    void record(double bias);

    // -----------------------------------------------------------------------
    // Accessors (lock-free)
    // -----------------------------------------------------------------------

    /// Current Shannon entropy of the rolling bias distribution.
    [[nodiscard]] double entropy() const noexcept {
        return entropy_.load(std::memory_order_relaxed);
    }

    /// Current ratchet floor (lowest entropy that required sustained stabilisation).
    [[nodiscard]] double entropy_floor() const noexcept {
        return floor_.load(std::memory_order_relaxed);
    }

    /// True when current entropy > spike_threshold.
    [[nodiscard]] bool is_spiked() const noexcept {
        return spiked_.load(std::memory_order_relaxed);
    }

    /// Total spike events fired.
    [[nodiscard]] uint64_t spike_events() const noexcept {
        return spike_events_.load(std::memory_order_relaxed);
    }

    /// Total floor-drop events fired.
    [[nodiscard]] uint64_t floor_drop_events() const noexcept {
        return floor_drop_events_.load(std::memory_order_relaxed);
    }

    /// Total observations recorded.
    [[nodiscard]] uint64_t total_records() const noexcept {
        return total_.load(std::memory_order_relaxed);
    }

    [[nodiscard]] std::string to_stats_json() const;

    void reset();
    void update_config(const Config& cfg);

private:
    int bucket_for(double bias) const;
    double compute_entropy_locked() const;

    mutable std::mutex mutex_;
    Config cfg_;

    // Rolling histogram via ring buffer of bucket indices
    std::vector<int>    win_buf_;    // ring buffer of bucket indices
    std::vector<int>    hist_;       // current histogram counts
    int win_head_{0};
    int win_fill_{0};

    double current_floor_{-1.0};    // <0 means uninitialised
    int    sub_floor_streak_{0};    // consecutive ticks below floor
    bool   prev_spiked_{false};

    std::atomic<double>   entropy_{0.0};
    std::atomic<double>   floor_{0.0};
    std::atomic<bool>     spiked_{false};
    std::atomic<uint64_t> spike_events_{0};
    std::atomic<uint64_t> floor_drop_events_{0};
    std::atomic<uint64_t> total_{0};
};

} // namespace llmquant
