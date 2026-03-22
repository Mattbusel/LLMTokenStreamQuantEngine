#pragma once

/// @file TokenWeightHistogram.h
/// @brief Discretized histogram of token bias values.
///
/// ## Motivation
/// Watching the raw bias stream gives no sense of its *distribution*.
/// A histogram makes the shape visible: is the weight distribution
/// symmetric? Bimodal? Heavily left-skewed?  Knowing the mode bucket and
/// the histogram entropy lets downstream modules adjust confidence and
/// risk limits accordingly.
///
/// ## Algorithm
/// Maintains a 20-bucket fixed-width histogram over [-1, 1].
/// Each `record(bias)` increments the appropriate bucket.
///
/// Histogram entropy:
///   H = -Σ p_i * log2(p_i)  (normalized, range [0, log2(buckets)])
///
/// Mode bucket: index with the highest count.
///
/// Fires `on_distribution_shift(new_mode)` when the mode bucket changes.
///
/// ## Feature flag
/// `LLMQUANT_ENABLE_WEIGHT_HISTOGRAM` (default ON).
///
/// Thread-safe.

#include <array>
#include <atomic>
#include <cstdint>
#include <functional>
#include <mutex>
#include <string>

namespace llmquant {

class TokenWeightHistogram {
public:
    static constexpr int kBuckets = 20;

    struct Config {
        /// Bias range minimum. Default -1.0.
        double range_min{-1.0};

        /// Bias range maximum. Default 1.0.
        double range_max{1.0};

        /// Fired when the mode bucket index changes (outside lock).
        std::function<void(int new_mode_bucket)> on_distribution_shift;
    };

    explicit TokenWeightHistogram(Config cfg = Config{});

    // -----------------------------------------------------------------------
    // Data intake
    // -----------------------------------------------------------------------

    /// Record a new bias value. Values outside [range_min, range_max] are clamped.
    void record(double bias);

    // -----------------------------------------------------------------------
    // Accessors (lock-free)
    // -----------------------------------------------------------------------

    /// Index of the bucket with the highest count (mode).
    [[nodiscard]] int mode_bucket() const noexcept {
        return mode_bucket_.load(std::memory_order_relaxed);
    }

    /// Center value of the mode bucket.
    [[nodiscard]] double mode_value() const noexcept;

    /// Shannon entropy of the distribution ∈ [0, log2(kBuckets)].
    [[nodiscard]] double entropy() const noexcept {
        return entropy_.load(std::memory_order_relaxed);
    }

    /// Total observations recorded.
    [[nodiscard]] uint64_t total() const noexcept {
        return total_.load(std::memory_order_relaxed);
    }

    /// Count in a specific bucket (0-indexed).
    [[nodiscard]] uint64_t bucket_count(int idx) const noexcept;

    [[nodiscard]] std::string to_stats_json() const;

    void reset();
    void update_config(const Config& cfg);

private:
    mutable std::mutex mutex_;
    Config cfg_;

    std::array<uint64_t, kBuckets> counts_{};
    int prev_mode_{-1};

    std::atomic<int>      mode_bucket_{0};
    std::atomic<double>   entropy_{0.0};
    std::atomic<uint64_t> total_{0};
};

} // namespace llmquant
