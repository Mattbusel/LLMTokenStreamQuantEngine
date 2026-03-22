#pragma once

/// @file SignalCompressor.h
/// @brief Computes the Lempel-Ziv (LZ76) complexity of the discretised bias
///        sequence as a proxy for algorithmic randomness vs. regime structure.
///
/// ## Motivation
/// LZ complexity measures how many distinct substrings are needed to describe
/// a sequence.  For LLM bias streams:
///   - Low LZ complexity → the bias sequence is highly predictable/compressible
///     (strong trend, momentum strategies applicable).
///   - High LZ complexity → near-random sequence (noise regime, reduce signal
///     reliance).
///
/// Normalised: LZC_norm = LZC / (n / log2(n))  so that white noise ≈ 1.
///
/// ## Algorithm
/// Discretises bias into `n_symbols` symbols.  Runs the LZ76 dictionary-build
/// algorithm online on the rolling window of length `window`.
///
/// Fires `on_complexity_change(old_norm, new_norm)` when normalised complexity
/// changes by more than `change_threshold`.
///
/// ## Feature flag
/// `LLMQUANT_ENABLE_SIGNAL_COMPRESSOR` (default ON).
///
/// Thread-safe.

#include <atomic>
#include <cstdint>
#include <functional>
#include <mutex>
#include <string>
#include <vector>

namespace llmquant {

class SignalCompressor {
public:
    struct Config {
        /// Rolling window length in symbols. Default 64.
        int window{64};

        /// Number of distinct symbols (bins) for bias discretisation. Default 4.
        int n_symbols{4};

        /// Bias range for bucketing. Defaults -1..+1.
        double range_min{-1.0};
        double range_max{1.0};

        /// Minimum normalised LZC change to fire callback. Default 0.1.
        double change_threshold{0.1};

        /// Minimum window fill before computing. Default 16.
        int min_samples{16};

        /// Fired when normalised LZC changes by > change_threshold.
        /// Parameters: (old_norm_lzc, new_norm_lzc).
        std::function<void(double, double)> on_complexity_change;
    };

    explicit SignalCompressor(Config cfg = Config{});

    // -----------------------------------------------------------------------
    // Data intake
    // -----------------------------------------------------------------------

    void record(double bias);

    // -----------------------------------------------------------------------
    // Accessors (lock-free)
    // -----------------------------------------------------------------------

    /// Raw LZ76 phrase count over the current window.
    [[nodiscard]] int lz_complexity() const noexcept {
        return lz_raw_.load(std::memory_order_relaxed);
    }

    /// Normalised LZC ∈ [0, ~2]. White noise ≈ 1.
    [[nodiscard]] double normalised_complexity() const noexcept {
        return lz_norm_.load(std::memory_order_relaxed);
    }

    /// Total observations recorded.
    [[nodiscard]] uint64_t total_records() const noexcept {
        return total_.load(std::memory_order_relaxed);
    }

    [[nodiscard]] std::string to_stats_json() const;

    void reset();
    void update_config(const Config& cfg);

private:
    int lz76_locked() const;  // compute LZ76 phrase count on current window
    int bucket_for(double bias) const;

    mutable std::mutex mutex_;
    Config cfg_;

    std::vector<int> win_buf_;   // ring buffer of symbol indices
    int win_head_{0};
    int win_fill_{0};

    double prev_lz_norm_{-1.0};  // <0 = uninitialised

    std::atomic<int>      lz_raw_{0};
    std::atomic<double>   lz_norm_{0.0};
    std::atomic<uint64_t> total_{0};
};

} // namespace llmquant
