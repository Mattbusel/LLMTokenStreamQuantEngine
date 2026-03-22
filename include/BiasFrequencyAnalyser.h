#pragma once

/// @file BiasFrequencyAnalyser.h
/// @brief Identifies dominant oscillation frequencies in the rolling LLM bias
///        series via a Type-II Discrete Cosine Transform (DCT-II).
///
/// ## Motivation
/// Bias signals often exhibit periodic patterns: a ~5-tick echo from repetitive
/// hedging phrases, a ~20-tick cycle from paragraph structure, or high-frequency
/// noise.  By extracting the dominant DCT frequency component, this module can
/// distinguish genuine momentum from periodic oscillation — enabling smarter
/// detrending and pattern matching.
///
/// ## Algorithm
/// 1. Maintains a rolling buffer of `window` bias values.
/// 2. On each `record()`, recomputes DCT-II coefficients for k=1..max_k:
///      X[k] = Σ_{n=0}^{N-1} x[n] * cos(π/N * (n+0.5) * k)
/// 3. Identifies the dominant k (maximum |X[k]|) and normalised frequency
///    f = k / window  (cycles per sample).
/// 4. Fires `on_dominant_frequency_change(old_k, new_k)` when dominant k shifts.
///
/// ## Feature flag
/// `LLMQUANT_ENABLE_FREQ_ANALYSER` (default ON).
///
/// Thread-safe.

#include <atomic>
#include <cstdint>
#include <functional>
#include <mutex>
#include <string>
#include <vector>

namespace llmquant {

class BiasFrequencyAnalyser {
public:
    /// Maximum DCT coefficient index computed (window/2 is the Nyquist limit).
    static constexpr int kMaxK = 32;

    struct Config {
        /// Rolling window size. Must be >= 8. Default 64.
        int window{64};

        /// Maximum DCT coefficient index to compute (≤ kMaxK, ≤ window/2). Default 16.
        int max_k{16};

        /// Minimum samples before emitting estimates. Default 16.
        int min_samples{16};

        /// Fired when dominant frequency index changes (outside lock).
        /// Parameters: (old_dominant_k, new_dominant_k, new_freq_normalised).
        std::function<void(int, int, double)> on_dominant_frequency_change;
    };

    explicit BiasFrequencyAnalyser(Config cfg = Config{});

    // -----------------------------------------------------------------------
    // Data intake
    // -----------------------------------------------------------------------

    /// Record a new bias value. Recomputes DCT when enough samples are available.
    void record(double bias);

    // -----------------------------------------------------------------------
    // Accessors (lock-free)
    // -----------------------------------------------------------------------

    /// Current dominant DCT coefficient index (1-based). 0 until enough data.
    [[nodiscard]] int dominant_k() const noexcept {
        return dominant_k_.load(std::memory_order_relaxed);
    }

    /// Normalised dominant frequency (dominant_k / window), cycles per sample.
    [[nodiscard]] double dominant_frequency() const noexcept {
        return dominant_freq_.load(std::memory_order_relaxed);
    }

    /// Power at dominant_k (square of DCT coefficient).
    [[nodiscard]] double dominant_power() const noexcept {
        return dominant_power_.load(std::memory_order_relaxed);
    }

    /// Total observations recorded.
    [[nodiscard]] uint64_t total_records() const noexcept {
        return total_.load(std::memory_order_relaxed);
    }

    [[nodiscard]] std::string to_stats_json() const;

    void reset();
    void update_config(const Config& cfg);

private:
    void recompute_locked();

    mutable std::mutex mutex_;
    Config cfg_;

    std::vector<double> buf_;
    int head_{0};
    int fill_{0};

    int prev_dominant_k_{0};

    std::atomic<int>      dominant_k_{0};
    std::atomic<double>   dominant_freq_{0.0};
    std::atomic<double>   dominant_power_{0.0};
    std::atomic<uint64_t> total_{0};
};

} // namespace llmquant
