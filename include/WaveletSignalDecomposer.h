#pragma once

/// @file WaveletSignalDecomposer.h
/// @brief Multi-resolution Haar wavelet decomposition of LLM sentiment streams.
///
/// ## Motivation
/// Sentiment signals contain structure at multiple time scales: tick-level noise
/// (high-frequency detail) and slow regime drifts (low-frequency approximation).
/// Standard EMA collapses all scales into one number and forces a trade-off.
/// Haar wavelet decomposition separates the signal into J frequency bands simultaneously,
/// letting downstream logic act on fine-grained noise AND coarse trend together.
///
/// ## Algorithm
/// For each filled window of 2^J samples, applies in-place Haar DWT:
///   approx[i] = (x[2i] + x[2i+1]) / 2     (average)
///   detail[i] = (x[2i] - x[2i+1]) / 2     (difference)
/// Iterates J times, shrinking the approximation sub-band by half each pass.
///
/// `detail_energy(level)` = mean squared detail coefficient at that level.
/// High energy at level 0 → rapid tick-level oscillations.
/// High energy at level J-1 → slow multi-period drift.
///
/// Fires `on_high_freq_spike(level, energy)` when detail energy at any level
/// exceeds `spike_threshold`.  Fires `on_spike_clear()` on hysteresis recovery.
///
/// ## Feature flag
/// `LLMQUANT_ENABLE_WAVELET_DECOMPOSER` (default ON).
///
/// Thread-safe.

#include <atomic>
#include <cstdint>
#include <functional>
#include <mutex>
#include <string>
#include <vector>

namespace llmquant {

class WaveletSignalDecomposer {
public:
    struct Config {
        /// Number of Haar decomposition levels (1-8). Window must be >= 2^levels.
        int levels{4};

        /// Sliding window size. Must be a power of two >= 2^levels. Default 64.
        int window{64};

        /// Minimum samples before transform fires. Default == window.
        int min_samples{0};   // 0 → use window

        /// Detail-energy threshold to fire spike callback. Default 0.05.
        double spike_threshold{0.05};

        /// Hysteresis factor: spike clears when max_energy < threshold * clear_hysteresis.
        double clear_hysteresis{0.7};

        /// Fired (outside lock) when detail energy at any level exceeds threshold.
        /// Parameters: (level index, detail energy at that level).
        std::function<void(int level, double energy)> on_high_freq_spike;

        /// Fired (outside lock) when all levels recover below threshold * hysteresis.
        std::function<void()> on_spike_clear;
    };

    explicit WaveletSignalDecomposer(Config cfg = Config{});

    // -----------------------------------------------------------------------
    // Data intake
    // -----------------------------------------------------------------------

    /// Append a new sample.  Transform runs automatically when window is full.
    void record(double x);

    // -----------------------------------------------------------------------
    // Accessors (lock-free)
    // -----------------------------------------------------------------------

    /// Mean of the current window (approximation coefficient after J levels).
    [[nodiscard]] double approx_mean() const noexcept {
        return approx_.load(std::memory_order_relaxed);
    }

    /// Mean-squared detail energy at the given decomposition level (0 = finest).
    /// Returns 0 if level is out of range or no transform has run yet.
    [[nodiscard]] double detail_energy(int level) const noexcept;

    /// Maximum detail energy across all levels (snapshot from last transform).
    [[nodiscard]] double max_detail_energy() const noexcept {
        return max_energy_.load(std::memory_order_relaxed);
    }

    /// True when any level's detail energy exceeds spike_threshold.
    [[nodiscard]] bool is_spiking() const noexcept {
        return spiking_.load(std::memory_order_relaxed);
    }

    /// Total samples seen since last reset.
    [[nodiscard]] uint64_t total_records() const noexcept {
        return total_.load(std::memory_order_relaxed);
    }

    /// Number of spike events fired.
    [[nodiscard]] uint64_t spike_events() const noexcept {
        return spike_events_.load(std::memory_order_relaxed);
    }

    [[nodiscard]] std::string to_stats_json() const;

    void reset();
    void update_config(const Config& cfg);

private:
    mutable std::mutex mutex_;
    Config             cfg_;

    std::vector<double> buf_;        ///< circular input buffer
    int                 head_{0};
    int                 fill_{0};

    bool   prev_spiking_{false};
    int    spike_level_{-1};         ///< level that last triggered

    // Per-level detail energies (updated each transform pass)
    std::vector<double> level_energy_;  // size = levels

    // Atomic shadows for lock-free reads
    std::atomic<double>   approx_{0.0};
    std::atomic<double>   max_energy_{0.0};
    std::atomic<bool>     spiking_{false};
    std::atomic<uint64_t> total_{0};
    std::atomic<uint64_t> spike_events_{0};
    std::vector<std::atomic<double>> energy_atoms_;  // one per level

    void run_transform_locked();
};

} // namespace llmquant
