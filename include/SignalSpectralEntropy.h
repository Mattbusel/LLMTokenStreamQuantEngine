#pragma once

/// @file SignalSpectralEntropy.h
/// @brief Computes the spectral entropy of the bias stream as a measure of
///        signal complexity in the frequency domain.
///
/// ## Motivation
/// Spectral entropy H_s = -Σ p_k log2(p_k) where p_k = |X_k|² / Σ|X_j|²
/// is the normalised power spectral density.  Unlike time-domain entropy,
/// spectral entropy captures frequency-domain structure:
///   - Low H_s → power concentrated at few frequencies (periodic/trending).
///   - High H_s (approaching log2(N/2)) → white noise, all frequencies equal.
///
/// Uses DCT-II (same as BiasFrequencyAnalyser) over a rolling window so the
/// two modules are complementary: BiasFrequencyAnalyser finds dominant freq;
/// SignalSpectralEntropy measures how concentrated the power is overall.
///
/// Fires `on_entropy_change(old, new)` when spectral entropy shifts by more
/// than `change_threshold`, signalling a frequency-domain regime change.
///
/// ## Feature flag
/// `LLMQUANT_ENABLE_SPECTRAL_ENTROPY` (default ON).
///
/// Thread-safe.

#include <atomic>
#include <cstdint>
#include <functional>
#include <mutex>
#include <string>
#include <vector>

namespace llmquant {

class SignalSpectralEntropy {
public:
    struct Config {
        /// Rolling window length (should be power-of-2 for efficiency). Default 64.
        int window{64};

        /// Minimum fill before computing. Default 16.
        int min_samples{16};

        /// |ΔH_s| above which on_entropy_change fires. Default 0.2.
        double change_threshold{0.2};

        /// Fired when spectral entropy changes significantly (outside lock).
        std::function<void(double old_hs, double new_hs)> on_entropy_change;
    };

    explicit SignalSpectralEntropy(Config cfg = Config{});

    // -----------------------------------------------------------------------
    // Data intake
    // -----------------------------------------------------------------------

    void record(double bias);

    // -----------------------------------------------------------------------
    // Accessors (lock-free)
    // -----------------------------------------------------------------------

    /// Current spectral entropy H_s (bits). Max = log2(window/2).
    [[nodiscard]] double spectral_entropy() const noexcept {
        return hs_.load(std::memory_order_relaxed);
    }

    /// Normalised spectral entropy ∈ [0, 1]. 1 = white noise.
    [[nodiscard]] double normalised_entropy() const noexcept {
        return hs_norm_.load(std::memory_order_relaxed);
    }

    /// Total entropy-change events fired.
    [[nodiscard]] uint64_t change_events() const noexcept {
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
    void compute_spectral_entropy_locked();

    mutable std::mutex mutex_;
    Config cfg_;

    std::vector<double> win_buf_;
    int win_head_{0};
    int win_fill_{0};

    double prev_hs_{-1.0};

    std::atomic<double>   hs_{0.0};
    std::atomic<double>   hs_norm_{0.0};
    std::atomic<uint64_t> change_events_{0};
    std::atomic<uint64_t> total_{0};
};

} // namespace llmquant
