#pragma once

/// @file SignalPolarizationMonitor.h
/// @brief Bimodality detector for the sentiment signal distribution.
///
/// ## Motivation
/// Most risk models assume unimodal signal distributions (roughly Gaussian).
/// When the market is genuinely divided — bulls and bears pulling in opposite
/// directions simultaneously — the signal distribution becomes bimodal.
/// This is a leading indicator of regime instability and elevated volatility.
///
/// ## Algorithm
/// Uses Sarle's bimodality coefficient:
///   BC = (γ² + 1) / (κ + 3·(n−1)²/((n−2)·(n−3)))
/// where γ = skewness, κ = excess kurtosis, n = sample count.
/// BC ∈ [0, 1].  BC > 5/9 ≈ 0.556 → bimodal (uniform distribution limit).
/// BC > threshold (default 0.6) → polarized.
///
/// Fires `on_polarized(bc)` when BC crosses above the threshold.
/// Fires `on_unified(bc)` when BC drops back below threshold × hysteresis.
///
/// ## Feature flag
/// `LLMQUANT_ENABLE_POLARIZATION_MONITOR` (default ON).
///
/// Thread-safe.

#include <atomic>
#include <cstdint>
#include <functional>
#include <mutex>
#include <string>
#include <vector>

namespace llmquant {

class SignalPolarizationMonitor {
public:
    struct Config {
        /// Rolling window size. Default 64.
        int window{64};

        /// BC threshold above which distribution is considered bimodal. Default 0.6.
        double polarization_threshold{0.6};

        /// Hysteresis: clear fires when BC < threshold × hysteresis. Default 0.85.
        double clear_hysteresis{0.85};

        /// Minimum observations before computing BC. Default 8.
        int min_observations{8};

        /// Fired when BC crosses above polarization_threshold.
        /// Parameter: current BC value.
        std::function<void(double bc)> on_polarized;

        /// Fired when BC drops back below threshold × hysteresis after being polarized.
        /// Parameter: current BC value.
        std::function<void(double bc)> on_unified;
    };

    explicit SignalPolarizationMonitor(Config cfg = Config{});

    // -----------------------------------------------------------------------
    // Data intake
    // -----------------------------------------------------------------------

    /// Record a new signal bias observation.
    void record(double bias);

    // -----------------------------------------------------------------------
    // Accessors (lock-free where possible)
    // -----------------------------------------------------------------------

    /// Bimodality coefficient ∈ [0, 1]. Returns 0 if insufficient data.
    [[nodiscard]] double bimodality_coefficient() const noexcept {
        return bc_.load(std::memory_order_relaxed);
    }

    /// True when BC > polarization_threshold.
    [[nodiscard]] bool is_polarized() const noexcept {
        return polarized_.load(std::memory_order_relaxed);
    }

    /// Number of times polarization threshold was crossed (rising edge only).
    [[nodiscard]] uint64_t polarization_events() const noexcept {
        return events_.load(std::memory_order_relaxed);
    }

    /// Total observations recorded.
    [[nodiscard]] uint64_t total_records() const noexcept {
        return total_.load(std::memory_order_relaxed);
    }

    /// Skewness of the current window (signed).
    [[nodiscard]] double skewness() const noexcept {
        return skewness_.load(std::memory_order_relaxed);
    }

    /// Excess kurtosis of the current window.
    [[nodiscard]] double excess_kurtosis() const noexcept {
        return kurtosis_.load(std::memory_order_relaxed);
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

    bool prev_polarized_{false};

    std::atomic<double>   bc_{0.0};
    std::atomic<double>   skewness_{0.0};
    std::atomic<double>   kurtosis_{0.0};
    std::atomic<bool>     polarized_{false};
    std::atomic<uint64_t> events_{0};
    std::atomic<uint64_t> total_{0};
};

} // namespace llmquant
