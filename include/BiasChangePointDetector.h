#pragma once

/// @file BiasChangePointDetector.h
/// @brief CUSUM sequential change-point detector for LLM bias mean shifts.
///
/// ## Motivation
/// LLM narrative bias can shift abruptly when the model transitions topics
/// (e.g., earnings miss → macro fear). A simple threshold on the current
/// bias value misses gradual drift and generates false alarms on noise.
///
/// The Page-CUSUM algorithm accumulates evidence of a sustained mean shift
/// before alarming, making it optimal (in the SPRT sense) for detecting
/// a shift of known magnitude δ from a known reference mean μ.
///
/// ## Algorithm
/// Two statistics track upside and downside shifts:
///   C+_t = max(0, C+_{t-1} + (x_t − μ − k))
///   C−_t = max(0, C−_{t-1} − (x_t − μ − k))
///
/// where:
///   μ = reference mean (expected neutral bias, default 0.0)
///   k = allowance = δ/2  (half the shift magnitude we want to detect)
///   h = decision threshold (alarm fires when C+ or C− exceeds h)
///
/// After an alarm the accumulator is reset (zero-restart CUSUM).
///
/// ## Feature flag
/// `LLMQUANT_ENABLE_CHANGE_POINT` (default ON).
///
/// Thread-safe.

#include <atomic>
#include <cstdint>
#include <functional>
#include <mutex>
#include <string>

namespace llmquant {

class BiasChangePointDetector {
public:
    struct Config {
        /// Reference mean for the bias series. Default 0.0 (neutral).
        double reference_mean{0.0};

        /// Allowance parameter k = δ/2 where δ is the shift to detect. Default 0.05.
        double allowance{0.05};

        /// Decision threshold h: alarm fires when C+ or C− exceeds this. Default 1.0.
        double threshold{1.0};

        /// Fired when upside shift is detected (C+ > threshold). Receives C+ value.
        std::function<void(double)> on_upshift;

        /// Fired when downside shift is detected (C− > threshold). Receives C− value.
        std::function<void(double)> on_downshift;
    };

    explicit BiasChangePointDetector(Config cfg = Config{});

    // -----------------------------------------------------------------------
    // Data intake
    // -----------------------------------------------------------------------

    /// Record a new bias observation. Returns +1 if upshift alarm, -1 if downshift, 0 otherwise.
    int record(double bias);

    // -----------------------------------------------------------------------
    // Accessors (lock-free)
    // -----------------------------------------------------------------------

    /// Current upside CUSUM accumulator value.
    [[nodiscard]] double cusum_plus() const noexcept {
        return cusum_plus_.load(std::memory_order_relaxed);
    }

    /// Current downside CUSUM accumulator value.
    [[nodiscard]] double cusum_minus() const noexcept {
        return cusum_minus_.load(std::memory_order_relaxed);
    }

    /// Total upside alarms fired.
    [[nodiscard]] uint64_t upshift_count() const noexcept {
        return upshift_count_.load(std::memory_order_relaxed);
    }

    /// Total downside alarms fired.
    [[nodiscard]] uint64_t downshift_count() const noexcept {
        return downshift_count_.load(std::memory_order_relaxed);
    }

    /// Total observations recorded.
    [[nodiscard]] uint64_t total_records() const noexcept {
        return total_.load(std::memory_order_relaxed);
    }

    [[nodiscard]] std::string to_stats_json() const;

    void reset();
    void update_config(const Config& cfg);

private:
    mutable std::mutex mutex_;
    Config cfg_;

    double cusum_p_{0.0};   // C+ accumulator
    double cusum_m_{0.0};   // C− accumulator

    std::atomic<double>   cusum_plus_{0.0};
    std::atomic<double>   cusum_minus_{0.0};
    std::atomic<uint64_t> upshift_count_{0};
    std::atomic<uint64_t> downshift_count_{0};
    std::atomic<uint64_t> total_{0};
};

} // namespace llmquant
