#pragma once

/// @file NarrativeConsistencyMeter.h
/// @brief Measures internal consistency of the LLM bias stream.
///
/// ## Motivation
/// A consistent narrative produces slowly-varying bias values; an
/// inconsistent model output produces large erratic jumps.
/// The Average Absolute Deviation (AAD) of consecutive bias differences
/// captures this: small AAD = smooth narrative, large AAD = chaotic output.
///
/// When consistency_score (= 1 − min(AAD/normalizer, 1)) drops below
/// a threshold, the signal quality is deemed unreliable and
/// on_inconsistent fires.
///
/// ## Algorithm
/// Δ_t = |bias_t − bias_{t-1}|
/// AAD = mean(Δ) over rolling window.
/// consistency_score = max(0, 1 − AAD / normalizer)   ∈ [0, 1]
///
/// ## Feature flag
/// `LLMQUANT_ENABLE_CONSISTENCY_METER` (default ON).
///
/// Thread-safe.

#include <atomic>
#include <cstdint>
#include <functional>
#include <mutex>
#include <string>
#include <vector>

namespace llmquant {

class NarrativeConsistencyMeter {
public:
    struct Config {
        /// Rolling window for AAD computation. Default 32.
        int window{32};

        /// Minimum observations before reporting. Default 4.
        int min_samples{4};

        /// Normalizer: AAD above this = score 0. Default 0.3.
        double normalizer{0.3};

        /// Consistency score below this fires on_inconsistent. Default 0.4.
        double consistency_threshold{0.4};

        /// Hysteresis for clearing. Default 0.85.
        double clear_hysteresis{0.85};

        /// Fired when consistency_score drops below threshold (rising edge).
        std::function<void(double score)> on_inconsistent;

        /// Fired when consistency_score recovers above threshold.
        std::function<void(double score)> on_recovered;
    };

    explicit NarrativeConsistencyMeter(Config cfg = Config{});

    // -----------------------------------------------------------------------
    // Data intake
    // -----------------------------------------------------------------------

    void record(double bias);

    // -----------------------------------------------------------------------
    // Accessors (lock-free)
    // -----------------------------------------------------------------------

    /// Consistency score ∈ [0, 1]. 1 = perfectly smooth narrative.
    [[nodiscard]] double consistency_score() const noexcept {
        return score_.load(std::memory_order_relaxed);
    }

    /// Mean |Δbias| over the rolling window.
    [[nodiscard]] double aad() const noexcept {
        return aad_.load(std::memory_order_relaxed);
    }

    /// True when consistency score < threshold.
    [[nodiscard]] bool is_inconsistent() const noexcept {
        return inconsistent_.load(std::memory_order_relaxed);
    }

    /// Rising-edge events (inconsistency state entered).
    [[nodiscard]] uint64_t inconsistency_events() const noexcept {
        return events_.load(std::memory_order_relaxed);
    }

    [[nodiscard]] uint64_t total_records() const noexcept {
        return total_.load(std::memory_order_relaxed);
    }

    [[nodiscard]] std::string to_stats_json() const;

    void reset();
    void update_config(const Config& cfg);

private:
    mutable std::mutex mutex_;
    Config cfg_;

    std::vector<double> delta_buf_;  // ring of |Δbias| values
    int head_{0};
    int fill_{0};
    double delta_sum_{0.0};

    double prev_bias_{0.0};
    bool   seeded_{false};
    bool   prev_inconsistent_{false};

    std::atomic<double>   score_{1.0};
    std::atomic<double>   aad_{0.0};
    std::atomic<bool>     inconsistent_{false};
    std::atomic<uint64_t> events_{0};
    std::atomic<uint64_t> total_{0};
};

} // namespace llmquant
