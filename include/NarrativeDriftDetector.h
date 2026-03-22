#pragma once

/// @file NarrativeDriftDetector.h
/// @brief Detects concept drift in the LLM bias stream using the
///        Page-Hinkley test — a sequential change-point detector designed
///        for online streams without requiring a window comparison.
///
/// ## Motivation
/// The CUSUM/Page-Hinkley test is the canonical statistical change-point
/// detector for sequential data.  Unlike sliding-window approaches it has
/// no hyper-parameter "lookback" bias — it accumulates evidence of drift
/// without discarding history prematurely.  In narrative analysis terms:
///   - A sustained upward PH statistic indicates the mean bias has drifted
///     upward (bullish narrative takeover).
///   - A sustained downward statistic signals bearish drift.
/// The test resets automatically after each detected change, providing
/// precise change-point timestamps and drift magnitude.
///
/// ## Algorithm
/// m_t = x_t - μ̂ - δ     (δ = allowed slack in each direction)
/// U_t = max(0, U_{t-1} + m_t)   (upward accumulator)
/// D_t = max(0, D_{t-1} - m_t)   (downward accumulator)
/// Alarm when U_t > λ (up-drift) or D_t > λ (down-drift).
/// μ̂ is updated via running mean.  After each alarm, accumulators reset.
///
/// Fires `on_upward_drift(magnitude)` and `on_downward_drift(magnitude)`.
///
/// ## Feature flag
/// `LLMQUANT_ENABLE_NARRATIVE_DRIFT` (default ON).
///
/// Thread-safe.

#include <atomic>
#include <cstdint>
#include <functional>
#include <mutex>
#include <string>

namespace llmquant {

class NarrativeDriftDetector {
public:
    struct Config {
        /// Allowed deviation per step before accumulating evidence. Default 0.02.
        double delta{0.02};

        /// Alarm threshold for accumulated statistic. Default 0.5.
        double lambda{0.5};

        /// Minimum observations before alarming (warm-up). Default 20.
        int min_samples{20};

        /// Fired when upward drift detected (outside lock). Param: U_t at alarm.
        std::function<void(double magnitude)> on_upward_drift;

        /// Fired when downward drift detected (outside lock). Param: D_t at alarm.
        std::function<void(double magnitude)> on_downward_drift;
    };

    explicit NarrativeDriftDetector(Config cfg = Config{});

    // -----------------------------------------------------------------------
    // Data intake
    // -----------------------------------------------------------------------

    void record(double bias);

    // -----------------------------------------------------------------------
    // Accessors (lock-free)
    // -----------------------------------------------------------------------

    /// Current upward accumulator U_t.
    [[nodiscard]] double upward_stat()   const noexcept { return up_.load(std::memory_order_relaxed); }

    /// Current downward accumulator D_t.
    [[nodiscard]] double downward_stat() const noexcept { return dn_.load(std::memory_order_relaxed); }

    /// Running mean of bias.
    [[nodiscard]] double running_mean()  const noexcept { return mean_.load(std::memory_order_relaxed); }

    /// Total upward drift alarms fired.
    [[nodiscard]] uint64_t upward_alarms()   const noexcept { return up_alarms_.load(std::memory_order_relaxed); }

    /// Total downward drift alarms fired.
    [[nodiscard]] uint64_t downward_alarms() const noexcept { return dn_alarms_.load(std::memory_order_relaxed); }

    /// Total observations recorded.
    [[nodiscard]] uint64_t total_records() const noexcept { return total_.load(std::memory_order_relaxed); }

    [[nodiscard]] std::string to_stats_json() const;

    void reset();
    void update_config(const Config& cfg);

private:
    mutable std::mutex mutex_;
    Config cfg_;

    double U_{0.0};       // upward accumulator
    double D_{0.0};       // downward accumulator
    double sum_{0.0};     // running sum for mean
    uint64_t n_{0};       // running count for mean

    std::atomic<double>   up_{0.0};
    std::atomic<double>   dn_{0.0};
    std::atomic<double>   mean_{0.0};
    std::atomic<uint64_t> up_alarms_{0};
    std::atomic<uint64_t> dn_alarms_{0};
    std::atomic<uint64_t> total_{0};
};

} // namespace llmquant
