#pragma once

/// @file RealizedVolatilityTracker.h
/// @brief Rolling realized-volatility estimator for the LLM bias signal.
///
/// ## Motivation
/// The magnitude of bias changes (returns) measures how uncertain or volatile
/// the LLM's sentiment is in real-time.  High realized volatility (RV) signals
/// a noisy regime where position sizes should be reduced; low RV indicates a
/// stable trend where larger positions are appropriate.
///
/// ## Algorithm
/// Maintains a rolling window of `window` squared returns:
///   return_t = bias_t − bias_{t-1}
///   RV = sqrt( (1/N) * Σ return_t² )
///
/// The first call seeds the previous-bias state without computing a return.
/// Once the window is full, RV is updated every `record()` call.
///
/// ## Feature flag
/// `LLMQUANT_ENABLE_REALIZED_VOL` (default ON).
///
/// Thread-safe: all public methods are thread-safe.

#include <atomic>
#include <cstdint>
#include <functional>
#include <mutex>
#include <string>
#include <vector>

namespace llmquant {

class RealizedVolatilityTracker {
public:
    struct Config {
        /// Rolling window of squared returns used for RV computation. Default 60.
        int window{60};

        /// When RV exceeds this threshold, `on_high_volatility` fires. Default 0.5.
        double alert_threshold{0.5};

        /// Hysteresis factor: on_low_volatility fires when RV < threshold * hysteresis.
        double clear_hysteresis{0.7};

        /// Fired when RV rises above alert_threshold (once per crossing until cleared).
        std::function<void(double)> on_high_volatility;

        /// Fired when RV falls back below alert_threshold * clear_hysteresis.
        std::function<void(double)> on_low_volatility;
    };

    explicit RealizedVolatilityTracker(Config cfg = Config{});

    // -----------------------------------------------------------------------
    // Core
    // -----------------------------------------------------------------------

    /// Record a new bias observation.  Computes the return from the previous
    /// bias value, updates the rolling RV, and fires callbacks as needed.
    void record(double bias);

    // -----------------------------------------------------------------------
    // Accessors (lock-free)
    // -----------------------------------------------------------------------

    /// Current realized volatility (sqrt of mean squared return). 0 until seeded.
    [[nodiscard]] double realized_vol() const noexcept {
        return rv_.load(std::memory_order_relaxed);
    }

    /// True when RV is currently above alert_threshold.
    [[nodiscard]] bool is_high_volatility() const noexcept {
        return high_vol_flag_.load(std::memory_order_relaxed);
    }

    /// Total observations recorded (including the seeding call).
    [[nodiscard]] uint64_t total_records() const noexcept {
        return total_.load(std::memory_order_relaxed);
    }

    /// Number of high-volatility alert events fired.
    [[nodiscard]] uint64_t alert_events() const noexcept {
        return alert_events_.load(std::memory_order_relaxed);
    }

    // -----------------------------------------------------------------------
    // Mutation
    // -----------------------------------------------------------------------

    void reset();
    void update_config(const Config& cfg);

    // -----------------------------------------------------------------------
    // Diagnostics
    // -----------------------------------------------------------------------

    [[nodiscard]] std::string to_stats_json() const;

private:
    mutable std::mutex mutex_;
    Config cfg_;

    // Internal state (modified under mutex_)
    std::vector<double> sq_returns_;   // ring buffer of squared returns
    int    head_{0};
    int    fill_{0};
    bool   seeded_{false};
    double prev_bias_{0.0};
    double sq_sum_{0.0};
    bool   high_vol_state_{false};

    // Atomic outputs
    std::atomic<double>   rv_{0.0};
    std::atomic<bool>     high_vol_flag_{false};
    std::atomic<uint64_t> total_{0};
    std::atomic<uint64_t> alert_events_{0};
};

} // namespace llmquant
