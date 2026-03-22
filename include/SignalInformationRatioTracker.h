#pragma once

/// @file SignalInformationRatioTracker.h
/// @brief Rolling Information Ratio (IR) for LLM trade signal quality.
///
/// ## Motivation
/// The Information Ratio IR = μ(excess_return) / σ(excess_return) is the
/// canonical quant measure of risk-adjusted signal quality.  Applied here:
/// excess_return = bias - reference (default 0), giving
/// IR = mean(bias) / std(bias) over a rolling window.
///
/// A persistently positive IR indicates the signal is consistently
/// directional above noise; IR near 0 signals random noise.  High
/// |IR| events trigger a callback for downstream sizing logic.
///
/// ## Algorithm
/// Welford online mean/variance over rolling window (circular buffer).
/// IR = μ / σ, clamped to [-10, 10] to prevent division-by-zero spikes.
///
/// ## Feature flag
/// `LLMQUANT_ENABLE_IR_TRACKER` (default ON).
///
/// Thread-safe.

#include <atomic>
#include <cstdint>
#include <functional>
#include <mutex>
#include <string>
#include <vector>

namespace llmquant {

class SignalInformationRatioTracker {
public:
    struct Config {
        /// Rolling window size for IR computation. Default 32.
        int window{32};

        /// Minimum samples before emitting IR. Default 8.
        int min_samples{8};

        /// Reference value subtracted from bias (benchmark). Default 0.0.
        double reference{0.0};

        /// |IR| above this fires on_high_ir. Default 1.5.
        double ir_threshold{1.5};

        /// Hysteresis for clearing. Default 0.75.
        double clear_hysteresis{0.75};

        /// Fired when |IR| rises above ir_threshold. Receives IR value.
        std::function<void(double ir)> on_high_ir;

        /// Fired when |IR| drops back below threshold. Receives IR value.
        std::function<void(double ir)> on_ir_normalized;
    };

    explicit SignalInformationRatioTracker(Config cfg = Config{});

    // -----------------------------------------------------------------------
    // Data intake
    // -----------------------------------------------------------------------

    void record(double bias);

    // -----------------------------------------------------------------------
    // Accessors (lock-free)
    // -----------------------------------------------------------------------

    [[nodiscard]] double ir() const noexcept {
        return ir_.load(std::memory_order_relaxed);
    }

    [[nodiscard]] double mean() const noexcept {
        return mean_.load(std::memory_order_relaxed);
    }

    [[nodiscard]] double stddev() const noexcept {
        return stddev_.load(std::memory_order_relaxed);
    }

    [[nodiscard]] bool is_high_ir() const noexcept {
        return high_ir_.load(std::memory_order_relaxed);
    }

    [[nodiscard]] uint64_t high_ir_events() const noexcept {
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

    std::vector<double> buf_;
    int head_{0};
    int fill_{0};

    bool prev_high_ir_{false};

    std::atomic<double>   ir_{0.0};
    std::atomic<double>   mean_{0.0};
    std::atomic<double>   stddev_{0.0};
    std::atomic<bool>     high_ir_{false};
    std::atomic<uint64_t> events_{0};
    std::atomic<uint64_t> total_{0};

    void recompute_locked();
};

} // namespace llmquant
