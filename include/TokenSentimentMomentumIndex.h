#pragma once

/// @file TokenSentimentMomentumIndex.h
/// @brief A composite momentum index for the LLM token sentiment stream,
///        combining rate-of-change, acceleration, and signal strength into
///        a single normalised score.
///
/// ## Motivation
/// Individual metrics (velocity, RSI, autocorrelation) each capture one facet
/// of momentum.  A composite index aggregates them into a single [-1, +1]
/// score that can directly gate or amplify trade signals.  When TSMI > threshold,
/// the engine should lean in; when TSMI < -threshold, fade.
///
/// ## Algorithm
/// TSMI = w_roc * roc_norm + w_acc * acc_norm + w_str * str_norm
///
/// where each component is normalised to [-1, 1]:
///   roc_norm = tanh(velocity / roc_scale)          velocity from first difference
///   acc_norm = tanh(acceleration / acc_scale)       second difference
///   str_norm = (ssi - 50) / 50                     ssi ∈ [0,100] → [-1,1]
///
/// Fires `on_momentum_shift(old_tsmi, new_tsmi)` when TSMI crosses zero.
///
/// ## Feature flag
/// `LLMQUANT_ENABLE_TSMI` (default ON).
///
/// Thread-safe.

#include <atomic>
#include <cstdint>
#include <functional>
#include <mutex>
#include <string>

namespace llmquant {

class TokenSentimentMomentumIndex {
public:
    struct Config {
        /// Weight for rate-of-change component. Default 0.4.
        double w_roc{0.4};

        /// Weight for acceleration component. Default 0.3.
        double w_acc{0.3};

        /// Weight for signal strength (SSI) component. Default 0.3.
        double w_str{0.3};

        /// Scale for tanh(velocity / roc_scale). Default 0.1.
        double roc_scale{0.1};

        /// Scale for tanh(acceleration / acc_scale). Default 0.05.
        double acc_scale{0.05};

        /// Fired when TSMI crosses zero (outside lock).
        /// Parameters: (old_tsmi, new_tsmi).
        std::function<void(double, double)> on_momentum_shift;
    };

    explicit TokenSentimentMomentumIndex(Config cfg = Config{});

    // -----------------------------------------------------------------------
    // Data intake
    // -----------------------------------------------------------------------

    /// Record a new bias value. Updates velocity, acceleration, and SSI estimate.
    void record(double bias);

    // -----------------------------------------------------------------------
    // Accessors (lock-free)
    // -----------------------------------------------------------------------

    /// Current composite TSMI ∈ [-1, 1].
    [[nodiscard]] double tsmi() const noexcept {
        return tsmi_.load(std::memory_order_relaxed);
    }

    /// Rate-of-change (velocity) component.
    [[nodiscard]] double velocity() const noexcept {
        return velocity_.load(std::memory_order_relaxed);
    }

    /// Acceleration component (second difference).
    [[nodiscard]] double acceleration() const noexcept {
        return accel_.load(std::memory_order_relaxed);
    }

    /// Approximate SSI-style strength in [0, 100].
    [[nodiscard]] double signal_strength() const noexcept {
        return ssi_.load(std::memory_order_relaxed);
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

    // Rolling state for RSI-style strength (Wilder, period=14)
    double prev_bias_{0.0};
    double prev_delta_{0.0};
    double avg_gain_{0.0};
    double avg_loss_{0.0};
    int    rsi_warmup_{0};
    bool   seeded_{false};

    double prev_tsmi_{0.0};
    bool   tsmi_seeded_{false};

    std::atomic<double>   tsmi_{0.0};
    std::atomic<double>   velocity_{0.0};
    std::atomic<double>   accel_{0.0};
    std::atomic<double>   ssi_{50.0};
    std::atomic<uint64_t> total_{0};
};

} // namespace llmquant
