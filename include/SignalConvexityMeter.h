#pragma once

/// @file SignalConvexityMeter.h
/// @brief Second-derivative regime classifier for signal streams.
///
/// ## Motivation
/// First-derivative (momentum) signals are well-studied.  The SECOND derivative
/// — the rate-of-change of momentum — reveals whether a trend is accelerating
/// (convex: positive d²) or decelerating/reversing (concave: negative d²).
///
/// In LLM sentiment trading:
///   - Accelerating positive sentiment = emerging narrative, momentum trade.
///   - Decelerating positive sentiment = narrative peak, mean-reversion trade.
///   - Stable (|d²| < threshold) = regime uncertainty, reduce position.
///
/// ## Algorithm
/// Maintains an EMA of the first difference (d1) and a second EMA of the
/// first difference of d1 (d2):
///   d1_n = α_fast × (x_n - x_{n-1}) + (1-α_fast) × d1_{n-1}
///   d2_n = α_slow × (d1_n - d1_{n-1}_raw) + (1-α_slow) × d2_{n-1}
///
/// Regime:
///   ACCELERATING  if d2 >  convex_threshold
///   DECELERATING  if d2 < -convex_threshold
///   STABLE        otherwise
///
/// ## Feature flag
/// `LLMQUANT_ENABLE_SIGNAL_CONVEXITY` (default ON).
///
/// Thread-safe.

#include <atomic>
#include <cstdint>
#include <functional>
#include <mutex>
#include <string>

namespace llmquant {

class SignalConvexityMeter {
public:
    enum class Regime { STABLE, ACCELERATING, DECELERATING };

    struct Config {
        /// EMA alpha for first-derivative smoothing. Default 0.3.
        double alpha_d1{0.3};

        /// EMA alpha for second-derivative smoothing. Default 0.15.
        double alpha_d2{0.15};

        /// Threshold for non-stable classification. Default 0.005.
        double convex_threshold{0.005};

        /// Minimum samples before regime can be non-STABLE. Default 5.
        int min_samples{5};

        /// Fired (outside lock) when regime enters ACCELERATING.
        /// Parameter: current d2 value.
        std::function<void(double d2)> on_accelerating;

        /// Fired (outside lock) when regime enters DECELERATING.
        /// Parameter: current d2 value.
        std::function<void(double d2)> on_decelerating;

        /// Fired (outside lock) when regime returns to STABLE from non-STABLE.
        std::function<void()> on_stabilized;
    };

    explicit SignalConvexityMeter(Config cfg = Config{});

    // -----------------------------------------------------------------------
    // Data intake
    // -----------------------------------------------------------------------

    void record(double x);

    // -----------------------------------------------------------------------
    // Accessors (lock-free)
    // -----------------------------------------------------------------------

    [[nodiscard]] double first_derivative() const noexcept {
        return d1_.load(std::memory_order_relaxed);
    }

    [[nodiscard]] double second_derivative() const noexcept {
        return d2_.load(std::memory_order_relaxed);
    }

    [[nodiscard]] Regime regime() const noexcept {
        return static_cast<Regime>(regime_.load(std::memory_order_relaxed));
    }

    [[nodiscard]] bool is_accelerating() const noexcept {
        return regime() == Regime::ACCELERATING;
    }

    [[nodiscard]] bool is_decelerating() const noexcept {
        return regime() == Regime::DECELERATING;
    }

    [[nodiscard]] uint64_t total_records() const noexcept {
        return total_.load(std::memory_order_relaxed);
    }

    [[nodiscard]] uint64_t regime_changes() const noexcept {
        return regime_changes_.load(std::memory_order_relaxed);
    }

    [[nodiscard]] std::string to_stats_json() const;

    void reset();
    void update_config(const Config& cfg);

private:
    mutable std::mutex mutex_;
    Config             cfg_;

    double prev_x_{0.0};
    double d1_v_{0.0};      ///< current EMA of first differences
    double d1_prev_{0.0};   ///< previous step d1 (for second difference)
    double d2_v_{0.0};      ///< current EMA of second differences
    bool   seeded_{false};
    bool   d1_seeded_{false};
    Regime prev_regime_{Regime::STABLE};

    std::atomic<double>   d1_{0.0};
    std::atomic<double>   d2_{0.0};
    std::atomic<int>      regime_{static_cast<int>(Regime::STABLE)};
    std::atomic<uint64_t> total_{0};
    std::atomic<uint64_t> regime_changes_{0};
};

} // namespace llmquant
