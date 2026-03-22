#pragma once

/// @file SignalAutoregressor.h
/// @brief Online AR(p) model of the bias stream fitted via Recursive Least
///        Squares (RLS) with exponential forgetting.
///
/// ## Motivation
/// A bias stream with detectable auto-correlation can be modelled as an
/// AR(p) process.  The one-step-ahead prediction error measures how much
/// unexplained variance the current observation carries — a spike in error
/// signals a regime change, a news shock, or structural break.
///
/// ## Algorithm
/// Model:  x[t] = a1*x[t-1] + a2*x[t-2] + ... + ap*x[t-p] + ε[t]
/// RLS update (λ = forgetting factor):
///   k   = P_prev * φ / (λ + φᵀ P_prev φ)
///   a   = a_prev + k * (x[t] - φᵀ a_prev)
///   P   = (I - k φᵀ) P_prev / λ
/// where φ = [x[t-1], ..., x[t-p]]ᵀ.
///
/// Fires `on_prediction_error_spike(pred_error)` when the absolute
/// prediction error exceeds `error_threshold`.
///
/// ## Feature flag
/// `LLMQUANT_ENABLE_AUTOREGRESSOR` (default ON).
///
/// Thread-safe.

#include <atomic>
#include <cstdint>
#include <functional>
#include <mutex>
#include <string>
#include <vector>

namespace llmquant {

class SignalAutoregressor {
public:
    struct Config {
        /// AR model order p (number of lags). Default 4.
        int order{4};

        /// RLS forgetting factor λ ∈ (0, 1]. Default 0.97.
        double lambda{0.97};

        /// Initial diagonal value for P (covariance initialisation). Default 1.0.
        double p_init{1.0};

        /// Minimum observations before computing (must fill lag buffer). Default = order+1.
        int min_samples{-1};  // <0 → use order+1

        /// |prediction error| above this fires on_prediction_error_spike. Default 0.2.
        double error_threshold{0.2};

        /// Fired when |error| > error_threshold (outside lock).
        std::function<void(double pred_error)> on_prediction_error_spike;
    };

    explicit SignalAutoregressor(Config cfg = Config{});

    // -----------------------------------------------------------------------
    // Data intake
    // -----------------------------------------------------------------------

    void record(double bias);

    // -----------------------------------------------------------------------
    // Accessors (lock-free)
    // -----------------------------------------------------------------------

    /// Most recent one-step-ahead prediction error (x[t] - x̂[t]).
    [[nodiscard]] double last_prediction_error() const noexcept {
        return pred_error_.load(std::memory_order_relaxed);
    }

    /// Most recent one-step-ahead prediction.
    [[nodiscard]] double last_prediction() const noexcept {
        return pred_.load(std::memory_order_relaxed);
    }

    /// Total spike events fired.
    [[nodiscard]] uint64_t spike_events() const noexcept {
        return spike_events_.load(std::memory_order_relaxed);
    }

    /// Total observations recorded.
    [[nodiscard]] uint64_t total_records() const noexcept {
        return total_.load(std::memory_order_relaxed);
    }

    [[nodiscard]] std::string to_stats_json() const;

    void reset();
    void update_config(const Config& cfg);

private:
    void init_rls();

    mutable std::mutex mutex_;
    Config cfg_;

    int p_{4};                        // effective order
    std::vector<double> a_;           // AR coefficients [p]
    std::vector<double> P_;           // p×p covariance (row-major)
    std::vector<double> lag_buf_;     // ring buffer of last p observations
    int lag_head_{0};
    int lag_fill_{0};

    std::atomic<double>   pred_{0.0};
    std::atomic<double>   pred_error_{0.0};
    std::atomic<uint64_t> spike_events_{0};
    std::atomic<uint64_t> total_{0};
};

} // namespace llmquant
