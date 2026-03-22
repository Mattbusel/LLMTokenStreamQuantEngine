#pragma once

/// @file SignalResidualAnalyser.h
/// @brief Computes and analyses residuals from an AR(1) prediction of the bias
///        stream, providing a Durbin-Watson autocorrelation proxy for residual
///        structure diagnostics.
///
/// ## Motivation
/// After removing the AR(1) predictable component (SignalAutoregressor uses
/// AR(p) with RLS; this module uses a simpler EMA-based AR(1) approximation
/// for speed), the residuals should be white noise if the model is adequate.
/// The Durbin-Watson statistic DW ≈ 2(1 − ρ_1) measures first-order residual
/// autocorrelation:
///   DW ≈ 0:   strong positive autocorrelation (model under-fits)
///   DW ≈ 2:   no autocorrelation (well-specified model)
///   DW ≈ 4:   strong negative autocorrelation (over-differencing)
///
/// Fires `on_residual_autocorr(dw)` when DW moves outside [dw_lo, dw_hi].
///
/// ## Feature flag
/// `LLMQUANT_ENABLE_RESIDUAL_ANALYSER` (default ON).
///
/// Thread-safe.

#include <atomic>
#include <cstdint>
#include <functional>
#include <mutex>
#include <string>
#include <vector>

namespace llmquant {

class SignalResidualAnalyser {
public:
    struct Config {
        /// EMA alpha for AR(1) coefficient estimation. Default 0.10.
        double ar_alpha{0.10};

        /// Rolling window of residuals for DW computation. Default 50.
        int window{50};

        /// Minimum fill before publishing DW. Default 10.
        int min_samples{10};

        /// DW outside [dw_lo, dw_hi] fires on_residual_autocorr. Defaults 1.5 / 2.5.
        double dw_lo{1.5};
        double dw_hi{2.5};

        /// Fired when DW leaves acceptable range (outside lock).
        std::function<void(double dw)> on_residual_autocorr;
    };

    explicit SignalResidualAnalyser(Config cfg = Config{});

    // -----------------------------------------------------------------------
    // Data intake
    // -----------------------------------------------------------------------

    void record(double bias);

    // -----------------------------------------------------------------------
    // Accessors (lock-free)
    // -----------------------------------------------------------------------

    /// Most recent Durbin-Watson statistic.
    [[nodiscard]] double durbin_watson() const noexcept {
        return dw_.load(std::memory_order_relaxed);
    }

    /// Current AR(1) coefficient estimate (EMA-based).
    [[nodiscard]] double ar1_coeff() const noexcept {
        return ar1_.load(std::memory_order_relaxed);
    }

    /// Most recent residual ε_t = x_t − â₁ x_{t-1}.
    [[nodiscard]] double last_residual() const noexcept {
        return last_resid_.load(std::memory_order_relaxed);
    }

    /// Total DW-alarm events fired.
    [[nodiscard]] uint64_t alarm_events() const noexcept {
        return alarms_.load(std::memory_order_relaxed);
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

    double ar1_est_{0.0};    // EMA-estimated AR(1) coefficient
    double prev_bias_{0.0};  // x_{t-1}
    bool   seeded_{false};

    // Ring buffer of residuals for DW calculation
    std::vector<double> resid_buf_;
    int resid_head_{0};
    int resid_fill_{0};

    bool prev_dw_ok_{true};

    std::atomic<double>   dw_{2.0};
    std::atomic<double>   ar1_{0.0};
    std::atomic<double>   last_resid_{0.0};
    std::atomic<uint64_t> alarms_{0};
    std::atomic<uint64_t> total_{0};
};

} // namespace llmquant
