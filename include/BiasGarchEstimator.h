#pragma once

/// @file BiasGarchEstimator.h
/// @brief GARCH(1,1) conditional volatility estimator for the LLM bias stream.
///
/// ## Motivation
/// Standard rolling-window volatility treats all observations equally and has
/// no memory of volatility clustering.  GARCH(1,1) captures the stylised fact
/// that high-volatility regimes persist: today's variance depends on yesterday's
/// variance and yesterday's squared innovation.
///
///   σ²_t = ω + α ε²_{t-1} + β σ²_{t-1}
///
/// This provides:
///   - A forward-looking variance estimate σ²_t for risk sizing.
///   - A volatility-surprise signal when ε²_{t-1} >> σ²_{t-1}.
///   - A persistence indicator (α + β → 1 means long-memory volatility).
///
/// Parameters are not estimated by MLE (which requires iterative optimisation);
/// instead sensible defaults are provided and the user can supply calibrated
/// values.  The `on_vol_spike` callback fires when σ_t jumps by more than
/// `vol_spike_ratio` times the previous σ.
///
/// ## Feature flag
/// `LLMQUANT_ENABLE_GARCH_ESTIMATOR` (default ON).
///
/// Thread-safe.

#include <atomic>
#include <cstdint>
#include <functional>
#include <mutex>
#include <string>

namespace llmquant {

class BiasGarchEstimator {
public:
    struct Config {
        /// GARCH omega (base variance floor). Default 0.0001.
        double omega{0.0001};

        /// GARCH alpha (ARCH term, shock persistence). Default 0.10.
        double alpha{0.10};

        /// GARCH beta (GARCH term, variance persistence). Default 0.85.
        double beta{0.85};

        /// Initial variance estimate σ²_0. Default 0.01.
        double sigma2_init{0.01};

        /// Minimum observations before publishing estimates. Default 5.
        int min_samples{5};

        /// Fires on_vol_spike when σ_t / σ_{t-1} > vol_spike_ratio. Default 1.5.
        double vol_spike_ratio{1.5};

        /// Fired when conditional vol spikes (outside lock). Param: new σ_t.
        std::function<void(double sigma)> on_vol_spike;
    };

    explicit BiasGarchEstimator(Config cfg = Config{});

    // -----------------------------------------------------------------------
    // Data intake
    // -----------------------------------------------------------------------

    void record(double bias);

    // -----------------------------------------------------------------------
    // Accessors (lock-free)
    // -----------------------------------------------------------------------

    /// Current conditional standard deviation σ_t.
    [[nodiscard]] double conditional_vol() const noexcept {
        return sigma_.load(std::memory_order_relaxed);
    }

    /// Current conditional variance σ²_t.
    [[nodiscard]] double conditional_var() const noexcept {
        return sigma2_.load(std::memory_order_relaxed);
    }

    /// Last innovation ε_{t-1} = bias_{t-1} − μ̂.
    [[nodiscard]] double last_innovation() const noexcept {
        return eps_.load(std::memory_order_relaxed);
    }

    /// Volatility persistence (α + β).  Values near 1.0 = long-memory regime.
    [[nodiscard]] double persistence() const noexcept {
        return cfg_.alpha + cfg_.beta;
    }

    /// Total vol-spike events fired.
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
    mutable std::mutex mutex_;
    Config cfg_;

    double sigma2_{0.01};  // current conditional variance
    double mu_{0.0};       // running mean (EMA, α=0.05)
    double prev_sigma_{0.0};

    std::atomic<double>   sigma_{0.0};
    std::atomic<double>   sigma2_a_{0.01};
    std::atomic<double>   eps_{0.0};
    std::atomic<uint64_t> spike_events_{0};
    std::atomic<uint64_t> total_{0};
};

} // namespace llmquant
