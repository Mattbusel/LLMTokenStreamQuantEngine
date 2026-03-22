#pragma once

/// @file BayesianSentimentPrior.h
/// @brief Normal-Gamma conjugate Bayesian prior over signal mean and precision.
///
/// ## Motivation
/// Frequentist estimators (EMA, rolling mean) give point estimates.  A Bayesian
/// posterior provides a full distribution over the signal mean, which quantifies
/// UNCERTAINTY — crucial when deciding whether a signal is strong enough to act on.
///
/// The Normal-Gamma prior is conjugate to the Gaussian likelihood, giving a
/// closed-form t-distributed posterior over the mean.  This avoids Monte Carlo.
///
/// ## Algorithm
/// Prior: (μ, τ) ~ NormalGamma(μ₀, κ₀, α₀, β₀)
/// Posterior after n observations {x₁,...,xₙ} with sample mean x̄ and sample variance s²:
///   μₙ = (κ₀μ₀ + n·x̄) / (κ₀ + n)
///   κₙ = κ₀ + n
///   αₙ = α₀ + n/2
///   βₙ = β₀ + n·s²/2 + κ₀·n·(x̄ - μ₀)² / (2(κ₀+n))
///
/// Posterior predictive mean: μₙ
/// Posterior credible interval (95%): μₙ ± t_{0.975, 2αₙ} × sqrt(βₙ / (αₙ·κₙ))
///
/// Fires `on_belief_shift` when posterior mean moves by more than shift_threshold
/// standard deviations of the prior.
///
/// ## Feature flag
/// `LLMQUANT_ENABLE_BAYESIAN_SENTIMENT` (default ON).
///
/// Thread-safe.

#include <atomic>
#include <cstdint>
#include <functional>
#include <mutex>
#include <string>

namespace llmquant {

class BayesianSentimentPrior {
public:
    struct Config {
        /// Prior mean μ₀. Default 0.0 (neutral).
        double prior_mean{0.0};

        /// Prior pseudo-sample count κ₀ (confidence in prior mean). Default 1.0.
        double prior_kappa{1.0};

        /// Prior shape α₀ (>0). Default 1.0.
        double prior_alpha{1.0};

        /// Prior rate β₀ (>0). Default 1.0.
        double prior_beta{1.0};

        /// Fires on_belief_shift when |posterior_mean - prior_mean| exceeds this.
        /// Units: posterior std devs. Default 2.0.
        double shift_threshold{2.0};

        /// Minimum observations before belief shift can fire. Default 5.
        int min_samples{5};

        /// Fired (outside lock) when posterior mean shifts decisively away from prior.
        /// Parameters: (posterior_mean, posterior_std).
        std::function<void(double post_mean, double post_std)> on_belief_shift;

        /// Fired (outside lock) when posterior mean returns within shift_threshold.
        std::function<void()> on_belief_restored;
    };

    explicit BayesianSentimentPrior(Config cfg = Config{});

    // -----------------------------------------------------------------------
    // Data intake
    // -----------------------------------------------------------------------

    void update(double x);

    // -----------------------------------------------------------------------
    // Accessors (lock-free)
    // -----------------------------------------------------------------------

    /// Posterior mean E[μ | data].
    [[nodiscard]] double posterior_mean() const noexcept {
        return post_mean_.load(std::memory_order_relaxed);
    }

    /// Posterior standard deviation of the mean (marginal t-dist std dev).
    [[nodiscard]] double posterior_std() const noexcept {
        return post_std_.load(std::memory_order_relaxed);
    }

    /// 95% credible interval lower bound.
    [[nodiscard]] double ci_lower() const noexcept {
        return ci_lo_.load(std::memory_order_relaxed);
    }

    /// 95% credible interval upper bound.
    [[nodiscard]] double ci_upper() const noexcept {
        return ci_hi_.load(std::memory_order_relaxed);
    }

    /// True when posterior mean has shifted decisively from prior.
    [[nodiscard]] bool is_shifted() const noexcept {
        return shifted_.load(std::memory_order_relaxed);
    }

    [[nodiscard]] uint64_t total_updates() const noexcept {
        return total_.load(std::memory_order_relaxed);
    }

    [[nodiscard]] uint64_t belief_shifts() const noexcept {
        return shifts_.load(std::memory_order_relaxed);
    }

    [[nodiscard]] std::string to_stats_json() const;

    void reset();
    void update_config(const Config& cfg);

private:
    mutable std::mutex mutex_;
    Config             cfg_;

    // Sufficient statistics (updated online)
    uint64_t n_{0};
    double   sum_x_{0.0};
    double   sum_x2_{0.0};

    // Posterior hyperparameters (derived)
    double post_kappa_{0.0};
    double post_alpha_{0.0};
    double post_beta_{0.0};

    bool prev_shifted_{false};

    std::atomic<double>   post_mean_{0.0};
    std::atomic<double>   post_std_{0.0};
    std::atomic<double>   ci_lo_{0.0};
    std::atomic<double>   ci_hi_{0.0};
    std::atomic<bool>     shifted_{false};
    std::atomic<uint64_t> total_{0};
    std::atomic<uint64_t> shifts_{0};

    void recompute_locked();
};

} // namespace llmquant
