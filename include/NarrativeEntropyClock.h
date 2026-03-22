#pragma once

/// @file NarrativeEntropyClock.h
/// @brief Cumulative KL-divergence clock for detecting narrative information exhaustion.
///
/// ## Motivation
/// LLM sentiment streams are not i.i.d.: a narrative (e.g., earnings surprise) arrives
/// in a burst of high-information tokens, then degrades into repetitive, low-information
/// confirmation.  Standard windowed metrics miss this because they treat each token
/// independently.
///
/// The Entropy Clock accumulates the KL divergence of each new token's bias bin from
/// the reference distribution (the long-run empirical distribution).  When the cumulative
/// "information budget" H_budget nats have been consumed, the narrative is considered
/// exhausted and the clock fires `on_budget_exhausted`.  The clock then resets,
/// ready to track the next narrative.
///
/// ## Algorithm
/// Reference distribution q[b]: exponentially decayed histogram of all observed bias bins
///   q_n[b] = (1-β) × q_{n-1}[b] + β × 1_{bin(x_n)==b}   (Jelinek-Mercer smoothing applied)
///
/// At each step, compute incremental KL contribution:
///   ΔKL_n = p_n × log(p_n / q_n)
/// where p_n is the one-hot distribution of the current observation.
/// Simplified: ΔKL_n = -log(q_n[bin(x_n)] + ε)  (surprisal of current bin)
///
/// Accumulated KL: H_accum += ΔKL_n
/// Budget exhausted when H_accum >= H_budget.
///
/// ## Feature flag
/// `LLMQUANT_ENABLE_NARRATIVE_ENTROPY_CLOCK` (default ON).
///
/// Thread-safe.

#include <atomic>
#include <cstdint>
#include <functional>
#include <mutex>
#include <string>
#include <vector>

namespace llmquant {

class NarrativeEntropyClock {
public:
    struct Config {
        /// Number of bias bins for the empirical distribution. Default 20.
        int n_bins{20};

        /// Bias range [range_min, range_max]. Default [-1, 1].
        double range_min{-1.0};
        double range_max{ 1.0};

        /// EMA decay for reference distribution update. Default 0.05.
        double ref_ema_alpha{0.05};

        /// Smoothing epsilon to avoid log(0). Default 1e-6.
        double epsilon{1e-6};

        /// Cumulative KL budget (nats) before exhaustion fires. Default 3.0.
        double budget_nats{3.0};

        /// Minimum samples before budget exhaustion can fire. Default 10.
        int min_samples{10};

        /// Fired (outside lock) when accumulated KL reaches budget_nats.
        /// Parameter: elapsed nats accumulated since last reset.
        std::function<void(double nats)> on_budget_exhausted;

        /// Fired (outside lock) on each step with current ΔKL and total accumulated.
        /// Useful for live monitoring. Set to nullptr to disable (default).
        std::function<void(double delta_kl, double total_kl)> on_kl_update;
    };

    explicit NarrativeEntropyClock(Config cfg = Config{});

    // -----------------------------------------------------------------------
    // Data intake
    // -----------------------------------------------------------------------

    void record(double bias);

    // -----------------------------------------------------------------------
    // Accessors (lock-free)
    // -----------------------------------------------------------------------

    /// Accumulated KL divergence since last reset.
    [[nodiscard]] double accumulated_kl() const noexcept {
        return accum_kl_.load(std::memory_order_relaxed);
    }

    /// Most recent ΔKL contribution (surprisal of last observation).
    [[nodiscard]] double last_delta_kl() const noexcept {
        return last_dkl_.load(std::memory_order_relaxed);
    }

    /// Number of budget-exhaustion events since construction.
    [[nodiscard]] uint64_t exhaustion_events() const noexcept {
        return exhaustion_events_.load(std::memory_order_relaxed);
    }

    /// Total observations processed.
    [[nodiscard]] uint64_t total_records() const noexcept {
        return total_.load(std::memory_order_relaxed);
    }

    [[nodiscard]] std::string to_stats_json() const;

    void reset();
    void update_config(const Config& cfg);

private:
    mutable std::mutex mutex_;
    Config             cfg_;

    std::vector<double> ref_dist_;  ///< smoothed reference distribution q[b]
    double accum_kl_v_{0.0};        ///< current accumulated KL (under lock)

    std::atomic<double>   accum_kl_{0.0};
    std::atomic<double>   last_dkl_{0.0};
    std::atomic<uint64_t> exhaustion_events_{0};
    std::atomic<uint64_t> total_{0};

    int bin_for(double bias) const;
    void init_ref_dist();
};

} // namespace llmquant
