#pragma once

/// @file SignalRegimeHMM.h
/// @brief Two-state Hidden Markov Model for online bias-regime classification.
///
/// ## Motivation
/// Explicit regime detection via thresholds (BiasHysteresisGate, RegimeDetector)
/// is brittle.  A two-state HMM models the latent regime as a hidden variable
/// with probabilistic transitions and Gaussian emission from each state.
/// This provides a soft probability P(regime=bullish | observations) updated
/// online via a scalar Kalman filter approximation to the forward algorithm.
///
/// ## Algorithm (online approximation)
///   States: S=0 (bearish), S=1 (bullish).
///   Emission: P(x | S=s) = N(x; μ_s, σ_s²).
///   Transition matrix A: A[i][j] = P(next=j | current=i).
///   Forward: α_t[s] = P(observations_1..t, S_t=s).
///   Online update at each observation x_t:
///     α_pred[j] = Σ_i α_{t-1}[i] * A[i][j]
///     α_t[j]    = α_pred[j] * N(x_t; μ_j, σ_j²)
///     Normalise: γ_t[j] = α_t[j] / Z_t
///   P(bullish) = γ_t[1].
///   State means/variances are fixed (user-provided or default).
///
/// Fires `on_regime_change(from, to)` when argmax state flips.
///
/// ## Feature flag
/// `LLMQUANT_ENABLE_REGIME_HMM` (default ON).
///
/// Thread-safe.

#include <array>
#include <atomic>
#include <cstdint>
#include <functional>
#include <mutex>
#include <string>

namespace llmquant {

class SignalRegimeHMM {
public:
    static constexpr int kStates = 2; // 0=bearish, 1=bullish

    struct Config {
        /// Emission mean for each state [bearish, bullish]. Default {-0.1, +0.1}.
        std::array<double, kStates> state_mean{-0.10, 0.10};

        /// Emission std dev for each state. Default {0.15, 0.15}.
        std::array<double, kStates> state_std{0.15, 0.15};

        /// Transition probabilities A[from][to]. Default: high self-transition 0.95.
        std::array<std::array<double, kStates>, kStates> transition{{
            {0.95, 0.05},   // bearish stays bearish 95%, switches 5%
            {0.05, 0.95}    // bullish stays bullish 95%, switches 5%
        }};

        /// Initial state probabilities [bearish, bullish]. Default uniform.
        std::array<double, kStates> init_prob{0.5, 0.5};

        /// Minimum observations before classifying. Default 5.
        int min_samples{5};

        /// P(bullish) threshold above which state is "bullish". Default 0.6.
        double bullish_threshold{0.60};

        /// Fired when hard-classified state changes (outside lock). Params: from, to (0/1).
        std::function<void(int from, int to)> on_regime_change;
    };

    explicit SignalRegimeHMM(Config cfg = Config{});

    // -----------------------------------------------------------------------
    // Data intake
    // -----------------------------------------------------------------------

    void record(double bias);

    // -----------------------------------------------------------------------
    // Accessors (lock-free)
    // -----------------------------------------------------------------------

    /// Posterior probability of bullish regime P(S=bullish | data).
    [[nodiscard]] double p_bullish() const noexcept {
        return p_bull_.load(std::memory_order_relaxed);
    }

    /// Posterior probability of bearish regime.
    [[nodiscard]] double p_bearish() const noexcept {
        return 1.0 - p_bull_.load(std::memory_order_relaxed);
    }

    /// Hard-classified current state: 0=bearish, 1=bullish.
    [[nodiscard]] int current_state() const noexcept {
        return state_.load(std::memory_order_relaxed);
    }

    /// Total regime-change events.
    [[nodiscard]] uint64_t regime_changes() const noexcept {
        return changes_.load(std::memory_order_relaxed);
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

    std::array<double, kStates> alpha_{0.5, 0.5}; // forward variable (normalised)
    int prev_state_{-1};

    std::atomic<double>   p_bull_{0.5};
    std::atomic<int>      state_{-1};
    std::atomic<uint64_t> changes_{0};
    std::atomic<uint64_t> total_{0};
};

} // namespace llmquant
