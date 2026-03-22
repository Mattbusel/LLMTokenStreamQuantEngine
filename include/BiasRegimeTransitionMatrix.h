#pragma once

/// @file BiasRegimeTransitionMatrix.h
/// @brief Empirical Markov transition matrix between discrete bias regimes,
///        updated in real-time to estimate next-regime probabilities.
///
/// ## Motivation
/// Sentiment regimes are not i.i.d.: a strong bullish regime is more likely
/// to stay bullish than to jump to bearish.  By maintaining an empirical
/// transition matrix over a rolling window, the engine can compute P(next=X|now=Y)
/// and use that probability to pre-position or hedge before regime flips.
///
/// ## Algorithm
/// Bias values are discretised into `n_regimes` equal-width bins over
/// [range_min, range_max].  Each transition (from bucket i to bucket j) is
/// counted over a rolling window.  Row-normalised to give P(j|i).
///
/// Fires `on_likely_transition(from_regime, to_regime, probability)` when
/// the most probable next regime changes and the probability exceeds `alert_prob`.
///
/// ## Feature flag
/// `LLMQUANT_ENABLE_REGIME_TRANSITION_MATRIX` (default ON).
///
/// Thread-safe.

#include <atomic>
#include <cstdint>
#include <functional>
#include <mutex>
#include <string>
#include <vector>

namespace llmquant {

class BiasRegimeTransitionMatrix {
public:
    static constexpr int kMaxRegimes = 8;

    struct Config {
        /// Number of regime buckets. Default 5.
        int n_regimes{5};

        /// Rolling window of transitions tracked. Default 100.
        int window{100};

        /// Bias range for discretisation. Defaults -1..+1.
        double range_min{-1.0};
        double range_max{1.0};

        /// Fires when most-probable next regime has probability > alert_prob. Default 0.6.
        double alert_prob{0.6};

        /// Fired when likely next regime changes (outside lock).
        /// Parameters: (from_regime, most_likely_next, probability).
        std::function<void(int, int, double)> on_likely_transition;
    };

    explicit BiasRegimeTransitionMatrix(Config cfg = Config{});

    // -----------------------------------------------------------------------
    // Data intake
    // -----------------------------------------------------------------------

    /// Record a new bias value. Updates current regime and transition counts.
    void record(double bias);

    // -----------------------------------------------------------------------
    // Accessors
    // -----------------------------------------------------------------------

    /// Current regime index ∈ [0, n_regimes).
    [[nodiscard]] int current_regime() const noexcept {
        return current_regime_.load(std::memory_order_relaxed);
    }

    /// P(next = j | current = i). Returns 0 if i is unknown.
    [[nodiscard]] double transition_prob(int from_regime, int to_regime) const;

    /// Most likely next regime from current state.
    [[nodiscard]] int most_likely_next() const noexcept {
        return most_likely_next_.load(std::memory_order_relaxed);
    }

    /// Probability of most_likely_next.
    [[nodiscard]] double most_likely_prob() const noexcept {
        return most_likely_prob_.load(std::memory_order_relaxed);
    }

    /// Total observations recorded.
    [[nodiscard]] uint64_t total_records() const noexcept {
        return total_.load(std::memory_order_relaxed);
    }

    [[nodiscard]] std::string to_stats_json() const;

    void reset();
    void update_config(const Config& cfg);

private:
    int bucket_for(double bias) const;
    void recompute_transition_locked();

    mutable std::mutex mutex_;
    Config cfg_;

    // Rolling window of (from, to) transitions as ring buffer
    struct Transition { int from{0}; int to{0}; };
    std::vector<Transition> win_buf_;
    int win_head_{0};
    int win_fill_{0};

    // Count matrix [n_regimes * n_regimes]
    std::vector<int> counts_;

    int  prev_regime_{-1};   // <0 means not set
    int  prev_most_likely_{-1};

    std::atomic<int>      current_regime_{0};
    std::atomic<int>      most_likely_next_{0};
    std::atomic<double>   most_likely_prob_{0.0};
    std::atomic<uint64_t> total_{0};
};

} // namespace llmquant
