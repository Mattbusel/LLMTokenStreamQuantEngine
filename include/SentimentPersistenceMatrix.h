#pragma once

/// @file SentimentPersistenceMatrix.h
/// @brief Markov-chain model of discretised sentiment state transitions.
///
/// Divides the bias range into N ordered states (e.g. very-bearish … very-bullish).
/// On each `record(bias)` call the module:
///   1. Maps `bias` to a state index s ∈ [0, N).
///   2. Increments the transition counter matrix C[prev_state][s].
///   3. Re-normalises the row to obtain the stochastic matrix P.
///   4. Advances the current state.
///
/// ## Key outputs
/// - `current_state()` — most recent discretised state.
/// - `predicted_state()` — argmax of the current state's transition row (most
///   likely next state).
/// - `state_probability(int s)` — P[current_state][s] (probability of moving to s).
/// - `transition_probability(int from, int to)` — empirical P[from][to].
/// - `stickiness()` — P[s][s] for the current state (self-transition rate).
/// - `dominant_period()` — expected steps to return to current state (1/π[s]),
///   where π is the empirical stationary distribution.
/// - `on_state_change` callback fires whenever the current state changes.
///
/// ## Stationarity
/// The module maintains a running count vector and computes the normalised
/// marginal distribution as a proxy for π.
///
/// Thread-safe: hot-path reads use atomics; matrix updates use a mutex.

#include <atomic>
#include <cstdint>
#include <functional>
#include <mutex>
#include <string>
#include <vector>

namespace llmquant {

class SentimentPersistenceMatrix {
public:
    static constexpr int kDefaultStates = 5;

    struct Config {
        /// Number of discrete sentiment states (default 5).
        /// States are evenly spaced over [range_min, range_max].
        int n_states{kDefaultStates};

        /// Bias range mapped onto [0, n_states).
        double range_min{-1.0};
        double range_max{ 1.0};

        /// Minimum transitions per row before that row's predictions are used.
        int min_row_count{4};

        /// Fired when the current discretised state changes.
        /// Parameters: (old_state, new_state, transition_probability).
        std::function<void(int old_state, int new_state, double prob)>
            on_state_change;
    };

    SentimentPersistenceMatrix();
    explicit SentimentPersistenceMatrix(const Config& cfg);

    /// Ingest one bias sample and advance the Markov chain.
    void record(double bias);

    // -----------------------------------------------------------------------
    // Accessors (all lock-free)
    // -----------------------------------------------------------------------

    /// Current discretised state index ∈ [0, n_states).
    int current_state() const noexcept;

    /// Most likely next state (argmax of P[current_state]).
    /// Returns -1 if the current row has fewer than min_row_count transitions.
    int predicted_state() const noexcept;

    /// Probability of transitioning from current_state to state s.
    /// Returns 0 if the row is insufficiently populated.
    double state_probability(int s) const noexcept;

    /// Empirical transition probability P[from][to].
    double transition_probability(int from, int to) const noexcept;

    /// Self-transition rate for the current state (P[s][s]).
    double stickiness() const noexcept;

    /// Marginal visit frequency for state s ∈ [0, n_states) (proxy for π[s]).
    double stationary_estimate(int s) const noexcept;

    /// Total samples recorded.
    uint64_t total_records() const noexcept;

    /// Total state changes observed.
    uint64_t state_changes() const noexcept;

    /// Reset transition counts and state.
    void reset();

    /// Reconfigure (resets state).
    void update_config(const Config& cfg);

    /// JSON snapshot for diagnostics.
    std::string to_stats_json() const;

private:
    int bias_to_state(double bias) const noexcept;

    mutable std::mutex mutex_;
    Config cfg_;

    // counts_[from][to] — raw transition counts.
    std::vector<std::vector<double>> counts_;
    // marginal_[s] — total visits to state s.
    std::vector<double> marginal_;

    int  current_state_val_   = 0;
    bool seeded_              = false;

    std::atomic<int>      current_state_{0};
    std::atomic<int>      predicted_state_{-1};
    std::atomic<double>   stickiness_{0.0};
    std::atomic<uint64_t> total_{0};
    std::atomic<uint64_t> changes_{0};
};

} // namespace llmquant
