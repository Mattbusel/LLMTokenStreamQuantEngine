#pragma once
/// NarrativeRegimeMarkov — Markov chain regime model over discretised bias.
/// Bins bias into n_regimes states. Builds empirical transition matrix via count updates.
/// Tracks current state and steady-state probability vector (dominant eigenvector).
/// Fires on_regime_change when the current state changes.
///
/// Feature flag: LLMQUANT_ENABLE_REGIME_MARKOV / LLMQUANT_REGIME_MARKOV_ENABLED

#ifdef LLMQUANT_REGIME_MARKOV_ENABLED

#include <atomic>
#include <functional>
#include <mutex>
#include <string>
#include <vector>

namespace llmquant {

class NarrativeRegimeMarkov {
public:
    struct Config {
        int    n_regimes   = 3;    ///< number of discrete states (bins)
        double bias_range  = 0.1;  ///< half-width of each regime bin
        int    min_samples = 5;    ///< minimum transitions before firing events

        /// Fires when the current regime changes.
        std::function<void(int old_state, int new_state)> on_regime_change;
    };

    explicit NarrativeRegimeMarkov(Config cfg = {});

    void record(double bias);

    int current_state()              const noexcept;
    double steady_state(int regime)  const;       ///< steady-state probability for regime
    double transition(int from, int to) const;    ///< empirical transition probability

    std::size_t regime_change_events() const noexcept;
    std::size_t total_records()        const noexcept;

    std::string to_stats_json() const;
    void reset();
    void update_config(const Config& cfg);

private:
    mutable std::mutex mutex_;
    Config cfg_;

    std::vector<std::vector<double>> trans_counts_; // transition counts[from][to]
    std::vector<double> steady_state_;              // estimated steady state
    int    prev_state_ = -1;

    std::atomic<int>         current_state_{0};
    std::atomic<std::size_t> change_events_{0};
    std::atomic<std::size_t> total_{0};

    int bias_to_state(double bias) const;
    void update_steady_state();
};

} // namespace llmquant

#endif // LLMQUANT_REGIME_MARKOV_ENABLED
