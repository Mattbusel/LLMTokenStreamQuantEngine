#pragma once

#include "RegimeDetector.h"

#include <array>
#include <atomic>
#include <cstdint>
#include <mutex>
#include <string>

namespace llmquant {

/**
 * @brief Online Markov-chain model of regime transitions.
 *
 * Plugs directly into `RegimeDetector::set_regime_change_callback`.
 * Every time the detected regime changes, this model records the
 * (old → new) transition and updates the 5×5 transition-count matrix.
 * From that count matrix it derives:
 *
 *   - **Transition probability matrix** P(next | current): P[i][j] = count[i][j] / Σ_k count[i][k]
 *   - **Predicted next regime**: argmax_j P[current][j]
 *   - **Steady-state distribution** π: the eigenvector of P with eigenvalue 1,
 *     computed via power-iteration (converges in ~30 iterations).
 *
 * ## Interpretation
 * ```
 * auto pred = model.predict_next(current_regime);
 * if (pred.prob > 0.7 && pred.regime == RegimeDetector::Regime::Bear) {
 *     // High confidence Bear is coming — tighten risk now.
 * }
 * ```
 *
 * ## Integration
 * ```cpp
 * RegimeDetector detector;
 * RegimeTransitionModel model;
 * detector.set_regime_change_callback(
 *     [&](auto next, auto prev) { model.record_transition(prev, next); });
 * ```
 *
 * ## Thread safety
 * All public methods are thread-safe (internal mutex).
 *
 * ## Feature flag
 * Controlled by `LLMQUANT_ENABLE_REGIME_TRANSITION_MODEL` (default ON).
 * Requires LLMQUANT_ENABLE_REGIME_DETECTOR=ON.
 */
class RegimeTransitionModel {
public:
    static constexpr int kRegimes = 5; ///< Number of distinct regimes.

    using Regime = RegimeDetector::Regime;

    /**
     * @brief Prediction for the next regime.
     */
    struct Prediction {
        Regime regime{Regime::Neutral};  ///< Most probable next regime.
        double prob{0.0};                ///< Probability in [0, 1].
        bool   reliable{false};          ///< true if enough data for confidence.
    };

    /**
     * @brief 5×5 transition probability matrix (row = from, col = to).
     */
    using ProbMatrix = std::array<std::array<double, kRegimes>, kRegimes>;

    /**
     * @brief Steady-state distribution over all regimes.
     */
    using SteadyState = std::array<double, kRegimes>;

    /**
     * @brief Minimum transitions before `reliable` is set to true.
     *
     * Default 10.
     */
    int min_transitions_for_reliability{10};

    // -----------------------------------------------------------------------
    // Construction
    // -----------------------------------------------------------------------

    RegimeTransitionModel() = default;

    // -----------------------------------------------------------------------
    // Data ingestion
    // -----------------------------------------------------------------------

    /**
     * @brief Record a regime transition.
     *
     * Pass directly as the `RegimeChangeCallback`:
     * @code
     * detector.set_regime_change_callback(
     *     [&](auto next, auto prev){ model.record_transition(prev, next); });
     * @endcode
     *
     * Thread-safe.
     *
     * @param from Previous regime.
     * @param to   New regime.
     */
    void record_transition(Regime from, Regime to);

    // -----------------------------------------------------------------------
    // Query
    // -----------------------------------------------------------------------

    /**
     * @brief Predict the most likely next regime given the current one.
     *
     * Returns Neutral with prob=0 if the current regime has never transitioned.
     * Thread-safe.
     *
     * @param current The current detected regime.
     * @return Prediction struct with regime, probability, and reliability flag.
     */
    [[nodiscard]] Prediction predict_next(Regime current) const;

    /**
     * @brief Compute the full 5×5 transition probability matrix.
     *
     * Rows sum to 1 (or 0 if a regime has never left).
     * Thread-safe (snapshots counts).
     */
    [[nodiscard]] ProbMatrix transition_probabilities() const;

    /**
     * @brief Compute the steady-state distribution via power iteration.
     *
     * Returns a uniform distribution if fewer than `min_transitions_for_reliability`
     * total transitions have been observed.
     * Thread-safe.
     */
    [[nodiscard]] SteadyState steady_state(int max_iter = 100) const;

    /**
     * @brief Total number of transitions recorded.  Thread-safe (atomic).
     */
    [[nodiscard]] uint64_t total_transitions() const noexcept {
        return total_.load(std::memory_order_relaxed);
    }

    /**
     * @brief Reset all counts.  Thread-safe.
     */
    void reset();

    /**
     * @brief Serialise the full transition probability matrix and steady-state
     *        distribution to a JSON string.  Thread-safe.
     */
    [[nodiscard]] std::string to_stats_json() const;

private:
    mutable std::mutex mutex_;

    // count_[from][to]: number of times we transitioned from->to.
    std::array<std::array<uint32_t, kRegimes>, kRegimes> count_{};

    std::atomic<uint64_t> total_{0};

    [[nodiscard]] static int regime_index(Regime r) noexcept {
        return static_cast<int>(r);
    }
};

} // namespace llmquant
