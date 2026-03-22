#pragma once

#include <atomic>
#include <cstdint>
#include <functional>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

namespace llmquant {

/**
 * @brief Online gradient-descent meta-learner that fuses multiple named signal sources.
 *
 * Each registered source contributes a scalar signal value.  The ensemble
 * output is a weighted linear combination:
 *
 *   y = Σ_i  w_i * s_i
 *
 * After each trade outcome the weights are updated via a single SGD step:
 *
 *   err  = y − target
 *   w_i -= lr * err * s_i
 *
 * Weights are projected back to the non-negative simplex after each update
 * (sum-to-one normalization with non-negativity) so the output remains a
 * convex combination ∈ [min(s), max(s)].
 *
 * ## Intended use
 * Combine heterogeneous upstream detectors (regime, entropy, vol, decay bias, etc.)
 * into a single output that maximises directional accuracy over time without
 * requiring manual weight tuning.
 *
 * @code
 * SignalEnsembleLayer ensemble;
 * int regime_id  = ensemble.register_source("regime");
 * int entropy_id = ensemble.register_source("entropy");
 * int decay_id   = ensemble.register_source("decay_bias");
 *
 * // Hot loop: update each source every token.
 * ensemble.update_source(regime_id,  regime_score);
 * ensemble.update_source(entropy_id, 1.0 - entropy);  // low entropy = conviction
 * ensemble.update_source(decay_id,   decayed_bias);
 *
 * double signal = ensemble.ensemble_output();
 *
 * // After trade resolves:
 * ensemble.record_outcome(profit > 0 ? 1.0 : -1.0);
 * @endcode
 *
 * ## Feature flag
 * Controlled by `LLMQUANT_ENABLE_SIGNAL_ENSEMBLE` (default ON).
 */
class SignalEnsembleLayer {
public:
    /**
     * @brief Configuration.
     */
    struct Config {
        /**
         * @brief SGD learning rate for weight updates.
         *
         * Smaller values → slower adaptation, more stable weights.
         * Default 0.01.
         */
        double learning_rate{0.01};

        /**
         * @brief Initial weight for each newly registered source.
         *
         * After registration all weights are renormalised.
         * Default 1.0 (uniform prior → each source starts with equal weight).
         */
        double initial_weight{1.0};

        /**
         * @brief Minimum weight any source can hold after projection.
         *
         * Prevents sources from being completely zeroed out.
         * Default 0.0 (allow complete suppression).
         */
        double min_weight{0.0};

        /**
         * @brief Optional callback fired after each weight update.
         *
         * Parameters: const reference to the current weight vector (indexed
         * by source registration order).
         * Useful for monitoring ensemble drift.
         */
        std::function<void(const std::vector<double>&)> on_weight_update;
    };

    // -----------------------------------------------------------------------
    // Construction
    // -----------------------------------------------------------------------

    explicit SignalEnsembleLayer(Config config = Config{});

    // -----------------------------------------------------------------------
    // Source management
    // -----------------------------------------------------------------------

    /**
     * @brief Register a named signal source.
     *
     * Returns a stable integer ID used by update_source().  IDs are
     * contiguous starting from 0.  Thread-safe.
     *
     * @param name  Human-readable label (used in to_stats_json()).
     * @return      Source ID.
     */
    int register_source(const std::string& name);

    /**
     * @brief Update the current value of a source.
     *
     * Silently ignored if the ID is out of range.  Thread-safe.
     *
     * @param source_id  ID returned by register_source().
     * @param value      Current signal value (any finite double).
     */
    void update_source(int source_id, double value);

    // -----------------------------------------------------------------------
    // Inference
    // -----------------------------------------------------------------------

    /**
     * @brief Compute and return the current weighted ensemble output.
     *
     * Returns 0 if no sources have been registered.  Thread-safe.
     */
    [[nodiscard]] double ensemble_output() const;

    // -----------------------------------------------------------------------
    // Learning
    // -----------------------------------------------------------------------

    /**
     * @brief Record the trade outcome and update weights via SGD.
     *
     * @param target  The desired output for the most recent signal.
     *                Typically +1.0 for a win, -1.0 for a loss, or
     *                the actual P&L normalised to [-1, 1].
     *
     * Thread-safe.
     */
    void record_outcome(double target);

    // -----------------------------------------------------------------------
    // Inspection
    // -----------------------------------------------------------------------

    /**
     * @brief Number of registered sources.
     *
     * Thread-safe.
     */
    [[nodiscard]] int source_count() const noexcept;

    /**
     * @brief Current weight for a source by ID.
     *
     * Returns 0 if the ID is out of range.  Thread-safe.
     */
    [[nodiscard]] double weight(int source_id) const noexcept;

    /**
     * @brief Copy of the full weight vector (indexed by source ID).
     *
     * Thread-safe.
     */
    [[nodiscard]] std::vector<double> weights() const;

    // -----------------------------------------------------------------------
    // Statistics
    // -----------------------------------------------------------------------

    /** @brief Total outcomes recorded. */
    [[nodiscard]] uint64_t total_outcomes() const noexcept {
        return total_outcomes_.load(std::memory_order_relaxed);
    }

    /**
     * @brief Reset weights to uniform distribution and clear statistics.
     *
     * Thread-safe.
     */
    void reset();

    /**
     * @brief Update configuration.  Does NOT reset weights.
     *
     * Thread-safe.
     */
    void update_config(const Config& config);

    /**
     * @brief Serialise current state (weights, source names, stats) to JSON.
     */
    [[nodiscard]] std::string to_stats_json() const;

private:
    mutable std::mutex mutex_;
    Config config_;

    std::vector<std::string> names_;
    std::vector<double>      weights_;
    std::vector<double>      values_;

    std::atomic<uint64_t> total_outcomes_{0};

    // Renormalise weights onto the non-negative simplex.  Called under lock.
    void project_weights();
};

} // namespace llmquant
