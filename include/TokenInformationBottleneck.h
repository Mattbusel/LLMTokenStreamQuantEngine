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
 * @brief Real-time Information Bottleneck analyser for the token stream.
 *
 * The Information Bottleneck (IB) framework asks: "How much of a token's
 * weight can we compress away while preserving the signal it carries?"
 *
 * This class approximates the IB trade-off online without requiring a
 * full probabilistic model:
 *
 * 1. **Relevance R(t)** — how strongly token t correlates with the running
 *    signal strength, estimated as the absolute Pearson correlation between
 *    the token's per-occurrence weight history and the per-occurrence bias
 *    shifts observed at the same positions.
 *
 * 2. **Complexity C(t)** — the Shannon entropy of the token's weight
 *    distribution (binned), normalised to [0,1].  High entropy = the token
 *    weight is unpredictable = high complexity.
 *
 * 3. **IB score** = R(t) / (C(t) + ε) — a relevance-per-complexity ratio.
 *    Tokens with high IB score are "efficient": they carry strong signal for
 *    little representational cost.  Tokens with low IB score are candidates
 *    for vocabulary pruning.
 *
 * ## Hot path
 * Only `record(token, weight, bias_shift)` is called on the signal path; it
 * is O(1) amortised.  Scoring (correlation + entropy) is computed on demand
 * via `ib_score(token)` which is O(W) in the per-token window size.
 *
 * ## Use cases
 * - Identify vocabulary noise in real time — tokens consistently scoring below
 *   `prune_threshold` are logged and can be suppressed in the LLMAdapter.
 * - Feed back into adaptive dictionary weighting.
 */
class TokenInformationBottleneck {
public:
    // -------------------------------------------------------------------------
    // Configuration
    // -------------------------------------------------------------------------
    struct Config {
        int    window_size      = 64;   ///< Per-token rolling window (weight + bias samples).
        int    n_weight_bins    = 8;    ///< Bins for weight entropy estimate.
        double prune_threshold  = 0.10; ///< IB score below which a token is flagged.
        double epsilon          = 1e-6; ///< Denominator guard for score computation.
        int    min_observations = 8;    ///< Min per-token samples before reporting score.
        int    top_k            = 10;   ///< Number of top/bottom tokens to track.

        /// Called when a token drops below prune_threshold for the first time.
        std::function<void(const std::string&, double)> on_low_score;
        /// Called when a previously flagged token recovers above prune_threshold.
        std::function<void(const std::string&, double)> on_recovered;
    };

    // -------------------------------------------------------------------------
    // Construction
    // -------------------------------------------------------------------------
    TokenInformationBottleneck();
    explicit TokenInformationBottleneck(const Config& cfg);

    // -------------------------------------------------------------------------
    // Core
    // -------------------------------------------------------------------------

    /**
     * @brief Record one token occurrence with its weight and the resulting bias shift.
     * @param token       Token string (typically the raw LLM token).
     * @param weight      Token weight as assigned by LLMAdapter (can be negative).
     * @param bias_shift  The delta_bias_shift of the signal this token contributed to.
     */
    void record(const std::string& token, double weight, double bias_shift);

    // -------------------------------------------------------------------------
    // Per-token accessors
    // -------------------------------------------------------------------------

    /// IB score for `token`; returns 0.0 if insufficient observations.
    double ib_score(const std::string& token) const noexcept;
    /// Relevance estimate [0,1] for `token`.
    double relevance(const std::string& token) const noexcept;
    /// Complexity estimate [0,1] (normalised entropy) for `token`.
    double complexity(const std::string& token) const noexcept;
    /// Observation count for `token`.
    int observation_count(const std::string& token) const noexcept;

    // -------------------------------------------------------------------------
    // Aggregate accessors
    // -------------------------------------------------------------------------

    /// Total token-occurrence records processed.
    uint64_t total_records() const noexcept;
    /// Number of distinct tokens seen.
    int distinct_tokens() const noexcept;
    /// Number of tokens currently flagged below prune_threshold.
    int flagged_count() const noexcept;
    /// Top-k tokens by IB score (sorted descending).
    std::vector<std::pair<std::string, double>> top_k_tokens() const;
    /// Bottom-k tokens by IB score (sorted ascending, min_observations required).
    std::vector<std::pair<std::string, double>> bottom_k_tokens() const;

    // -------------------------------------------------------------------------
    // Mutation
    // -------------------------------------------------------------------------

    /** Reset all per-token state. */
    void reset();
    /** Replace config and reset. */
    void update_config(const Config& cfg);

    // -------------------------------------------------------------------------
    // Diagnostics
    // -------------------------------------------------------------------------

    /** JSON with aggregate IB stats. */
    std::string to_stats_json() const;

private:
    struct TokenStats {
        std::vector<double> weights;     // rolling window
        std::vector<double> biases;      // rolling window (same index)
        int   head       = 0;
        int   count      = 0;
        bool  flagged    = false;
    };

    Config cfg_;
    mutable std::mutex mutex_;

    std::unordered_map<std::string, TokenStats> stats_;
    int flagged_count_ = 0;

    std::atomic<uint64_t> total_{0};

    // Helpers
    double compute_relevance(const TokenStats& ts) const noexcept;
    double compute_complexity(const TokenStats& ts) const noexcept;
    double compute_score(const TokenStats& ts) const noexcept;
};

}  // namespace llmquant
