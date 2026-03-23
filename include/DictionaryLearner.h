#pragma once

/**
 * @file DictionaryLearner.h
 * @brief Token dictionary self-learning via Naive Bayes from labelled trade outcomes.
 *
 * DictionaryLearner observes trade outcomes (profitable / not profitable) for
 * sets of tokens that were present at the time of signal generation and
 * computes a posterior weight for each token using a Laplace-smoothed
 * Bernoulli Naive Bayes estimator:
 *
 *   log_likelihood_ratio = log( P(token | profit) / P(token | loss) )
 *
 * The learned log-likelihood ratio is mapped to a weight in (0, 1] via a
 * sigmoid-like normalisation so it can directly replace the static bias weight
 * in the token dictionary.
 *
 * Hot-reload: get_updated_dictionary() returns the full weight map which can
 * be passed directly to DynamicTokenDictionary::reload() or
 * LLMAdapter's static map — no process restart required.
 *
 * ## Thread safety
 * All public methods acquire an internal mutex and are safe to call from
 * multiple threads.
 */

#include <cstddef>
#include <cstdint>
#include <mutex>
#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

namespace llmquant {

// ---------------------------------------------------------------------------
// Data types
// ---------------------------------------------------------------------------

/// A single labelled training observation.
struct LabelledToken {
    std::string token;
    bool        profitable{false};  ///< true  = the following trade was profitable.
    double      pnl{0.0};          ///< Actual P&L (used for weighted counting).
    int64_t     timestamp_ns{0};
};

/// Learned statistics for one token.
struct LearnedWeight {
    std::string token;
    double      weight{0.5};            ///< Current inferred weight in [0, 1].
    double      prior_weight{0.5};      ///< Original dictionary weight (from base dict load).
    size_t      positive_count{0};      ///< Observations where token co-occurred with profit.
    size_t      negative_count{0};      ///< Observations where token co-occurred with loss.
    double      log_likelihood_ratio{0.0}; ///< log( P(t|profit) / P(t|loss) ).
};

// ---------------------------------------------------------------------------
// DictionaryLearner
// ---------------------------------------------------------------------------

/**
 * @brief Bayesian token weight learner from labelled trade data.
 *
 * The learner does NOT require the base dictionary to be present at runtime —
 * if `base_dict_path` is empty or the file does not exist, all prior weights
 * default to 0.5 and learning proceeds from scratch.
 *
 * ### Smoothing
 * Laplace smoothing with `alpha = laplace_smoothing_` is applied to all
 * frequency counts to avoid zero-probability tokens dominating the posterior.
 *
 * ### Weight mapping
 * The log-likelihood ratio is mapped to [0, 1] via:
 *
 *   weight = sigmoid(log_likelihood_ratio * scale)
 *   where scale is chosen so LLR = 2.0  =>  weight ~ 0.88
 *                              LLR = -2.0 => weight ~ 0.12
 */
class DictionaryLearner {
public:
    /**
     * @param base_dict_path  Path to a JSON/YAML token dictionary to seed
     *                        prior weights from. Pass "" to use default 0.5
     *                        priors for all tokens.
     */
    explicit DictionaryLearner(const std::string& base_dict_path);

    // Non-copyable.
    DictionaryLearner(const DictionaryLearner&)            = delete;
    DictionaryLearner& operator=(const DictionaryLearner&) = delete;

    ~DictionaryLearner() = default;

    // -----------------------------------------------------------------------
    // Training
    // -----------------------------------------------------------------------

    /**
     * @brief Record a trade outcome for a set of tokens.
     *
     * Each token in `tokens` has its positive or negative count incremented.
     * Duplicate tokens within a single call are deduplicated (one observation
     * per token per trade).
     *
     * @param tokens      Tokens that were active when the trade signal fired.
     * @param profitable  true if the trade resulted in a profit.
     * @param pnl         Actual P&L (currently used for weighted count; a
     *                    positive PnL increments the positive accumulator by
     *                    |pnl| / normaliser rather than by 1).  Set to 1.0
     *                    for unweighted counting.
     */
    void record_outcome(const std::vector<std::string>& tokens,
                        bool profitable,
                        double pnl);

    // -----------------------------------------------------------------------
    // Inference / hot-reload
    // -----------------------------------------------------------------------

    /**
     * @brief Current learned weight for a token in [0, 1].
     *
     * If the token has never been observed, returns its prior weight (loaded
     * from the base dictionary, or 0.5 if unknown).
     */
    double get_weight(const std::string& token) const;

    /**
     * @brief Return the full token -> weight map suitable for hot-reloading
     *        into DynamicTokenDictionary or LLMAdapter.
     *
     * Only tokens with at least `min_observations` total observations are
     * included; all others are excluded so the base dictionary priors are
     * not overwritten by noisy estimates.
     *
     * @param min_observations  Minimum (positive_count + negative_count) for
     *                          inclusion. Default: 5.
     */
    std::unordered_map<std::string, double>
    get_updated_dictionary(size_t min_observations = 5) const;

    // -----------------------------------------------------------------------
    // Serialisation
    // -----------------------------------------------------------------------

    /**
     * @brief Export all learned weights to a compact JSON string.
     *
     * The format is intentionally a superset of the base dictionary JSON so
     * it can be written to disk and re-loaded as a new base dictionary.
     */
    std::string export_json() const;

    /**
     * @brief Import previously exported JSON, merging into current state.
     *
     * Existing counts are added to imported counts (no overwrite).  To do a
     * clean reload, call clear() first.
     *
     * @throws std::runtime_error if `json_str` is not valid JSON.
     */
    void import_json(const std::string& json_str);

    // -----------------------------------------------------------------------
    // Statistics
    // -----------------------------------------------------------------------

    size_t total_observations() const;  ///< Total (positive + negative) trade count.
    size_t vocabulary_size()    const;  ///< Number of distinct tokens observed.

    /**
     * @brief Top-N tokens by weight (highest conviction bullish tokens first).
     * @param n  Number of entries to return.
     */
    std::vector<LearnedWeight> top_weights(size_t n = 20) const;

    /**
     * @brief Bottom-N tokens by weight (lowest weight = strongest bearish signal).
     * @param n  Number of entries to return.
     */
    std::vector<LearnedWeight> bottom_weights(size_t n = 20) const;

    /**
     * @brief Return the LearnedWeight entry for a specific token, if known.
     */
    std::optional<LearnedWeight> lookup(const std::string& token) const;

    /// Reset all learned counts (prior weights from the base dictionary are kept).
    void clear();

    // -----------------------------------------------------------------------
    // Configuration
    // -----------------------------------------------------------------------

    /// Laplace smoothing parameter (default: 1.0).
    void set_laplace_smoothing(double alpha);

    double laplace_smoothing() const;

private:
    // -----------------------------------------------------------------------
    // Internal state
    // -----------------------------------------------------------------------

    /// Per-token learned statistics.
    std::unordered_map<std::string, LearnedWeight> weights_;

    /// Global trade-level counters (each call to record_outcome = one trade).
    size_t total_positive_{0};
    size_t total_negative_{0};

    double laplace_smoothing_{1.0};

    mutable std::mutex mutex_;

    // -----------------------------------------------------------------------
    // Internal helpers
    // -----------------------------------------------------------------------

    /**
     * @brief Recompute weight and log_likelihood_ratio for an entry.
     *
     * Must be called with mutex held after any count update.
     */
    void recompute_weight(LearnedWeight& lw) const;

    /**
     * @brief sigmoid(x) = 1 / (1 + exp(-x)).
     */
    static double sigmoid(double x) noexcept;

    /**
     * @brief Load prior weights from a JSON or YAML base dictionary file.
     *
     * Silently returns if the file does not exist or cannot be parsed.
     */
    void load_base_dict(const std::string& path);
};

} // namespace llmquant
