#pragma once
#include <vector>
#include <cmath>
#include <algorithm>

/// A single beam search hypothesis.
struct Hypothesis {
    std::vector<int> token_ids;   ///< Generated token sequence
    double           log_prob;    ///< Accumulated log-probability
    double           length_penalty_alpha; ///< Length penalty exponent (Google NMT)

    /// Normalised score for ranking hypotheses.
    double normalized_score() const {
        if (token_ids.empty()) return log_prob;
        double lp = std::pow(static_cast<double>(token_ids.size()), length_penalty_alpha);
        return log_prob / lp;
    }
};

/// Tracks a pool of beam-search hypotheses, manages pruning, and provides
/// n-best extraction and diversity metrics.
class HypothesisTracker {
public:
    HypothesisTracker(int max_hypotheses, int eos_token_id, float length_penalty_alpha = 0.6f);

    /// Add a pre-built hypothesis to the pool.
    void add(Hypothesis hyp);

    /// Extend an existing hypothesis with a new token and its log-probability.
    /// Returns the new (extended) hypothesis without adding it to the pool.
    Hypothesis extend(int hypothesis_idx, int next_token_id, float log_prob);

    /// Keep only the top-n hypotheses by normalised score.
    void prune_to(int n);

    /// Return the hypothesis with the highest normalised score.
    const Hypothesis& best() const;

    /// Return the top-n hypotheses, sorted by normalised score descending.
    std::vector<Hypothesis> n_best(int n) const;

    /// True if every hypothesis in the pool ends with the EOS token.
    bool all_finished() const;

    /// Return only the hypotheses whose last token is EOS.
    std::vector<Hypothesis> finished_hypotheses() const;

    /// Return only hypotheses that have not yet produced EOS.
    std::vector<Hypothesis> active_hypotheses() const;

    /// Average pairwise token-level disagreement across all hypothesis pairs.
    float diversity_score() const;

    /// Number of hypotheses currently in the pool.
    int size() const;

private:
    int              max_hypotheses_;
    int              eos_token_id_;
    double           length_penalty_alpha_;
    std::vector<Hypothesis> hypotheses_;

    static bool score_greater(const Hypothesis& a, const Hypothesis& b) {
        return a.normalized_score() > b.normalized_score();
    }
};
