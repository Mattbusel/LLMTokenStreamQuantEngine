#pragma once
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

/// Result of applying the filter to a single token.
struct FilterResult {
    bool filtered;             ///< True if the token was filtered out
    int  replacement_token_id; ///< -1 = drop, >= 0 = replace with this token
    std::string reason;        ///< Human-readable reason for filtering
};

/// Logit bias entry: token_id -> additive bias applied before sampling.
struct BiasEntry {
    int   token_id;
    float bias_logit;
};

/// Token-level content filter with hard bans, soft penalties, logit biases,
/// and repetition control. Has no external dependencies.
class TokenFilter {
public:
    TokenFilter();

    // -----------------------------------------------------------------------
    // Configuration
    // -----------------------------------------------------------------------

    /// Permanently ban a token: it is always dropped.
    void add_banned_token(int token_id);

    /// Soft-ban: apply a negative logit penalty to this token.
    void add_soft_banned_token(int token_id, float penalty);

    /// Add a logit bias (positive = boost, negative = suppress).
    void add_bias(int token_id, float bias);

    /// Set the maximum number of identical consecutive tokens allowed.
    void set_max_repetition(int max_same_consecutive);

    // -----------------------------------------------------------------------
    // Runtime filtering
    // -----------------------------------------------------------------------

    /// Filter a single token given the recent generation history.
    /// Returns FilterResult describing the outcome.
    FilterResult filter(int token_id, const std::vector<int>& history);

    /// Apply all registered biases to a full logits vector in-place.
    void apply_biases(std::vector<float>& logits) const;

    /// Apply repetition penalty to logits based on token history.
    /// Tokens that already appeared have their logit divided by `penalty`.
    void apply_repetition_penalty(std::vector<float>& logits,
                                  const std::vector<int>& history,
                                  float penalty) const;

    /// Return true if token_id is hard-banned.
    bool is_banned(int token_id) const;

    /// Clear any state that depends on the current generation history.
    void clear_history_dependent_state();

private:
    std::unordered_set<int>           banned_tokens_;
    std::unordered_map<int, float>    soft_banned_;   ///< token_id -> penalty (< 0)
    std::unordered_map<int, float>    biases_;         ///< token_id -> additive bias
    int                               max_repetition_; ///< 0 = unlimited
    int                               consecutive_count_;
    int                               last_token_;

    int count_consecutive(int token_id, const std::vector<int>& history) const;
};
