#pragma once

#include <string>
#include <vector>
#include <unordered_map>
#include <atomic>
#include <stdexcept>

namespace llmquant {

/// Normalised semantic weight extracted from a single token or token sequence.
///
/// All fields are in the range [-1.0, 1.0] except confidence_score which is
/// in [0.0, 1.0].  A fully neutral token has all fields at 0.0 except
/// confidence_score which defaults to 0.5.
struct SemanticWeight {
    /// Overall sentiment polarity: negative = bearish/fearful, positive = bullish.
    double sentiment_score{0.0};
    /// How strongly the model believes this mapping is accurate (0 = none, 1 = certain).
    double confidence_score{0.5};
    /// Implied market volatility contribution (0 = calm, 1 = high volatility).
    double volatility_score{0.0};
    /// Directional market bias: negative = sell pressure, positive = buy pressure.
    double directional_bias{0.0};
};

/// Maps raw LLM tokens to their quantitative SemanticWeight representations.
///
/// A default dictionary of ~15 high-signal tokens is loaded at construction.
/// Additional mappings can be injected at runtime via add_token_mapping() or
/// loaded in bulk from a tab-separated dictionary file.
///
/// Thread safety: map_token_to_weight() and map_sequence_to_weight() are
/// safe to call from multiple threads concurrently (atomic stat counters,
/// read-only map access after initialisation).  Mutation methods
/// (add_token_mapping, load_sentiment_dictionary) must not be called
/// concurrently with read methods.
class LLMAdapter {
public:
    /// Construct an adapter pre-loaded with the built-in default token dictionary.
    LLMAdapter();

    /// Look up the SemanticWeight for a single token.
    ///
    /// # Arguments
    /// * `token` — Raw token string (case-sensitive).
    ///
    /// # Returns
    /// The registered SemanticWeight, or a neutral weight
    /// `{0.0, 0.5, 0.1, 0.0}` if the token is not in the dictionary.
    SemanticWeight map_token_to_weight(const std::string& token) const;

    /// Compute a confidence-weighted aggregate SemanticWeight for a token sequence.
    ///
    /// Each token is looked up individually; the results are averaged with each
    /// token's confidence_score used as its weight.
    ///
    /// # Arguments
    /// * `tokens` — Ordered list of raw token strings.
    ///
    /// # Returns
    /// Aggregated SemanticWeight, or a zero weight if `tokens` is empty.
    SemanticWeight map_sequence_to_weight(const std::vector<std::string>& tokens) const;

    /// Load additional token-to-weight mappings from a whitespace-delimited file.
    ///
    /// Each line must contain: `<token> <sentiment> <confidence> <volatility> <bias>`
    ///
    /// # Arguments
    /// * `filepath` — Path to the dictionary file.
    ///
    /// # Throws
    /// `std::runtime_error` if the file cannot be opened.
    void load_sentiment_dictionary(const std::string& filepath);

    /// Insert or overwrite a single token mapping.
    ///
    /// # Arguments
    /// * `token`  — Raw token string (case-sensitive).
    /// * `weight` — SemanticWeight to associate with the token.
    void add_token_mapping(const std::string& token, const SemanticWeight& weight);

private:
    void initialize_default_mappings();

    std::unordered_map<std::string, SemanticWeight> token_weights_;

    /// Internal statistics; mutable so const query methods can update them.
    mutable struct {
        std::atomic<uint64_t> tokens_processed{0};
        std::atomic<uint64_t> cache_hits{0};
        std::atomic<uint64_t> cache_misses{0};
    } stats_;
};

} // namespace llmquant
