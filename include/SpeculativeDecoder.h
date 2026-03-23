#pragma once

#include <cstdint>
#include <functional>
#include <string>
#include <vector>

namespace llmquant {

/// Configuration for speculative decoding.
struct SpeculativeConfig {
    /// Number of draft tokens to generate per speculative step.
    int draft_steps{4};
    /// Minimum acceptance probability threshold (tokens below this are always rejected).
    float acceptance_threshold{0.0f};
};

/// Accumulated statistics for a SpeculativeDecoder session.
struct SpeculativeStats {
    uint64_t total_draft_tokens{0};
    uint64_t accepted_tokens{0};
    float acceptance_rate{0.0f};
};

/**
 * @brief Speculative decoding accelerator.
 *
 * Uses a fast draft model to propose `draft_steps` tokens at a time, then
 * validates them against a slower but more accurate target model.  Accepted
 * draft tokens are kept; on the first rejection a correction token is sampled
 * from an adjusted target distribution.  This allows the target model to run
 * in a batch over the draft positions rather than autoregressively, reducing
 * the average number of target model forward passes per output token.
 *
 * Both models are supplied as `std::function` objects that map a token context
 * to logits over the vocabulary.
 *
 * ## Acceptance criterion
 * For each draft token t at position k:
 *   acceptance_prob = min(1, p_target[t] / p_draft[t])
 * Token t is accepted with that probability (sampled via LCG).
 * On rejection, a correction token is drawn from:
 *   p_corrected[i] = max(0, p_target[i] - p_draft[i]) (renormalized)
 *
 * ## Usage
 * @code
 * SpeculativeConfig cfg;
 * cfg.draft_steps = 4;
 *
 * SpeculativeDecoder dec(draft_fn, target_fn, cfg);
 * uint64_t rng = 12345;
 * auto tokens = dec.decode_step(context, rng);
 * auto stats = dec.get_stats();
 * @endcode
 */
class SpeculativeDecoder {
public:
    using ModelFn = std::function<std::vector<float>(const std::vector<int>&)>;

    /**
     * @brief Construct a SpeculativeDecoder.
     *
     * @param draft_model  Fast draft model: context → logits.
     * @param target_model Accurate target model: context → logits.
     * @param config       Speculative decoding configuration.
     */
    SpeculativeDecoder(ModelFn draft_model, ModelFn target_model,
                       const SpeculativeConfig& config = {});

    /**
     * @brief Execute one speculative decoding step.
     *
     * Generates up to `draft_steps` candidate tokens from the draft model,
     * validates them against the target model, and returns the accepted tokens
     * plus one correction token.
     *
     * @param context    Current token context (prefix).
     * @param rng_state  LCG random state (updated in-place).
     * @return           Accepted draft tokens followed by the correction token.
     */
    std::vector<int> decode_step(const std::vector<int>& context,
                                 uint64_t& rng_state);

    /// Return accumulated acceptance statistics.
    SpeculativeStats get_stats() const;

private:
    ModelFn draft_model_;
    ModelFn target_model_;
    SpeculativeConfig config_;
    SpeculativeStats stats_;

    /// Convert raw logits to a probability distribution (softmax).
    static std::vector<float> softmax(const std::vector<float>& logits);

    /// Sample a token index from a probability distribution using the LCG state.
    static int sample_from_probs(const std::vector<float>& probs, uint64_t& rng_state);

    /// Advance the LCG and return a float in [0, 1).
    static float lcg_uniform(uint64_t& state);
};

}  // namespace llmquant
