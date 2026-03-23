#pragma once

#include <cmath>
#include <functional>
#include <string>
#include <vector>

namespace llmquant {

/// A single hypothesis maintained during beam search.
struct BeamHypothesis {
    /// Sequence of token IDs generated so far.
    std::vector<int> token_ids;
    /// Cumulative log-probability of this sequence.
    float log_prob{0.0f};
    /// True when the sequence has hit EOS or max_length.
    bool is_done{false};
};

/// Configuration for beam search decoding.
struct BeamSearchConfig {
    /// Number of hypotheses to maintain in parallel.
    int beam_width{4};
    /// Maximum sequence length (including the prompt tokens).
    int max_length{128};
    /// Length penalty exponent: score = log_prob / length^length_penalty.
    float length_penalty{1.0f};
    /// Repetition penalty: logit for repeated tokens is divided by this value.
    float repetition_penalty{1.2f};
    /// End-of-sequence token ID.
    int eos_token_id{2};
};

/**
 * @brief Beam search decoder for autoregressive token generation.
 *
 * Maintains `beam_width` active hypotheses, expands each at every step by
 * querying a `next_token_fn`, applies repetition penalty, accumulates
 * log-probabilities, and applies a length penalty to final scores.
 *
 * ## Usage
 * @code
 * BeamSearchConfig cfg;
 * cfg.beam_width = 4;
 * cfg.max_length = 64;
 *
 * BeamSearchDecoder decoder(cfg);
 * auto hypotheses = decoder.decode(
 *     initial_logits, next_token_fn, vocab_size);
 * @endcode
 */
class BeamSearchDecoder {
public:
    explicit BeamSearchDecoder(const BeamSearchConfig& config);

    /**
     * @brief Run beam search and return completed hypotheses sorted by score.
     *
     * @param initial_logits  Logits over the vocabulary for the first step.
     * @param next_token_fn   Function that takes the current token sequence and
     *                        returns logits over the vocabulary for the next step.
     * @param vocab_size      Total vocabulary size.
     * @return Completed hypotheses sorted by length-penalized score descending.
     */
    std::vector<BeamHypothesis> decode(
        std::vector<float> initial_logits,
        std::function<std::vector<float>(const std::vector<int>&)> next_token_fn,
        int vocab_size);

    /**
     * @brief Apply length penalty to a raw log-probability.
     *
     * score = log_prob / pow(length, alpha)
     *
     * @param log_prob Raw cumulative log-probability.
     * @param length   Sequence length.
     * @param alpha    Length penalty exponent.
     */
    static float apply_length_penalty(float log_prob, int length, float alpha);

private:
    BeamSearchConfig config_;

    /// Apply log-softmax to convert logits to log-probabilities.
    static std::vector<float> log_softmax(const std::vector<float>& logits);

    /// Apply repetition penalty: divide logits of already-seen tokens.
    void apply_repetition_penalty(
        std::vector<float>& logits,
        const std::vector<int>& token_ids) const;
};

}  // namespace llmquant
