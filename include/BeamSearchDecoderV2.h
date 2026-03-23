#pragma once

#include <cmath>
#include <functional>
#include <vector>

namespace llmquant {

/// A single beam hypothesis in beam search.
struct Beam {
    std::vector<int> tokens;
    float log_prob{0.0f};
    float score{0.0f};
    bool finished{false};
};

/**
 * @brief Beam search decoder (v2) with length penalty and softmax helpers.
 *
 * Maintains `beam_width` active beams, expands each step with top-k
 * candidates, prunes to beam_width, and applies a length penalty to the
 * final score.
 */
class BeamSearchDecoderV2 {
public:
    BeamSearchDecoderV2(
        int beam_width,
        int max_length,
        int eos_token_id,
        float length_penalty = 1.0f);

    /**
     * @brief Run beam search.
     *
     * @param initial_logits  Logits for the first decoding step.
     * @param next_logits_fn  Function taking the current token sequence and
     *                        returning logits for the next token.
     * @return All beams (finished and unfinished), sorted by score descending.
     */
    std::vector<Beam> search(
        const std::vector<float>& initial_logits,
        std::function<std::vector<float>(const std::vector<int>&)> next_logits_fn);

    /// Expand all active beams with top-beam_width candidates from logits.
    void expand_beams(std::vector<Beam>& beams, const std::vector<float>& logits);

    /// Keep only the top beam_width beams by score.
    void prune_beams(std::vector<Beam>& beams);

    /// Apply length penalty: score = log_prob / pow(length, length_penalty).
    float length_penalized_score(const Beam& beam) const;

    /// Return the token sequence of the highest-scored beam.
    std::vector<int> best_sequence(const std::vector<Beam>& beams) const;

    /// Compute softmax probabilities from logits (numerically stable).
    std::vector<float> softmax(const std::vector<float>& logits) const;

private:
    int beam_width_;
    int max_length_;
    int eos_token_id_;
    float length_penalty_;

    /// Log-sum-exp for numerical stability.
    static float log_sum_exp(float a, float b);

    /// Return the log-softmax of logits.
    std::vector<float> log_softmax(const std::vector<float>& logits) const;
};

}  // namespace llmquant
