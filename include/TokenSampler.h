#pragma once

#include <cstdint>
#include <vector>

namespace llmquant {

/// Configuration for token sampling strategies.
struct SamplingConfig {
    float temperature{1.0f};
    int top_k{50};
    float top_p{0.9f};
    float repetition_penalty{1.2f};
    uint64_t seed{12345ULL};
};

/**
 * @brief Token sampling strategies for autoregressive decoding.
 *
 * Supports temperature scaling, top-k filtering, nucleus (top-p) sampling,
 * repetition penalty, and multinomial sampling via an LCG PRNG.
 *
 * LCG: state = state * 6364136223846793005 + 1442695040888963407
 */
class TokenSampler {
public:
    explicit TokenSampler(SamplingConfig config);

    /// Divide logits by temperature (softens / sharpens the distribution).
    std::vector<float> apply_temperature(const std::vector<float>& logits) const;

    /// Zero out all but the top-k probabilities, then renormalise.
    std::vector<float> apply_top_k(const std::vector<float>& probs, int k) const;

    /// Nucleus sampling: keep the smallest set of tokens whose cumulative
    /// probability exceeds p, then renormalise.
    std::vector<float> apply_top_p(const std::vector<float>& probs, float p) const;

    /// Divide logits of tokens that appear in recent_tokens by repetition_penalty.
    std::vector<float> apply_repetition_penalty(
        const std::vector<float>& logits,
        const std::vector<int>& recent_tokens) const;

    /// Multinomial sample from a probability vector using the internal LCG.
    int sample(const std::vector<float>& probs);

    /// Greedy decoding: return the argmax of logits.
    int greedy(const std::vector<float>& logits) const;

    /**
     * @brief Full sampling pipeline.
     *
     * 1. Apply repetition penalty to logits.
     * 2. Apply temperature.
     * 3. Compute softmax to get probabilities.
     * 4. Apply top-k.
     * 5. Apply top-p.
     * 6. Multinomial sample.
     */
    int sample_from_logits(
        const std::vector<float>& logits,
        const std::vector<int>& recent_tokens);

    /// Re-seed the internal LCG.
    void set_seed(uint64_t seed);

private:
    SamplingConfig config_;
    uint64_t lcg_state_;

    /// Advance the LCG and return a uniform double in [0, 1).
    double lcg_next();

    /// Compute softmax probabilities from logits.
    static std::vector<float> softmax(const std::vector<float>& logits);
};

}  // namespace llmquant
