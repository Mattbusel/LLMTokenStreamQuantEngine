#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace llmquant {

/// A single candidate token with its raw logit and computed probability.
struct TokenCandidate {
    int   token_id;
    float logit;
    float probability;
};

/// Configuration for the token sampling dispatcher.
struct SamplingConfig {
    float temperature;       ///< Softmax temperature (> 0). 1.0 = no change.
    float top_p;             ///< Nucleus sampling threshold (0, 1].
    int   top_k;             ///< Top-k sampling (0 = disabled).
    float repetition_penalty;///< Penalty divisor for recently seen tokens (>= 1).
    bool  do_sample;         ///< true = stochastic; false = greedy argmax.
};

/**
 * @brief Collection of token sampling algorithms for autoregressive decoding.
 *
 * All methods are static and stateless; the caller manages the RNG state.
 *
 * ## LCG RNG
 *
 * A minimal 64-bit linear congruential generator is used for reproducible
 * stochastic sampling without external dependencies:
 *
 *   state = state * 6364136223846793005ULL + 1442695040888963407ULL
 *   float  = float(state >> 33) / float(1 << 31)
 *
 * ## Feature flag
 * Controlled by `LLMQUANT_ENABLE_SAMPLING_STRATEGIES` (default ON).
 */
class SamplingStrategies {
public:
    // ----------------------------------------------------------------
    // Numerics
    // ----------------------------------------------------------------

    /**
     * @brief Numerically stable softmax over logits with optional temperature.
     *
     * @param logits    Raw logits from the language model.
     * @param temperature Softmax temperature (> 0). Values < 1 sharpen the
     *                    distribution; values > 1 flatten it.
     * @return Probability vector that sums to 1.
     */
    static std::vector<float> softmax(const std::vector<float>& logits,
                                      float temperature = 1.0f);

    // ----------------------------------------------------------------
    // Deterministic sampling
    // ----------------------------------------------------------------

    /**
     * @brief Greedy decoding: return the argmax token id.
     *
     * @param logits Raw logits (unnormalised).
     * @return Index of the highest logit, or -1 if logits is empty.
     */
    static int greedy(const std::vector<float>& logits);

    // ----------------------------------------------------------------
    // Stochastic sampling
    // ----------------------------------------------------------------

    /**
     * @brief Sample from the top-k tokens by logit value.
     *
     * Restricts the candidate set to the k tokens with the highest logits,
     * then draws from their normalised probability distribution.
     *
     * @param logits    Raw logits.
     * @param k         Number of candidates to keep (clamped to vocab size).
     * @param rng_state Mutable LCG state updated in-place.
     * @return Sampled token id.
     */
    static int top_k_sample(const std::vector<float>& logits,
                             int k,
                             uint64_t& rng_state);

    /**
     * @brief Nucleus (top-p) sampling.
     *
     * Sorts tokens by probability descending, accumulates until cumulative
     * probability exceeds p, then samples from the nucleus.
     *
     * @param logits    Raw logits.
     * @param p         Nucleus threshold in (0, 1].
     * @param rng_state Mutable LCG state updated in-place.
     * @return Sampled token id.
     */
    static int top_p_sample(const std::vector<float>& logits,
                             float p,
                             uint64_t& rng_state);

    /**
     * @brief Unified sampling dispatcher.
     *
     * Applies repetition penalty, then dispatches to greedy / top-k / top-p
     * based on the provided config.
     *
     * @param logits        Raw logits.
     * @param config        Sampling configuration.
     * @param rng_state     Mutable LCG state.
     * @return Sampled token id.
     */
    static int sample(const std::vector<float>& logits,
                      const SamplingConfig& config,
                      uint64_t& rng_state);

    // ----------------------------------------------------------------
    // Repetition penalty
    // ----------------------------------------------------------------

    /**
     * @brief Apply repetition penalty in-place.
     *
     * For each token in `recent_tokens`, the corresponding logit is divided
     * by `penalty` (if positive) or multiplied by `penalty` (if negative).
     * This follows the Ctrl/HuggingFace convention:
     *   logit[t] /= penalty   (if logit[t] > 0)
     *   logit[t] *= penalty   (if logit[t] < 0)
     *
     * @param logits        Logit vector modified in-place.
     * @param recent_tokens Token ids seen recently in the context.
     * @param penalty       Penalty factor >= 1. 1.0 = no penalty.
     */
    static void apply_repetition_penalty(std::vector<float>& logits,
                                         const std::vector<int>& recent_tokens,
                                         float penalty);

    // ----------------------------------------------------------------
    // LCG random number generator
    // ----------------------------------------------------------------

    /**
     * @brief Advance the LCG state and return a float in [0, 1).
     *
     * Uses the 64-bit Knuth multiplicative hash with the Newlib additive
     * constant:
     *   state = state * 6364136223846793005ULL + 1442695040888963407ULL
     *   return float(state >> 33) / float(1 << 31)
     *
     * @param state LCG state, modified in-place.
     * @return Pseudo-random float in [0, 1).
     */
    static float lcg_float(uint64_t& state);
};

} // namespace llmquant
