#include "SpeculativeDecoder.h"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <numeric>
#include <stdexcept>

namespace llmquant {

// ---------------------------------------------------------------------------
// Construction
// ---------------------------------------------------------------------------

SpeculativeDecoder::SpeculativeDecoder(
    ModelFn draft_model,
    ModelFn target_model,
    const SpeculativeConfig& config)
    : draft_model_(std::move(draft_model))
    , target_model_(std::move(target_model))
    , config_(config)
{
    if (!draft_model_)
        throw std::invalid_argument("SpeculativeDecoder: draft_model must not be null");
    if (!target_model_)
        throw std::invalid_argument("SpeculativeDecoder: target_model must not be null");
    if (config_.draft_steps <= 0)
        throw std::invalid_argument("SpeculativeDecoder: draft_steps must be > 0");
}

// ---------------------------------------------------------------------------
// Static helpers
// ---------------------------------------------------------------------------

std::vector<float> SpeculativeDecoder::softmax(const std::vector<float>& logits)
{
    if (logits.empty()) return {};
    float max_val = *std::max_element(logits.begin(), logits.end());
    std::vector<float> probs(logits.size());
    float sum = 0.0f;
    for (std::size_t i = 0; i < logits.size(); ++i) {
        probs[i] = std::exp(logits[i] - max_val);
        sum += probs[i];
    }
    if (sum > 0.0f) {
        for (auto& p : probs) p /= sum;
    }
    return probs;
}

float SpeculativeDecoder::lcg_uniform(uint64_t& state)
{
    // Knuth multiplicative LCG
    state = state * 6364136223846793005ULL + 1442695040888963407ULL;
    // Use upper 32 bits for better quality
    uint32_t upper = static_cast<uint32_t>(state >> 32);
    return static_cast<float>(upper) / static_cast<float>(UINT32_MAX + 1ULL);
}

int SpeculativeDecoder::sample_from_probs(
    const std::vector<float>& probs, uint64_t& rng_state)
{
    if (probs.empty()) return 0;
    float u = lcg_uniform(rng_state);
    float cumulative = 0.0f;
    for (std::size_t i = 0; i < probs.size(); ++i) {
        cumulative += probs[i];
        if (u <= cumulative) return static_cast<int>(i);
    }
    // Fallback: return last non-zero probability index
    for (int i = static_cast<int>(probs.size()) - 1; i >= 0; --i) {
        if (probs[static_cast<std::size_t>(i)] > 0.0f) return i;
    }
    return 0;
}

// ---------------------------------------------------------------------------
// decode_step
// ---------------------------------------------------------------------------

std::vector<int> SpeculativeDecoder::decode_step(
    const std::vector<int>& context,
    uint64_t& rng_state)
{
    // Phase 1: generate draft_steps tokens from draft model greedily
    std::vector<int> draft_tokens;
    draft_tokens.reserve(static_cast<std::size_t>(config_.draft_steps));

    std::vector<std::vector<float>> draft_probs_per_step;
    draft_probs_per_step.reserve(static_cast<std::size_t>(config_.draft_steps));

    std::vector<int> current_context = context;

    for (int step = 0; step < config_.draft_steps; ++step) {
        std::vector<float> logits = draft_model_(current_context);
        if (logits.empty()) break;

        std::vector<float> probs = softmax(logits);
        draft_probs_per_step.push_back(probs);

        // Greedy selection for the draft
        int draft_tok = static_cast<int>(
            std::max_element(probs.begin(), probs.end()) - probs.begin());
        draft_tokens.push_back(draft_tok);
        current_context.push_back(draft_tok);
    }

    if (draft_tokens.empty()) {
        // No draft tokens: just sample one from the target model
        std::vector<float> logits = target_model_(context);
        std::vector<float> probs = softmax(logits);
        int tok = sample_from_probs(probs, rng_state);
        ++stats_.total_draft_tokens;
        ++stats_.accepted_tokens;
        stats_.acceptance_rate = static_cast<float>(stats_.accepted_tokens)
            / static_cast<float>(stats_.total_draft_tokens);
        return {tok};
    }

    // Phase 2: validate draft tokens against target model
    std::vector<int> accepted;
    accepted.reserve(draft_tokens.size() + 1);

    int correction_token = -1;
    std::vector<int> validation_context = context;

    for (std::size_t k = 0; k < draft_tokens.size(); ++k) {
        int draft_tok = draft_tokens[k];
        ++stats_.total_draft_tokens;

        // Get target model distribution at this position
        std::vector<float> target_logits = target_model_(validation_context);
        std::vector<float> target_probs = softmax(target_logits);
        const std::vector<float>& draft_probs = draft_probs_per_step[k];

        // Compute acceptance probability: min(1, p_target[t] / p_draft[t])
        float p_draft_tok = (draft_tok < static_cast<int>(draft_probs.size()))
            ? draft_probs[static_cast<std::size_t>(draft_tok)]
            : 0.0f;
        float p_target_tok = (draft_tok < static_cast<int>(target_probs.size()))
            ? target_probs[static_cast<std::size_t>(draft_tok)]
            : 0.0f;

        float accept_prob = (p_draft_tok > 0.0f)
            ? std::min(1.0f, p_target_tok / p_draft_tok)
            : 0.0f;

        // Apply threshold floor
        if (accept_prob < config_.acceptance_threshold) accept_prob = 0.0f;

        float u = lcg_uniform(rng_state);
        if (u <= accept_prob) {
            // Accept this draft token
            accepted.push_back(draft_tok);
            ++stats_.accepted_tokens;
            validation_context.push_back(draft_tok);
        } else {
            // Reject: sample correction token from adjusted distribution
            // p_corrected[i] = max(0, p_target[i] - p_draft[i]), renormalized
            std::size_t vocab_size = std::max(target_probs.size(), draft_probs.size());
            std::vector<float> corrected(vocab_size, 0.0f);
            float sum_corrected = 0.0f;
            for (std::size_t i = 0; i < vocab_size; ++i) {
                float pt = (i < target_probs.size()) ? target_probs[i] : 0.0f;
                float pd = (i < draft_probs.size()) ? draft_probs[i] : 0.0f;
                corrected[i] = std::max(0.0f, pt - pd);
                sum_corrected += corrected[i];
            }
            // Renormalize
            if (sum_corrected > 0.0f) {
                for (auto& p : corrected) p /= sum_corrected;
            } else {
                // Fallback: use target distribution
                corrected = target_probs;
                if (corrected.size() < vocab_size)
                    corrected.resize(vocab_size, 0.0f);
            }
            correction_token = sample_from_probs(corrected, rng_state);
            break;
        }
    }

    // If all draft tokens were accepted, sample one more from target at the end
    if (correction_token < 0) {
        std::vector<float> target_logits = target_model_(validation_context);
        std::vector<float> target_probs = softmax(target_logits);
        correction_token = sample_from_probs(target_probs, rng_state);
    }

    accepted.push_back(correction_token);

    // Update acceptance rate
    if (stats_.total_draft_tokens > 0) {
        stats_.acceptance_rate = static_cast<float>(stats_.accepted_tokens)
            / static_cast<float>(stats_.total_draft_tokens);
    }

    return accepted;
}

// ---------------------------------------------------------------------------
// Statistics
// ---------------------------------------------------------------------------

SpeculativeStats SpeculativeDecoder::get_stats() const
{
    return stats_;
}

}  // namespace llmquant
