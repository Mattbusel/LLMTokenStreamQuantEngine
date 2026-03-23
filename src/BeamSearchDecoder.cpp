#include "BeamSearchDecoder.h"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <limits>
#include <numeric>
#include <stdexcept>

namespace llmquant {

// ---------------------------------------------------------------------------
// Construction
// ---------------------------------------------------------------------------

BeamSearchDecoder::BeamSearchDecoder(const BeamSearchConfig& config)
    : config_(config)
{
    if (config_.beam_width <= 0)
        throw std::invalid_argument("BeamSearchDecoder: beam_width must be > 0");
    if (config_.max_length <= 0)
        throw std::invalid_argument("BeamSearchDecoder: max_length must be > 0");
}

// ---------------------------------------------------------------------------
// Static helpers
// ---------------------------------------------------------------------------

float BeamSearchDecoder::apply_length_penalty(float log_prob, int length, float alpha)
{
    if (length <= 0) return log_prob;
    return log_prob / std::pow(static_cast<float>(length), alpha);
}

std::vector<float> BeamSearchDecoder::log_softmax(const std::vector<float>& logits)
{
    if (logits.empty()) return {};

    // Numeric stability: subtract max before exp
    float max_logit = *std::max_element(logits.begin(), logits.end());
    std::vector<float> exps(logits.size());
    float sum_exp = 0.0f;
    for (std::size_t i = 0; i < logits.size(); ++i) {
        exps[i] = std::exp(logits[i] - max_logit);
        sum_exp += exps[i];
    }
    float log_sum = std::log(sum_exp);
    std::vector<float> log_probs(logits.size());
    for (std::size_t i = 0; i < logits.size(); ++i) {
        log_probs[i] = (logits[i] - max_logit) - log_sum;
    }
    return log_probs;
}

void BeamSearchDecoder::apply_repetition_penalty(
    std::vector<float>& logits,
    const std::vector<int>& token_ids) const
{
    for (int tok : token_ids) {
        if (tok < 0 || static_cast<std::size_t>(tok) >= logits.size()) continue;
        if (logits[tok] > 0.0f) {
            logits[tok] /= config_.repetition_penalty;
        } else {
            logits[tok] *= config_.repetition_penalty;
        }
    }
}

// ---------------------------------------------------------------------------
// Decode
// ---------------------------------------------------------------------------

std::vector<BeamHypothesis> BeamSearchDecoder::decode(
    std::vector<float> initial_logits,
    std::function<std::vector<float>(const std::vector<int>&)> next_token_fn,
    int vocab_size)
{
    if (vocab_size <= 0)
        throw std::invalid_argument("BeamSearchDecoder: vocab_size must be > 0");
    if (!next_token_fn)
        throw std::invalid_argument("BeamSearchDecoder: next_token_fn must not be null");
    if (static_cast<int>(initial_logits.size()) < vocab_size)
        initial_logits.resize(static_cast<std::size_t>(vocab_size), 0.0f);

    // Step 1: initialize beams from top-k of initial logits
    std::vector<float> init_log_probs = log_softmax(initial_logits);

    // Get top-beam_width token indices
    std::vector<int> indices(static_cast<std::size_t>(vocab_size));
    std::iota(indices.begin(), indices.end(), 0);
    int top_k = std::min(config_.beam_width, vocab_size);
    std::partial_sort(indices.begin(), indices.begin() + top_k, indices.end(),
        [&init_log_probs](int a, int b) {
            return init_log_probs[a] > init_log_probs[b];
        });

    std::vector<BeamHypothesis> active_beams;
    std::vector<BeamHypothesis> completed_beams;

    for (int i = 0; i < top_k; ++i) {
        int tok = indices[static_cast<std::size_t>(i)];
        BeamHypothesis hyp;
        hyp.token_ids = {tok};
        hyp.log_prob = init_log_probs[static_cast<std::size_t>(tok)];
        hyp.is_done = (tok == config_.eos_token_id);
        if (hyp.is_done) {
            completed_beams.push_back(hyp);
        } else {
            active_beams.push_back(hyp);
        }
    }

    // Step 2: expand beams until all are done or max_length reached
    while (!active_beams.empty()) {
        // Collect all candidate extensions: (score, BeamHypothesis)
        std::vector<std::pair<float, BeamHypothesis>> candidates;

        for (const auto& beam : active_beams) {
            if (static_cast<int>(beam.token_ids.size()) >= config_.max_length) {
                BeamHypothesis done = beam;
                done.is_done = true;
                completed_beams.push_back(done);
                continue;
            }

            // Get next token logits
            std::vector<float> logits = next_token_fn(beam.token_ids);
            if (static_cast<int>(logits.size()) < vocab_size)
                logits.resize(static_cast<std::size_t>(vocab_size), 0.0f);

            // Apply repetition penalty
            apply_repetition_penalty(logits, beam.token_ids);

            // Convert to log-probs
            std::vector<float> log_probs = log_softmax(logits);

            // Keep top-beam_width extensions
            std::vector<int> ext_indices(static_cast<std::size_t>(vocab_size));
            std::iota(ext_indices.begin(), ext_indices.end(), 0);
            int k = std::min(config_.beam_width, vocab_size);
            std::partial_sort(ext_indices.begin(), ext_indices.begin() + k,
                ext_indices.end(),
                [&log_probs](int a, int b) {
                    return log_probs[a] > log_probs[b];
                });

            for (int j = 0; j < k; ++j) {
                int next_tok = ext_indices[static_cast<std::size_t>(j)];
                BeamHypothesis new_hyp;
                new_hyp.token_ids = beam.token_ids;
                new_hyp.token_ids.push_back(next_tok);
                new_hyp.log_prob = beam.log_prob + log_probs[static_cast<std::size_t>(next_tok)];
                new_hyp.is_done = (next_tok == config_.eos_token_id)
                    || (static_cast<int>(new_hyp.token_ids.size()) >= config_.max_length);

                float score = apply_length_penalty(
                    new_hyp.log_prob,
                    static_cast<int>(new_hyp.token_ids.size()),
                    config_.length_penalty);

                candidates.emplace_back(score, new_hyp);
            }
        }

        // Select top-beam_width candidates for next active set
        std::sort(candidates.begin(), candidates.end(),
            [](const auto& a, const auto& b) { return a.first > b.first; });

        active_beams.clear();
        int kept = 0;
        for (auto& [score, hyp] : candidates) {
            if (kept >= config_.beam_width) break;
            if (hyp.is_done) {
                completed_beams.push_back(hyp);
            } else {
                active_beams.push_back(hyp);
            }
            ++kept;
        }

        // Safety: if we got no active beams but kept was 0, break
        if (kept == 0) break;
    }

    // Add any remaining active beams as completed
    for (auto& beam : active_beams) {
        beam.is_done = true;
        completed_beams.push_back(beam);
    }

    // Sort completed beams by length-penalized score descending
    std::sort(completed_beams.begin(), completed_beams.end(),
        [this](const BeamHypothesis& a, const BeamHypothesis& b) {
            float score_a = apply_length_penalty(
                a.log_prob, static_cast<int>(a.token_ids.size()), config_.length_penalty);
            float score_b = apply_length_penalty(
                b.log_prob, static_cast<int>(b.token_ids.size()), config_.length_penalty);
            return score_a > score_b;
        });

    return completed_beams;
}

}  // namespace llmquant
