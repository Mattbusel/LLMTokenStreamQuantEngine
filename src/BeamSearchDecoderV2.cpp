#include "BeamSearchDecoderV2.h"

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

BeamSearchDecoderV2::BeamSearchDecoderV2(
    int beam_width,
    int max_length,
    int eos_token_id,
    float length_penalty)
    : beam_width_(beam_width)
    , max_length_(max_length)
    , eos_token_id_(eos_token_id)
    , length_penalty_(length_penalty)
{
    if (beam_width_ <= 0)
        throw std::invalid_argument("BeamSearchDecoderV2: beam_width must be > 0");
    if (max_length_ <= 0)
        throw std::invalid_argument("BeamSearchDecoderV2: max_length must be > 0");
}

// ---------------------------------------------------------------------------
// Softmax helpers
// ---------------------------------------------------------------------------

std::vector<float> BeamSearchDecoderV2::softmax(const std::vector<float>& logits) const
{
    if (logits.empty()) return {};
    float max_val = *std::max_element(logits.begin(), logits.end());
    std::vector<float> probs(logits.size());
    float sum = 0.0f;
    for (std::size_t i = 0; i < logits.size(); ++i) {
        probs[i] = std::exp(logits[i] - max_val);
        sum += probs[i];
    }
    if (sum > 0.0f)
        for (auto& p : probs) p /= sum;
    return probs;
}

std::vector<float> BeamSearchDecoderV2::log_softmax(const std::vector<float>& logits) const
{
    if (logits.empty()) return {};
    float max_val = *std::max_element(logits.begin(), logits.end());
    float sum = 0.0f;
    for (float v : logits) sum += std::exp(v - max_val);
    float log_norm = max_val + std::log(sum);
    std::vector<float> out(logits.size());
    for (std::size_t i = 0; i < logits.size(); ++i)
        out[i] = logits[i] - log_norm;
    return out;
}

float BeamSearchDecoderV2::log_sum_exp(float a, float b)
{
    float max_ab = std::max(a, b);
    return max_ab + std::log(std::exp(a - max_ab) + std::exp(b - max_ab));
}

// ---------------------------------------------------------------------------
// Length-penalised score
// ---------------------------------------------------------------------------

float BeamSearchDecoderV2::length_penalized_score(const Beam& beam) const
{
    int len = static_cast<int>(beam.tokens.size());
    if (len == 0) return 0.0f;
    float denom = std::pow(static_cast<float>(len), length_penalty_);
    return beam.log_prob / denom;
}

// ---------------------------------------------------------------------------
// Expand beams
// ---------------------------------------------------------------------------

void BeamSearchDecoderV2::expand_beams(
    std::vector<Beam>& beams,
    const std::vector<float>& logits)
{
    if (logits.empty()) return;
    int vocab = static_cast<int>(logits.size());

    auto log_probs = log_softmax(logits);

    // For each active beam, generate beam_width candidates
    std::vector<Beam> candidates;
    for (const auto& beam : beams) {
        if (beam.finished) {
            candidates.push_back(beam);
            continue;
        }

        // Find top-beam_width tokens
        std::vector<int> top_idx(vocab);
        std::iota(top_idx.begin(), top_idx.end(), 0);
        int k = std::min(beam_width_, vocab);
        std::partial_sort(top_idx.begin(), top_idx.begin() + k, top_idx.end(),
                          [&](int a, int b) { return log_probs[a] > log_probs[b]; });

        for (int i = 0; i < k; ++i) {
            int tok = top_idx[i];
            Beam new_beam;
            new_beam.tokens = beam.tokens;
            new_beam.tokens.push_back(tok);
            new_beam.log_prob = beam.log_prob + log_probs[tok];
            new_beam.finished =
                (tok == eos_token_id_) ||
                (static_cast<int>(new_beam.tokens.size()) >= max_length_);
            new_beam.score = length_penalized_score(new_beam);
            candidates.push_back(std::move(new_beam));
        }
    }

    // Replace beams with candidates
    beams = std::move(candidates);
}

// ---------------------------------------------------------------------------
// Prune beams
// ---------------------------------------------------------------------------

void BeamSearchDecoderV2::prune_beams(std::vector<Beam>& beams)
{
    if (static_cast<int>(beams.size()) <= beam_width_) return;
    std::partial_sort(beams.begin(), beams.begin() + beam_width_, beams.end(),
                      [](const Beam& a, const Beam& b) { return a.score > b.score; });
    beams.resize(beam_width_);
}

// ---------------------------------------------------------------------------
// Best sequence
// ---------------------------------------------------------------------------

std::vector<int> BeamSearchDecoderV2::best_sequence(const std::vector<Beam>& beams) const
{
    if (beams.empty()) return {};
    auto it = std::max_element(beams.begin(), beams.end(),
                               [](const Beam& a, const Beam& b) { return a.score < b.score; });
    return it->tokens;
}

// ---------------------------------------------------------------------------
// Full beam search
// ---------------------------------------------------------------------------

std::vector<Beam> BeamSearchDecoderV2::search(
    const std::vector<float>& initial_logits,
    std::function<std::vector<float>(const std::vector<int>&)> next_logits_fn)
{
    if (initial_logits.empty()) return {};

    // Seed: one empty beam
    std::vector<Beam> beams(1);
    beams[0].log_prob = 0.0f;
    beams[0].score = 0.0f;
    beams[0].finished = false;

    // First step using initial_logits
    expand_beams(beams, initial_logits);
    prune_beams(beams);

    // Subsequent steps
    for (int step = 1; step < max_length_; ++step) {
        bool all_done = true;
        for (const auto& b : beams)
            if (!b.finished) { all_done = false; break; }
        if (all_done) break;

        // Expand each unfinished beam
        std::vector<Beam> next_beams;
        for (auto& beam : beams) {
            if (beam.finished) {
                next_beams.push_back(beam);
                continue;
            }
            auto logits = next_logits_fn(beam.tokens);
            // Temporarily put this beam in a one-element vector for expand
            std::vector<Beam> single = {beam};
            expand_beams(single, logits);
            for (auto& b : single)
                next_beams.push_back(std::move(b));
        }

        beams = std::move(next_beams);
        prune_beams(beams);
    }

    // Update scores (final)
    for (auto& b : beams)
        b.score = length_penalized_score(b);

    std::sort(beams.begin(), beams.end(),
              [](const Beam& a, const Beam& b) { return a.score > b.score; });
    return beams;
}

}  // namespace llmquant
