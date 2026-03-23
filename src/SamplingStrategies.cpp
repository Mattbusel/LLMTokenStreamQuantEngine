#include "SamplingStrategies.h"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <numeric>
#include <stdexcept>

namespace llmquant {

// ─── LCG RNG ─────────────────────────────────────────────────────────────────

float SamplingStrategies::lcg_float(uint64_t& state) {
    state = state * 6364136223846793005ULL + 1442695040888963407ULL;
    // Take upper 31 bits (after discarding the bottom 33) for better quality.
    return static_cast<float>(state >> 33) / static_cast<float>(1 << 31);
}

// ─── SOFTMAX ─────────────────────────────────────────────────────────────────

std::vector<float> SamplingStrategies::softmax(const std::vector<float>& logits,
                                               float temperature) {
    if (logits.empty()) {
        return {};
    }

    // Guard against degenerate temperature.
    float temp = (temperature > 0.0f) ? temperature : 1.0f;

    // Scale logits by temperature.
    std::vector<float> scaled(logits.size());
    for (std::size_t i = 0; i < logits.size(); ++i) {
        scaled[i] = logits[i] / temp;
    }

    // Numerically stable: subtract max before exponentiating.
    float max_val = *std::max_element(scaled.begin(), scaled.end());
    std::vector<float> probs(scaled.size());
    float sum = 0.0f;
    for (std::size_t i = 0; i < scaled.size(); ++i) {
        probs[i] = std::exp(scaled[i] - max_val);
        sum += probs[i];
    }

    if (sum == 0.0f) sum = 1.0f; // guard
    for (auto& p : probs) {
        p /= sum;
    }
    return probs;
}

// ─── GREEDY ──────────────────────────────────────────────────────────────────

int SamplingStrategies::greedy(const std::vector<float>& logits) {
    if (logits.empty()) {
        return -1;
    }
    return static_cast<int>(
        std::max_element(logits.begin(), logits.end()) - logits.begin()
    );
}

// ─── REPETITION PENALTY ──────────────────────────────────────────────────────

void SamplingStrategies::apply_repetition_penalty(std::vector<float>& logits,
                                                  const std::vector<int>& recent_tokens,
                                                  float penalty) {
    if (penalty <= 1.0f || logits.empty() || recent_tokens.empty()) {
        return;
    }

    for (int tok : recent_tokens) {
        if (tok < 0 || static_cast<std::size_t>(tok) >= logits.size()) {
            continue;
        }
        float& l = logits[static_cast<std::size_t>(tok)];
        if (l >= 0.0f) {
            l /= penalty;
        } else {
            l *= penalty;
        }
    }
}

// ─── TOP-K SAMPLING ──────────────────────────────────────────────────────────

int SamplingStrategies::top_k_sample(const std::vector<float>& logits,
                                     int k,
                                     uint64_t& rng_state) {
    if (logits.empty()) {
        return -1;
    }

    // Build index list and sort by logit descending.
    std::vector<int> indices(logits.size());
    std::iota(indices.begin(), indices.end(), 0);

    int effective_k = std::min(k, static_cast<int>(logits.size()));
    if (effective_k <= 0) effective_k = static_cast<int>(logits.size());

    std::partial_sort(indices.begin(), indices.begin() + effective_k, indices.end(),
                      [&logits](int a, int b) { return logits[a] > logits[b]; });
    indices.resize(static_cast<std::size_t>(effective_k));

    // Softmax over the top-k.
    std::vector<float> top_logits(static_cast<std::size_t>(effective_k));
    for (int i = 0; i < effective_k; ++i) {
        top_logits[static_cast<std::size_t>(i)] = logits[static_cast<std::size_t>(indices[static_cast<std::size_t>(i)])];
    }
    std::vector<float> probs = softmax(top_logits, 1.0f);

    // Sample.
    float r = lcg_float(rng_state);
    float cumulative = 0.0f;
    for (int i = 0; i < effective_k; ++i) {
        cumulative += probs[static_cast<std::size_t>(i)];
        if (r <= cumulative) {
            return indices[static_cast<std::size_t>(i)];
        }
    }
    return indices[static_cast<std::size_t>(effective_k - 1)];
}

// ─── TOP-P (NUCLEUS) SAMPLING ────────────────────────────────────────────────

int SamplingStrategies::top_p_sample(const std::vector<float>& logits,
                                     float p,
                                     uint64_t& rng_state) {
    if (logits.empty()) {
        return -1;
    }

    // Compute full softmax distribution.
    std::vector<float> probs = softmax(logits, 1.0f);

    // Build (prob, index) pairs sorted by probability descending.
    std::vector<std::pair<float, int>> sorted_probs(probs.size());
    for (std::size_t i = 0; i < probs.size(); ++i) {
        sorted_probs[i] = {probs[i], static_cast<int>(i)};
    }
    std::sort(sorted_probs.begin(), sorted_probs.end(),
              [](const auto& a, const auto& b) { return a.first > b.first; });

    // Accumulate until cumulative probability exceeds p.
    float cumulative = 0.0f;
    std::vector<std::pair<float, int>> nucleus;
    for (auto& [prob, idx] : sorted_probs) {
        nucleus.push_back({prob, idx});
        cumulative += prob;
        if (cumulative >= p) break;
    }

    // Re-normalise the nucleus.
    float nucleus_sum = 0.0f;
    for (auto& [prob, _] : nucleus) nucleus_sum += prob;
    if (nucleus_sum == 0.0f) nucleus_sum = 1.0f;

    // Sample from the nucleus.
    float r = lcg_float(rng_state);
    float cum = 0.0f;
    for (auto& [prob, idx] : nucleus) {
        cum += prob / nucleus_sum;
        if (r <= cum) return idx;
    }
    return nucleus.back().second;
}

// ─── UNIFIED DISPATCHER ──────────────────────────────────────────────────────

int SamplingStrategies::sample(const std::vector<float>& logits,
                               const SamplingConfig& config,
                               uint64_t& rng_state) {
    if (logits.empty()) {
        return -1;
    }

    // Work on a local copy so we can apply penalties without mutating the caller's data.
    std::vector<float> local_logits = logits;

    // Repetition penalty would be applied here if recent_tokens were passed in.
    // (The caller should invoke apply_repetition_penalty before calling sample
    //  if they have a recent-token list.)

    if (!config.do_sample) {
        // Greedy: apply temperature scaling then argmax.
        std::vector<float> probs = softmax(local_logits, config.temperature);
        return static_cast<int>(
            std::max_element(probs.begin(), probs.end()) - probs.begin()
        );
    }

    // Apply temperature via softmax — store scaled logits for dispatch.
    // We convert back to log-space to keep the interface consistent:
    //   scaled_logit[i] = logit[i] / temperature
    float temp = (config.temperature > 0.0f) ? config.temperature : 1.0f;
    for (auto& l : local_logits) l /= temp;

    // Prefer top_k if set; otherwise top_p; otherwise full softmax sample.
    if (config.top_k > 0) {
        // Undo temperature scaling — top_k_sample applies softmax internally.
        for (auto& l : local_logits) l *= temp;
        return top_k_sample(local_logits, config.top_k, rng_state);
    }

    if (config.top_p > 0.0f && config.top_p < 1.0f) {
        for (auto& l : local_logits) l *= temp;
        return top_p_sample(local_logits, config.top_p, rng_state);
    }

    // Full distribution sample.
    for (auto& l : local_logits) l *= temp;
    std::vector<float> probs = softmax(local_logits, 1.0f);
    float r = lcg_float(rng_state);
    float cum = 0.0f;
    for (std::size_t i = 0; i < probs.size(); ++i) {
        cum += probs[i];
        if (r <= cum) return static_cast<int>(i);
    }
    return static_cast<int>(probs.size() - 1);
}

} // namespace llmquant
