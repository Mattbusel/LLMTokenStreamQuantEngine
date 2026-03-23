// LogitsProcessor.cpp — Implementation of logit processing transformations.

#include "LogitsProcessor.h"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <numeric>
#include <stdexcept>

namespace llmquant {

// ---------------------------------------------------------------------------
// apply_temperature
// ---------------------------------------------------------------------------

void LogitsProcessor::apply_temperature(
    std::vector<float>& logits, float temperature) const
{
    if (temperature == 1.0f || logits.empty()) {
        return;
    }
    if (temperature <= 0.0f) {
        throw std::invalid_argument("temperature must be > 0");
    }
    for (float& l : logits) {
        l /= temperature;
    }
}

// ---------------------------------------------------------------------------
// apply_repetition_penalty
// ---------------------------------------------------------------------------

void LogitsProcessor::apply_repetition_penalty(
    std::vector<float>&       logits,
    const std::vector<size_t>& generated_ids,
    float                      penalty) const
{
    if (penalty == 1.0f || generated_ids.empty() || logits.empty()) {
        return;
    }
    for (size_t id : generated_ids) {
        if (id >= logits.size()) {
            continue;
        }
        float& l = logits[id];
        if (l > 0.0f) {
            l /= penalty;
        } else {
            l *= penalty;
        }
    }
}

// ---------------------------------------------------------------------------
// apply_top_k
// ---------------------------------------------------------------------------

void LogitsProcessor::apply_top_k(
    std::vector<float>& logits, size_t k) const
{
    if (k == 0 || k >= logits.size()) {
        return;
    }

    // Find the k-th largest value via partial sort on indices.
    std::vector<size_t> indices(logits.size());
    std::iota(indices.begin(), indices.end(), 0u);
    std::partial_sort(indices.begin(), indices.begin() + static_cast<std::ptrdiff_t>(k),
                      indices.end(),
                      [&](size_t a, size_t b) { return logits[a] > logits[b]; });

    float threshold = logits[indices[k - 1]];

    for (size_t i = 0; i < logits.size(); ++i) {
        if (logits[i] < threshold) {
            logits[i] = NEG_INF;
        }
    }

    // Make sure exactly the top-k remain (handle ties by keeping
    // the first k when we iterate over sorted indices).
    size_t kept = 0;
    for (size_t idx : indices) {
        if (kept < k) {
            ++kept;
        } else {
            logits[idx] = NEG_INF;
        }
    }
}

// ---------------------------------------------------------------------------
// apply_top_p
// ---------------------------------------------------------------------------

void LogitsProcessor::apply_top_p(
    std::vector<float>& logits, float p) const
{
    if (p >= 1.0f || logits.empty()) {
        return;
    }

    // Compute softmax probabilities.
    std::vector<float> probs = softmax(logits);

    // Sort indices by probability descending.
    std::vector<size_t> indices(probs.size());
    std::iota(indices.begin(), indices.end(), 0u);
    std::sort(indices.begin(), indices.end(),
              [&](size_t a, size_t b) { return probs[a] > probs[b]; });

    // Find the nucleus: smallest set whose cumulative probability >= p.
    float cumulative = 0.0f;
    size_t nucleus_size = 0;
    for (size_t idx : indices) {
        cumulative += probs[idx];
        ++nucleus_size;
        if (cumulative >= p) {
            break;
        }
    }

    // Mask tokens outside the nucleus.
    for (size_t i = nucleus_size; i < indices.size(); ++i) {
        logits[indices[i]] = NEG_INF;
    }
}

// ---------------------------------------------------------------------------
// softmax
// ---------------------------------------------------------------------------

std::vector<float> LogitsProcessor::softmax(
    const std::vector<float>& logits) const
{
    if (logits.empty()) {
        return {};
    }

    float max_val = *std::max_element(logits.begin(), logits.end());

    std::vector<float> probs(logits.size());
    float sum = 0.0f;
    for (size_t i = 0; i < logits.size(); ++i) {
        probs[i] = std::exp(logits[i] - max_val);
        sum += probs[i];
    }
    if (sum > 0.0f) {
        for (float& p : probs) {
            p /= sum;
        }
    }
    return probs;
}

// ---------------------------------------------------------------------------
// sample
// ---------------------------------------------------------------------------

size_t LogitsProcessor::sample(
    const std::vector<float>& probs, uint64_t seed) const
{
    if (probs.empty()) {
        return 0;
    }

    // Simple LCG RNG to generate a uniform [0,1) value from the seed.
    uint64_t state = seed * 6364136223846793005ULL + 1442695040888963407ULL;
    state ^= state >> 33;
    state *= 0xff51afd7ed558ccdULL;
    state ^= state >> 33;
    state *= 0xc4ceb9fe1a85ec53ULL;
    state ^= state >> 33;

    double u = static_cast<double>(state) / static_cast<double>(UINT64_MAX);
    double cumulative = 0.0;
    for (size_t i = 0; i < probs.size(); ++i) {
        cumulative += static_cast<double>(probs[i]);
        if (u < cumulative) {
            return i;
        }
    }
    // Fallback: last valid token.
    return probs.size() - 1;
}

// ---------------------------------------------------------------------------
// greedy
// ---------------------------------------------------------------------------

size_t LogitsProcessor::greedy(const std::vector<float>& logits) const
{
    if (logits.empty()) {
        return 0;
    }
    return static_cast<size_t>(
        std::max_element(logits.begin(), logits.end()) - logits.begin());
}

}  // namespace llmquant
