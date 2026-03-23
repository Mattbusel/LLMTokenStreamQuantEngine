#include "TokenSampler.h"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <numeric>
#include <stdexcept>

namespace llmquant {

// ---------------------------------------------------------------------------
// Construction
// ---------------------------------------------------------------------------

TokenSampler::TokenSampler(SamplingConfig config)
    : config_(config), lcg_state_(config.seed)
{
}

// ---------------------------------------------------------------------------
// LCG
// ---------------------------------------------------------------------------

double TokenSampler::lcg_next()
{
    lcg_state_ = lcg_state_ * 6364136223846793005ULL + 1442695040888963407ULL;
    // Map to [0, 1) using upper 32 bits
    return static_cast<double>(lcg_state_ >> 32) / 4294967296.0;
}

void TokenSampler::set_seed(uint64_t seed)
{
    lcg_state_ = seed;
    config_.seed = seed;
}

// ---------------------------------------------------------------------------
// Softmax
// ---------------------------------------------------------------------------

std::vector<float> TokenSampler::softmax(const std::vector<float>& logits)
{
    if (logits.empty()) return {};
    float max_val = *std::max_element(logits.begin(), logits.end());
    std::vector<float> probs(logits.size());
    float sum = 0.0f;
    for (std::size_t i = 0; i < logits.size(); ++i) {
        probs[i] = std::exp(logits[i] - max_val);
        sum += probs[i];
    }
    for (auto& p : probs) p /= sum;
    return probs;
}

// ---------------------------------------------------------------------------
// Temperature
// ---------------------------------------------------------------------------

std::vector<float> TokenSampler::apply_temperature(const std::vector<float>& logits) const
{
    if (logits.empty()) return {};
    float t = config_.temperature;
    if (t <= 0.0f) t = 1e-6f;
    std::vector<float> out(logits.size());
    for (std::size_t i = 0; i < logits.size(); ++i)
        out[i] = logits[i] / t;
    return out;
}

// ---------------------------------------------------------------------------
// Top-k
// ---------------------------------------------------------------------------

std::vector<float> TokenSampler::apply_top_k(const std::vector<float>& probs, int k) const
{
    if (probs.empty()) return {};
    int n = static_cast<int>(probs.size());
    k = std::max(1, std::min(k, n));

    // Find the k-th largest probability using a partial sort of indices
    std::vector<int> idx(n);
    std::iota(idx.begin(), idx.end(), 0);
    std::partial_sort(idx.begin(), idx.begin() + k, idx.end(),
                      [&](int a, int b) { return probs[a] > probs[b]; });

    std::vector<float> out(n, 0.0f);
    float sum = 0.0f;
    for (int i = 0; i < k; ++i) {
        out[idx[i]] = probs[idx[i]];
        sum += probs[idx[i]];
    }
    if (sum > 0.0f)
        for (int i = 0; i < k; ++i)
            out[idx[i]] /= sum;
    return out;
}

// ---------------------------------------------------------------------------
// Top-p (nucleus sampling)
// ---------------------------------------------------------------------------

std::vector<float> TokenSampler::apply_top_p(const std::vector<float>& probs, float p) const
{
    if (probs.empty()) return {};
    int n = static_cast<int>(probs.size());

    // Sort indices by descending probability
    std::vector<int> idx(n);
    std::iota(idx.begin(), idx.end(), 0);
    std::sort(idx.begin(), idx.end(),
              [&](int a, int b) { return probs[a] > probs[b]; });

    // Accumulate until we exceed p
    float cumulative = 0.0f;
    std::vector<float> out(n, 0.0f);
    float sum = 0.0f;
    for (int i : idx) {
        if (cumulative >= p && sum > 0.0f) break;
        out[i] = probs[i];
        sum += probs[i];
        cumulative += probs[i];
    }
    // Renormalise
    if (sum > 0.0f)
        for (auto& v : out) v /= sum;
    return out;
}

// ---------------------------------------------------------------------------
// Repetition penalty
// ---------------------------------------------------------------------------

std::vector<float> TokenSampler::apply_repetition_penalty(
    const std::vector<float>& logits,
    const std::vector<int>& recent_tokens) const
{
    if (logits.empty()) return {};
    std::vector<float> out = logits;
    float penalty = config_.repetition_penalty;
    for (int tok : recent_tokens) {
        if (tok >= 0 && tok < static_cast<int>(out.size())) {
            if (out[tok] > 0.0f)
                out[tok] /= penalty;
            else
                out[tok] *= penalty;
        }
    }
    return out;
}

// ---------------------------------------------------------------------------
// Multinomial sample
// ---------------------------------------------------------------------------

int TokenSampler::sample(const std::vector<float>& probs)
{
    if (probs.empty()) return 0;
    double r = lcg_next();
    double cumulative = 0.0;
    for (std::size_t i = 0; i < probs.size(); ++i) {
        cumulative += static_cast<double>(probs[i]);
        if (r < cumulative) return static_cast<int>(i);
    }
    return static_cast<int>(probs.size()) - 1;
}

// ---------------------------------------------------------------------------
// Greedy
// ---------------------------------------------------------------------------

int TokenSampler::greedy(const std::vector<float>& logits) const
{
    if (logits.empty()) return 0;
    return static_cast<int>(
        std::max_element(logits.begin(), logits.end()) - logits.begin());
}

// ---------------------------------------------------------------------------
// Full pipeline
// ---------------------------------------------------------------------------

int TokenSampler::sample_from_logits(
    const std::vector<float>& logits,
    const std::vector<int>& recent_tokens)
{
    // 1. Repetition penalty
    auto penalized = apply_repetition_penalty(logits, recent_tokens);
    // 2. Temperature
    auto tempered = apply_temperature(penalized);
    // 3. Softmax
    auto probs = softmax(tempered);
    // 4. Top-k
    probs = apply_top_k(probs, config_.top_k);
    // 5. Top-p
    probs = apply_top_p(probs, config_.top_p);
    // 6. Sample
    return sample(probs);
}

}  // namespace llmquant
