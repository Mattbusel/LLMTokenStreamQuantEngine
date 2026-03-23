#include "AttentionScorer.h"

#include <algorithm>
#include <cmath>
#include <numeric>
#include <stdexcept>

namespace llmquant {

AttentionScorer::AttentionScorer(int num_heads, int head_dim)
    : num_heads_(num_heads), head_dim_(head_dim) {}

std::vector<float> AttentionScorer::scaled_dot_product(
    const std::vector<float>& q,
    const std::vector<float>& k) const
{
    if (head_dim_ <= 0) {
        return {};
    }
    const int seq_len = static_cast<int>(k.size()) / head_dim_;
    if (seq_len <= 0) {
        return {};
    }
    const float scale = std::sqrt(static_cast<float>(head_dim_));
    std::vector<float> scores(static_cast<size_t>(seq_len), 0.0f);
    for (int j = 0; j < seq_len; ++j) {
        float dot = 0.0f;
        for (int d = 0; d < head_dim_; ++d) {
            dot += q[static_cast<size_t>(d)] *
                   k[static_cast<size_t>(j * head_dim_ + d)];
        }
        scores[static_cast<size_t>(j)] = dot / scale;
    }
    return scores;
}

std::vector<float> AttentionScorer::softmax(const std::vector<float>& scores) const {
    if (scores.empty()) {
        return {};
    }
    const float max_val = *std::max_element(scores.begin(), scores.end());
    std::vector<float> result(scores.size());
    float sum = 0.0f;
    for (size_t i = 0; i < scores.size(); ++i) {
        result[i] = std::exp(scores[i] - max_val);
        sum += result[i];
    }
    if (sum > 0.0f) {
        for (auto& v : result) {
            v /= sum;
        }
    }
    return result;
}

std::vector<float> AttentionScorer::apply_causal_mask(
    const std::vector<float>& scores,
    int seq_len) const
{
    // For a query at position (seq_len - 1), mask out all positions >= query_pos.
    // Here we treat the score vector as belonging to query at the last valid
    // position: score[j] is masked when j > query_pos.
    // Since the caller passes a flat score vector of length seq_len for one query,
    // we mask positions beyond the query index (assumed to be seq_len - 1).
    std::vector<float> masked = scores;
    const float neg_inf = -std::numeric_limits<float>::infinity();
    const int query_pos = seq_len - 1;
    for (int j = query_pos + 1; j < static_cast<int>(masked.size()); ++j) {
        masked[static_cast<size_t>(j)] = neg_inf;
    }
    return masked;
}

std::vector<float> AttentionScorer::apply_attention_bias(
    const std::vector<float>& scores,
    const std::vector<float>& bias) const
{
    const size_t n = std::min(scores.size(), bias.size());
    std::vector<float> result = scores;
    for (size_t i = 0; i < n; ++i) {
        result[i] += bias[i];
    }
    return result;
}

float AttentionScorer::attention_entropy(const std::vector<float>& weights) const {
    constexpr float eps = 1e-9f;
    float entropy = 0.0f;
    for (const float w : weights) {
        if (w > 0.0f) {
            entropy -= w * std::log(w + eps);
        }
    }
    return entropy;
}

bool AttentionScorer::is_attending_uniformly(
    const std::vector<float>& weights,
    float threshold) const
{
    if (weights.empty()) {
        return true;
    }
    const float max_w = *std::max_element(weights.begin(), weights.end());
    const float min_w = *std::min_element(weights.begin(), weights.end());
    return (max_w - min_w) < threshold;
}

std::vector<std::vector<float>> AttentionScorer::multi_head_attention(
    const std::vector<std::vector<float>>& queries,
    const std::vector<std::vector<float>>& keys) const
{
    const int n = std::min(static_cast<int>(queries.size()),
                           static_cast<int>(keys.size()));
    std::vector<std::vector<float>> result(static_cast<size_t>(n));
    for (int h = 0; h < n; ++h) {
        result[static_cast<size_t>(h)] =
            scaled_dot_product(queries[static_cast<size_t>(h)],
                               keys[static_cast<size_t>(h)]);
    }
    return result;
}

} // namespace llmquant
