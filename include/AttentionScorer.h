#pragma once

/// @file AttentionScorer.h
/// @brief Attention score computation and masking utilities.

#include <cmath>
#include <limits>
#include <vector>

namespace llmquant {

/// Computes scaled dot-product attention scores, applies masks, and measures
/// attention entropy for a multi-head attention layer.
class AttentionScorer {
public:
    /// @param num_heads  Number of attention heads.
    /// @param head_dim   Dimensionality of each head (d_k).
    AttentionScorer(int num_heads, int head_dim);

    /// Scaled dot-product: returns score vector of length |k| where
    /// score[j] = (q · k[j]) / sqrt(d_k).
    ///
    /// @param q  Query vector (length head_dim_).
    /// @param k  Flat key matrix (seq_len * head_dim_ values, row-major).
    /// @returns  Score vector of length seq_len.
    std::vector<float> scaled_dot_product(
        const std::vector<float>& q,
        const std::vector<float>& k) const;

    /// Numerically stable softmax.
    std::vector<float> softmax(const std::vector<float>& scores) const;

    /// Apply causal (lower-triangular) mask: set future positions to -infinity.
    ///
    /// @param scores   Flat score vector of length seq_len (scores for one query).
    /// @param seq_len  Total sequence length.
    /// @returns        Masked score vector.
    std::vector<float> apply_causal_mask(
        const std::vector<float>& scores,
        int seq_len) const;

    /// Element-wise add bias to scores.
    std::vector<float> apply_attention_bias(
        const std::vector<float>& scores,
        const std::vector<float>& bias) const;

    /// Shannon entropy of an attention weight distribution.
    /// H = -sum(w * log(w + eps))
    float attention_entropy(const std::vector<float>& weights) const;

    /// True when max_weight - min_weight < threshold (near-uniform distribution).
    bool is_attending_uniformly(
        const std::vector<float>& weights,
        float threshold = 0.1f) const;

    /// Compute per-head attention scores for a batch of queries and keys.
    ///
    /// @param queries  One vector per head of length head_dim_.
    /// @param keys     One vector per head of length head_dim_ (flat key per head).
    /// @returns        Per-head score vectors.
    std::vector<std::vector<float>> multi_head_attention(
        const std::vector<std::vector<float>>& queries,
        const std::vector<std::vector<float>>& keys) const;

private:
    int num_heads_;
    int head_dim_;
};

} // namespace llmquant
