#pragma once

/// @file ContextEncoder.h
/// @brief Context encoding and position embeddings for transformer inference.
///
/// Provides sinusoidal positional encoding, token encoding, RoPE (Rotary
/// Position Embedding), and ALiBi (Attention with Linear Biases) for the
/// LLMTokenStreamQuantEngine inference pipeline.
///
/// Feature flag: LLMQUANT_CONTEXT_ENCODER_ENABLED

#ifdef LLMQUANT_CONTEXT_ENCODER_ENABLED

#include <cmath>
#include <cstddef>
#include <utility>
#include <vector>

namespace llmquant {

/// Positional encoding data for a single (position, dimension) pair.
struct PositionalEncoding {
    int position;
    int dim;
    float value;
};

/// Context encoder implementing sinusoidal PE, RoPE, and ALiBi.
class ContextEncoder {
public:
    /// @param max_seq_len  Maximum sequence length supported.
    /// @param embed_dim    Embedding dimensionality (should be even).
    ContextEncoder(int max_seq_len, int embed_dim);

    /// Sinusoidal positional encoding for a single (position, dim_index) pair.
    /// PE(pos, 2i)   = sin(pos / 10000^(2i/d))
    /// PE(pos, 2i+1) = cos(pos / 10000^(2i/d))
    float sinusoidal_pe(int position, int dim) const;

    /// Full embed_dim-length positional encoding vector for a position.
    std::vector<float> encode_position(int position) const;

    /// Token encoding: hash of token_id mapped to float, added to PE.
    std::vector<float> encode_token(int token_id, int position) const;

    /// Encode a full sequence of token ids.
    std::vector<std::vector<float>> encode_sequence(
        const std::vector<int>& token_ids) const;

    /// RoPE angle for a given position and dimension pair.
    /// theta = position * 10000^(-2*dim_pair / embed_dim_)
    float rope_angle(int position, int dim_pair) const;

    /// Apply RoPE rotation to query/key scalars.
    /// @returns {rotated_q, rotated_k}
    std::pair<float, float> apply_rope(
        float q, float k, int position, int dim_pair) const;

    /// ALiBi bias vector for a sequence of length seq_len and attention head.
    /// bias[j] = -|query_pos - j| * slope where slope = 2^(-8*head/num_heads).
    /// num_heads is fixed at 8.
    std::vector<float> alibi_bias(int seq_len, int head) const;

private:
    int max_seq_len_;
    int embed_dim_;

    static constexpr int kNumHeads = 8;
};

} // namespace llmquant

#endif // LLMQUANT_CONTEXT_ENCODER_ENABLED
