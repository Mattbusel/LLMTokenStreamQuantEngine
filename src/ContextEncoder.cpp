/// @file ContextEncoder.cpp
/// @brief Implementation of ContextEncoder positional encoding schemes.

#ifdef LLMQUANT_CONTEXT_ENCODER_ENABLED

#include "ContextEncoder.h"
#include <cmath>
#include <stdexcept>

namespace llmquant {

ContextEncoder::ContextEncoder(int max_seq_len, int embed_dim)
    : max_seq_len_(max_seq_len), embed_dim_(embed_dim)
{
    if (max_seq_len <= 0) {
        throw std::invalid_argument("max_seq_len must be > 0");
    }
    if (embed_dim <= 0) {
        throw std::invalid_argument("embed_dim must be > 0");
    }
}

float ContextEncoder::sinusoidal_pe(int position, int dim) const {
    // dim is the index into the embedding vector [0, embed_dim_)
    // Even indices: sin, Odd indices: cos
    int pair = dim / 2;
    double exponent = static_cast<double>(2 * pair) / static_cast<double>(embed_dim_);
    double denom = std::pow(10000.0, exponent);
    double angle = static_cast<double>(position) / denom;
    if (dim % 2 == 0) {
        return static_cast<float>(std::sin(angle));
    } else {
        return static_cast<float>(std::cos(angle));
    }
}

std::vector<float> ContextEncoder::encode_position(int position) const {
    std::vector<float> pe(static_cast<std::size_t>(embed_dim_), 0.0f);
    for (int d = 0; d < embed_dim_; ++d) {
        pe[static_cast<std::size_t>(d)] = sinusoidal_pe(position, d);
    }
    return pe;
}

std::vector<float> ContextEncoder::encode_token(int token_id, int position) const {
    // Map token_id to a float embedding via hash, then add positional encoding
    // Simple hash: token embedding = token_id modulo small prime, normalized
    std::vector<float> pe = encode_position(position);
    // Token contribution: use a simple hash spread across dimensions
    const float scale = 1.0f / static_cast<float>(embed_dim_);
    for (int d = 0; d < embed_dim_; ++d) {
        // Deterministic hash per (token_id, dim)
        unsigned int h = static_cast<unsigned int>(token_id * 2654435761u) ^
                         static_cast<unsigned int>(d * 2246822519u);
        float token_val = static_cast<float>(h % 1000u) * scale * 0.01f;
        pe[static_cast<std::size_t>(d)] += token_val;
    }
    return pe;
}

std::vector<std::vector<float>> ContextEncoder::encode_sequence(
    const std::vector<int>& token_ids) const
{
    std::vector<std::vector<float>> result;
    result.reserve(token_ids.size());
    for (std::size_t i = 0; i < token_ids.size(); ++i) {
        result.push_back(encode_token(token_ids[i], static_cast<int>(i)));
    }
    return result;
}

float ContextEncoder::rope_angle(int position, int dim_pair) const {
    // theta = position * 10000^(-2 * dim_pair / embed_dim_)
    double exponent = -2.0 * static_cast<double>(dim_pair) / static_cast<double>(embed_dim_);
    double freq = std::pow(10000.0, exponent);
    return static_cast<float>(static_cast<double>(position) * freq);
}

std::pair<float, float> ContextEncoder::apply_rope(
    float q, float k, int position, int dim_pair) const
{
    float theta = rope_angle(position, dim_pair);
    float cos_theta = std::cos(theta);
    float sin_theta = std::sin(theta);
    // Standard 2D rotation applied to scalar pair
    float q_rot = q * cos_theta - k * sin_theta;
    float k_rot = q * sin_theta + k * cos_theta;
    return {q_rot, k_rot};
}

std::vector<float> ContextEncoder::alibi_bias(int seq_len, int head) const {
    // slope = 2^(-8 * head / num_heads)
    double slope = std::pow(2.0, -8.0 * static_cast<double>(head) / static_cast<double>(kNumHeads));
    // For a query at position seq_len-1 (last position), compute bias to each key
    // bias[j] = -|query_pos - j| * slope
    int query_pos = seq_len - 1;
    std::vector<float> bias(static_cast<std::size_t>(seq_len));
    for (int j = 0; j < seq_len; ++j) {
        int dist = std::abs(query_pos - j);
        bias[static_cast<std::size_t>(j)] = static_cast<float>(-dist * slope);
    }
    return bias;
}

} // namespace llmquant

#endif // LLMQUANT_CONTEXT_ENCODER_ENABLED
