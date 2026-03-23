/// @file PositionalEncoding.cpp
/// @brief Implementation of positional encoding schemes.

#ifdef LLMQUANT_POSITIONAL_ENCODING_ENABLED

#include "PositionalEncoding.h"
#include <cmath>
#include <stdexcept>

namespace llmquant {

// ---------------------------------------------------------------------------
// SinusoidalEncoding
// ---------------------------------------------------------------------------

std::vector<float> SinusoidalEncoding::encode(int position, int dim) const {
    if (dim <= 0) {
        throw std::invalid_argument("dim must be > 0");
    }
    std::vector<float> pe(dim, 0.0f);
    for (int k = 0; k < dim / 2; ++k) {
        double exponent = static_cast<double>(2 * k) / static_cast<double>(dim);
        double denom = std::pow(10000.0, exponent);
        pe[2 * k]     = static_cast<float>(std::sin(position / denom));
        pe[2 * k + 1] = static_cast<float>(std::cos(position / denom));
    }
    // Handle odd dim: last element is sin only.
    if (dim % 2 != 0) {
        int k = dim / 2;
        double exponent = static_cast<double>(2 * k) / static_cast<double>(dim);
        double denom = std::pow(10000.0, exponent);
        pe[2 * k] = static_cast<float>(std::sin(position / denom));
    }
    return pe;
}

// ---------------------------------------------------------------------------
// LearnedEncoding
// ---------------------------------------------------------------------------

LearnedEncoding::LearnedEncoding(int max_positions, int dim)
    : max_positions_(max_positions), dim_(dim), table_(max_positions * dim, 0.0f)
{
    if (max_positions <= 0 || dim <= 0) {
        throw std::invalid_argument("max_positions and dim must be > 0");
    }
    // Initialise with sinusoidal values.
    SinusoidalEncoding sinusoidal;
    for (int pos = 0; pos < max_positions; ++pos) {
        auto pe = sinusoidal.encode(pos, dim);
        for (int d = 0; d < dim; ++d) {
            table_[pos * dim + d] = pe[d];
        }
    }
}

std::vector<float> LearnedEncoding::encode(int position, int /*dim*/) const {
    if (position < 0 || position >= max_positions_) {
        throw std::out_of_range("position out of range for LearnedEncoding");
    }
    return std::vector<float>(
        table_.begin() + position * dim_,
        table_.begin() + position * dim_ + dim_);
}

void LearnedEncoding::update(int position, const std::vector<float>& new_embedding) {
    if (position < 0 || position >= max_positions_) {
        throw std::out_of_range("position out of range for LearnedEncoding::update");
    }
    if (static_cast<int>(new_embedding.size()) != dim_) {
        throw std::invalid_argument(
            "new_embedding.size() must match dim");
    }
    for (int d = 0; d < dim_; ++d) {
        table_[position * dim_ + d] = new_embedding[d];
    }
}

// ---------------------------------------------------------------------------
// RotaryPositionalEncoding
// ---------------------------------------------------------------------------

RotaryPositionalEncoding::RotaryPositionalEncoding(int dim) : dim_(dim) {
    if (dim <= 0 || dim % 2 != 0) {
        throw std::invalid_argument("dim must be a positive even integer for RoPE");
    }
    rope_freq_.resize(dim / 2);
    for (int i = 0; i < dim / 2; ++i) {
        double exponent = static_cast<double>(2 * i) / static_cast<double>(dim);
        rope_freq_[i] = static_cast<float>(1.0 / std::pow(10000.0, exponent));
    }
}

std::vector<float> RotaryPositionalEncoding::encode(int /*position*/, int dim) const {
    // RoPE does not produce an additive encoding; return zeros.
    return std::vector<float>(dim, 0.0f);
}

std::pair<std::vector<float>, std::vector<float>>
RotaryPositionalEncoding::apply_rope(
    const std::vector<float>& query,
    const std::vector<float>& key,
    int position) const
{
    if (static_cast<int>(query.size()) != dim_ ||
        static_cast<int>(key.size()) != dim_) {
        throw std::invalid_argument(
            "query and key must have size equal to dim_");
    }

    auto rotate = [&](const std::vector<float>& v) {
        std::vector<float> out(dim_);
        for (int i = 0; i < dim_ / 2; ++i) {
            float angle = static_cast<float>(position) * rope_freq_[i];
            float cos_a = std::cos(angle);
            float sin_a = std::sin(angle);
            float v0 = v[2 * i];
            float v1 = v[2 * i + 1];
            out[2 * i]     = v0 * cos_a - v1 * sin_a;
            out[2 * i + 1] = v0 * sin_a + v1 * cos_a;
        }
        return out;
    };

    return {rotate(query), rotate(key)};
}

// ---------------------------------------------------------------------------
// AlibiPositionalBias
// ---------------------------------------------------------------------------

AlibiPositionalBias::AlibiPositionalBias(int num_heads) : num_heads_(num_heads) {
    if (num_heads <= 0) {
        throw std::invalid_argument("num_heads must be > 0");
    }
    slopes_.resize(num_heads);
    for (int h = 0; h < num_heads; ++h) {
        // m_h = 2^(-8 * h / num_heads)  (0-indexed heads, 1-indexed in paper)
        slopes_[h] = static_cast<float>(
            std::pow(2.0, -8.0 * static_cast<double>(h + 1) / static_cast<double>(num_heads)));
    }
}

std::vector<float> AlibiPositionalBias::encode(int /*position*/, int dim) const {
    return std::vector<float>(dim, 0.0f);
}

std::vector<std::vector<float>>
AlibiPositionalBias::compute_bias(int seq_len, int num_heads) const {
    if (num_heads != num_heads_) {
        throw std::invalid_argument(
            "num_heads passed to compute_bias must match constructor num_heads");
    }
    std::vector<std::vector<float>> biases(num_heads,
        std::vector<float>(seq_len * seq_len, 0.0f));

    for (int h = 0; h < num_heads; ++h) {
        float slope = slopes_[h];
        for (int i = 0; i < seq_len; ++i) {
            for (int j = 0; j < seq_len; ++j) {
                biases[h][i * seq_len + j] = -slope * std::abs(i - j);
            }
        }
    }
    return biases;
}

} // namespace llmquant

#endif // LLMQUANT_POSITIONAL_ENCODING_ENABLED
