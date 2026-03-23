#pragma once

/// @file PositionalEncoding.h
/// @brief Positional encoding schemes for transformer models.
///
/// Provides sinusoidal, learned, rotary (RoPE), and ALiBi encodings.
///
/// Feature flag: LLMQUANT_POSITIONAL_ENCODING_ENABLED

#ifdef LLMQUANT_POSITIONAL_ENCODING_ENABLED

#include <cstddef>
#include <unordered_map>
#include <utility>
#include <vector>

namespace llmquant {

// ---------------------------------------------------------------------------
// Abstract base
// ---------------------------------------------------------------------------

/// Abstract interface for positional encodings.
class PositionalEncoding {
public:
    virtual ~PositionalEncoding() = default;

    /// Compute the positional encoding vector for @p position in a model of
    /// dimensionality @p dim.
    virtual std::vector<float> encode(int position, int dim) const = 0;
};

// ---------------------------------------------------------------------------
// Sinusoidal encoding (Vaswani et al., 2017)
// ---------------------------------------------------------------------------

/// PE[pos, 2k]   = sin(pos / 10000^(2k/d))
/// PE[pos, 2k+1] = cos(pos / 10000^(2k/d))
class SinusoidalEncoding : public PositionalEncoding {
public:
    std::vector<float> encode(int position, int dim) const override;
};

// ---------------------------------------------------------------------------
// Learned positional encoding
// ---------------------------------------------------------------------------

/// Stores a learned embedding table (max_positions × dim).
/// Initialized with sinusoidal values; individual positions may be updated.
class LearnedEncoding : public PositionalEncoding {
public:
    /// @param max_positions  Maximum sequence length supported.
    /// @param dim            Embedding dimensionality.
    explicit LearnedEncoding(int max_positions, int dim);

    std::vector<float> encode(int position, int dim) const override;

    /// Replace the embedding for @p position with @p new_embedding.
    /// @throws std::out_of_range if position >= max_positions or
    ///         new_embedding.size() != dim_.
    void update(int position, const std::vector<float>& new_embedding);

private:
    int max_positions_;
    int dim_;
    /// Table stored as a flat vector: table_[pos * dim_ + d].
    std::vector<float> table_;
};

// ---------------------------------------------------------------------------
// Rotary Positional Encoding (RoPE) – Su et al., 2021
// ---------------------------------------------------------------------------

/// Applies rotary position embeddings to query/key vectors.
///
/// rope_freq[i] = 1 / 10000^(2i / dim)
///
/// Pair (q_{2i}, q_{2i+1}) is rotated by angle: position * rope_freq[i]
class RotaryPositionalEncoding : public PositionalEncoding {
public:
    /// @param dim  Must be even.
    explicit RotaryPositionalEncoding(int dim);

    /// Returns a dummy encoding vector (for interface compatibility).
    /// Use apply_rope() for the actual rotation.
    std::vector<float> encode(int position, int dim) const override;

    /// Apply RoPE to a query-key pair at the given position.
    /// @returns {rotated_query, rotated_key}
    std::pair<std::vector<float>, std::vector<float>> apply_rope(
        const std::vector<float>& query,
        const std::vector<float>& key,
        int position) const;

private:
    int dim_;
    std::vector<float> rope_freq_; // size = dim / 2
};

// ---------------------------------------------------------------------------
// ALiBi positional bias (Press et al., 2022)
// ---------------------------------------------------------------------------

/// Adds a linear distance bias to attention scores instead of position
/// embeddings.
///
/// Head h uses slope m_h = 2^(-8 * h / num_heads).
/// bias[i][j] = -m_h * |i - j|
class AlibiPositionalBias : public PositionalEncoding {
public:
    /// @param num_heads  Number of attention heads.
    explicit AlibiPositionalBias(int num_heads);

    /// Returns a zero vector (ALiBi biases are applied to attention scores,
    /// not added to token representations).
    std::vector<float> encode(int position, int dim) const override;

    /// Compute the ALiBi bias matrix for all heads.
    ///
    /// @returns  Outer vector has size num_heads; each inner vector is a flat
    ///           seq_len × seq_len bias matrix (row-major).
    std::vector<std::vector<float>> compute_bias(int seq_len, int num_heads) const;

private:
    int num_heads_;
    std::vector<float> slopes_; // size = num_heads
};

} // namespace llmquant

#endif // LLMQUANT_POSITIONAL_ENCODING_ENABLED
