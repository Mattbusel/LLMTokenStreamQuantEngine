#pragma once

/// @file AttentionMaskBuilder.h
/// @brief Attention mask construction utilities.
///
/// Supports causal (lower-triangular), full, sliding-window, and prefix-LM
/// attention patterns.  All builders are pure static methods.
///
/// Feature flag: LLMQUANT_ATTENTION_MASK_ENABLED

#ifdef LLMQUANT_ATTENTION_MASK_ENABLED

#include <string>
#include <vector>

namespace llmquant {

/// A 2-D boolean attention mask together with the sequence length it was
/// built for.
struct AttentionMask {
    /// mask[i][j] == true  → position i may attend to position j.
    std::vector<std::vector<bool>> mask;
    /// Sequence length (both dimensions are seq_len × seq_len).
    int seq_len{0};
};

/// Factory class for common attention mask patterns.
class AttentionMaskBuilder {
public:
    /// Build a causal (lower-triangular) mask.
    /// Position i attends to all positions j ≤ i.
    static AttentionMask build_causal(int seq_len);

    /// Build a full (all-true) attention mask.
    /// Every position attends to every other position.
    static AttentionMask build_full(int seq_len);

    /// Build a sliding-window mask.
    /// Position i attends to positions max(0, i - window_size + 1) … i.
    static AttentionMask build_sliding_window(int seq_len, int window_size);

    /// Build a prefix-LM mask.
    ///
    /// * Prefix positions [0, prefix_len) attend to all other prefix positions
    ///   (bidirectional within the prefix).
    /// * Non-prefix positions [prefix_len, seq_len) attend causally
    ///   (only to positions ≤ their own index).
    static AttentionMask build_prefix_lm(int seq_len, int prefix_len);

    /// Apply an attention mask to a flat (row-major) score vector.
    ///
    /// Positions where mask[i][j] == false are replaced with @p mask_value
    /// (typically a large negative number to suppress softmax weight).
    ///
    /// @param attention_scores  Flat seq_len × seq_len score vector.
    /// @param mask              Attention mask (must match the score dimensions).
    /// @param mask_value        Value to write at masked positions.
    /// @returns Modified score vector (same size as input).
    static std::vector<float> apply_mask(
        const std::vector<float>& attention_scores,
        const AttentionMask& mask,
        float mask_value = -1e9f);

    /// Produce a human-readable ASCII visualisation of the mask.
    ///
    /// '1' denotes an attended position, '0' a masked one.
    static std::string visualize(const AttentionMask& mask);
};

} // namespace llmquant

#endif // LLMQUANT_ATTENTION_MASK_ENABLED
