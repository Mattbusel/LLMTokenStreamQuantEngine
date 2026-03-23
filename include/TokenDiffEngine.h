#pragma once
// TokenDiffEngine.h — LCS-based diff between token sequences.

#include <cstddef>
#include <string>
#include <vector>

namespace llmquant {

/// Operation type for a token diff entry.
enum class DiffOp { ADD, DELETE, KEEP };

/// A single diff operation on a token at a given position in the original sequence.
struct TokenDiff {
    DiffOp      op;
    std::string token;
    std::size_t position;  ///< 0-based index in seq_a (ADD: position in seq_b).
};

/// Computes diffs between token sequences using Wagner–Fischer DP (LCS / Levenshtein).
class TokenDiffEngine {
public:
    TokenDiffEngine() = default;

    /// Compute the edit script (LCS-based diff) transforming seq_a into seq_b.
    [[nodiscard]]
    std::vector<TokenDiff> diff(
        const std::vector<std::string>& seq_a,
        const std::vector<std::string>& seq_b) const;

    /// Levenshtein edit distance on token sequences.
    [[nodiscard]]
    std::size_t edit_distance(
        const std::vector<std::string>& seq_a,
        const std::vector<std::string>& seq_b) const;

    /// Similarity in [0, 1]: 1 - edit_distance / max(|a|, |b|).
    /// Returns 1.0 when both sequences are empty.
    [[nodiscard]]
    double similarity(
        const std::vector<std::string>& seq_a,
        const std::vector<std::string>& seq_b) const;

    /// Reconstruct the target sequence by applying diffs to base.
    [[nodiscard]]
    std::vector<std::string> apply_diff(
        const std::vector<std::string>& base,
        const std::vector<TokenDiff>&   diffs) const;
};

}  // namespace llmquant
