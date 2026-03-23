#pragma once
// StreamCompressor.h — Run-length encoding + dictionary compression for token streams.

#include <cstddef>
#include <string>
#include <utility>
#include <vector>

namespace llmquant {

/// A run-length encoded token entry.
struct CompressedToken {
    bool        is_run;  ///< True when this entry encodes a repeated run.
    std::string token;
    std::size_t count;   ///< Number of consecutive repetitions (1 for non-runs).
};

/// Run-length encoding and dictionary (index-based) compression for token streams.
class StreamCompressor {
public:
    StreamCompressor() = default;

    // -----------------------------------------------------------------------
    // Run-length encoding
    // -----------------------------------------------------------------------

    /// RLE compress: collapse consecutive identical tokens into a single entry.
    [[nodiscard]]
    std::vector<CompressedToken> rle_compress(
        const std::vector<std::string>& tokens) const;

    /// RLE decompress: expand compressed entries back to the original sequence.
    [[nodiscard]]
    std::vector<std::string> rle_decompress(
        const std::vector<CompressedToken>& compressed) const;

    /// Compression ratio: original token count / compressed entry count.
    /// Returns 1.0 when compressed is empty.
    [[nodiscard]]
    double compression_ratio(
        const std::vector<std::string>&     original,
        const std::vector<CompressedToken>& compressed) const;

    // -----------------------------------------------------------------------
    // Dictionary compression
    // -----------------------------------------------------------------------

    /// Dictionary compress: map each token to a vocabulary index.
    ///
    /// @return pair of (index sequence, vocabulary in insertion order).
    [[nodiscard]]
    std::pair<std::vector<std::size_t>, std::vector<std::string>>
    dict_compress(const std::vector<std::string>& tokens) const;

    /// Dictionary decompress: reconstruct token sequence from indices + vocabulary.
    [[nodiscard]]
    std::vector<std::string> dict_decompress(
        const std::vector<std::size_t>&  indices,
        const std::vector<std::string>&  dictionary) const;
};

}  // namespace llmquant
