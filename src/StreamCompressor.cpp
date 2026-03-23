// StreamCompressor.cpp — RLE + dictionary compression for token streams.
#include "StreamCompressor.h"

#include <stdexcept>
#include <unordered_map>

namespace llmquant {

// ---------------------------------------------------------------------------
// RLE compress
// ---------------------------------------------------------------------------

std::vector<CompressedToken> StreamCompressor::rle_compress(
    const std::vector<std::string>& tokens) const
{
    std::vector<CompressedToken> result;
    if (tokens.empty()) return result;

    result.reserve(tokens.size());
    std::string current = tokens[0];
    std::size_t count   = 1;

    for (std::size_t i = 1; i < tokens.size(); ++i) {
        if (tokens[i] == current) {
            ++count;
        } else {
            result.push_back({count > 1, current, count});
            current = tokens[i];
            count   = 1;
        }
    }
    result.push_back({count > 1, current, count});
    return result;
}

// ---------------------------------------------------------------------------
// RLE decompress
// ---------------------------------------------------------------------------

std::vector<std::string> StreamCompressor::rle_decompress(
    const std::vector<CompressedToken>& compressed) const
{
    std::vector<std::string> result;
    for (const auto& ct : compressed) {
        for (std::size_t k = 0; k < ct.count; ++k) {
            result.push_back(ct.token);
        }
    }
    return result;
}

// ---------------------------------------------------------------------------
// compression_ratio
// ---------------------------------------------------------------------------

double StreamCompressor::compression_ratio(
    const std::vector<std::string>&     original,
    const std::vector<CompressedToken>& compressed) const
{
    if (compressed.empty()) return 1.0;
    return static_cast<double>(original.size()) /
           static_cast<double>(compressed.size());
}

// ---------------------------------------------------------------------------
// dict_compress
// ---------------------------------------------------------------------------

std::pair<std::vector<std::size_t>, std::vector<std::string>>
StreamCompressor::dict_compress(const std::vector<std::string>& tokens) const
{
    std::unordered_map<std::string, std::size_t> vocab_map;
    std::vector<std::string> dictionary;
    std::vector<std::size_t> indices;
    indices.reserve(tokens.size());

    for (const auto& tok : tokens) {
        auto it = vocab_map.find(tok);
        if (it == vocab_map.end()) {
            std::size_t idx = dictionary.size();
            vocab_map.emplace(tok, idx);
            dictionary.push_back(tok);
            indices.push_back(idx);
        } else {
            indices.push_back(it->second);
        }
    }
    return {std::move(indices), std::move(dictionary)};
}

// ---------------------------------------------------------------------------
// dict_decompress
// ---------------------------------------------------------------------------

std::vector<std::string> StreamCompressor::dict_decompress(
    const std::vector<std::size_t>&  indices,
    const std::vector<std::string>&  dictionary) const
{
    std::vector<std::string> result;
    result.reserve(indices.size());
    for (const auto idx : indices) {
        if (idx >= dictionary.size()) {
            throw std::out_of_range("dict_decompress: index out of range");
        }
        result.push_back(dictionary[idx]);
    }
    return result;
}

}  // namespace llmquant
