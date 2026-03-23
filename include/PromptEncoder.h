#pragma once

/// @file PromptEncoder.h
/// @brief Prompt tokenization and encoding utilities.

#include <string>
#include <unordered_map>
#include <vector>

namespace llmquant {

/// An encoded prompt ready for model input.
struct EncodedPrompt {
    std::vector<int> token_ids;
    std::vector<int> attention_mask;
    std::vector<int> position_ids;
    size_t length{0};
};

/// Tokenizes and encodes text prompts using a fixed vocabulary.
class PromptEncoder {
public:
    static constexpr int UNK_ID = 0;

    /// @param vocab  Mapping from token string to integer id.
    explicit PromptEncoder(const std::unordered_map<std::string, int>& vocab);

    /// Split text into tokens on whitespace and punctuation.
    std::vector<std::string> split_bpe(const std::string& text) const;

    /// Look up a token; returns UNK_ID (0) if not in vocabulary.
    int token_to_id(const std::string& token) const;

    /// Encode a prompt: tokenize, truncate to max_length, build masks.
    EncodedPrompt encode(const std::string& prompt, int max_length = 512) const;

    /// Decode token ids back to a space-joined string.
    std::string decode(const std::vector<int>& token_ids) const;

    /// Pad an encoded prompt to exactly @p length with @p pad_id.
    EncodedPrompt pad_to_length(
        const EncodedPrompt& encoded,
        int length,
        int pad_id = 0) const;

    /// Truncate an encoded prompt to at most @p max_length tokens.
    EncodedPrompt truncate(const EncodedPrompt& encoded, int max_length) const;

    /// Encode a batch of prompts, padding each to max_length.
    std::vector<EncodedPrompt> batch_encode(
        const std::vector<std::string>& prompts,
        int max_length = 512) const;

    /// Build a vocabulary from a corpus of texts.
    ///
    /// Tokens are assigned ids in order of decreasing frequency; id 0 is
    /// reserved for UNK.
    static std::unordered_map<std::string, int> build_vocab(
        const std::vector<std::string>& texts,
        int max_vocab_size = 10000);

private:
    std::unordered_map<std::string, int>  vocab_;
    std::unordered_map<int, std::string>  id_to_token_;
};

} // namespace llmquant
