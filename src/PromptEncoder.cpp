#include "PromptEncoder.h"

#include <algorithm>
#include <cctype>
#include <sstream>
#include <stdexcept>
#include <unordered_map>

namespace llmquant {

PromptEncoder::PromptEncoder(const std::unordered_map<std::string, int>& vocab)
    : vocab_(vocab)
{
    for (const auto& [token, id] : vocab_) {
        id_to_token_[id] = token;
    }
}

std::vector<std::string> PromptEncoder::split_bpe(const std::string& text) const {
    std::vector<std::string> tokens;
    std::string current;
    for (char ch : text) {
        if (std::isspace(static_cast<unsigned char>(ch))) {
            if (!current.empty()) {
                tokens.push_back(current);
                current.clear();
            }
        } else if (std::ispunct(static_cast<unsigned char>(ch))) {
            if (!current.empty()) {
                tokens.push_back(current);
                current.clear();
            }
            tokens.push_back(std::string(1, ch));
        } else {
            current += std::tolower(static_cast<unsigned char>(ch));
        }
    }
    if (!current.empty()) {
        tokens.push_back(current);
    }
    return tokens;
}

int PromptEncoder::token_to_id(const std::string& token) const {
    auto it = vocab_.find(token);
    return (it != vocab_.end()) ? it->second : UNK_ID;
}

EncodedPrompt PromptEncoder::encode(const std::string& prompt, int max_length) const {
    auto tokens = split_bpe(prompt);
    if (max_length > 0 && static_cast<int>(tokens.size()) > max_length) {
        tokens.resize(static_cast<size_t>(max_length));
    }
    EncodedPrompt ep;
    ep.length = tokens.size();
    ep.token_ids.reserve(ep.length);
    ep.attention_mask.reserve(ep.length);
    ep.position_ids.reserve(ep.length);
    for (size_t i = 0; i < ep.length; ++i) {
        ep.token_ids.push_back(token_to_id(tokens[i]));
        ep.attention_mask.push_back(1);
        ep.position_ids.push_back(static_cast<int>(i));
    }
    return ep;
}

std::string PromptEncoder::decode(const std::vector<int>& token_ids) const {
    std::string result;
    for (size_t i = 0; i < token_ids.size(); ++i) {
        auto it = id_to_token_.find(token_ids[i]);
        if (it != id_to_token_.end()) {
            if (i > 0) result += ' ';
            result += it->second;
        }
    }
    return result;
}

EncodedPrompt PromptEncoder::pad_to_length(
    const EncodedPrompt& encoded,
    int length,
    int pad_id) const
{
    EncodedPrompt result = encoded;
    while (static_cast<int>(result.token_ids.size()) < length) {
        result.token_ids.push_back(pad_id);
        result.attention_mask.push_back(0);
        result.position_ids.push_back(static_cast<int>(result.token_ids.size()) - 1);
    }
    result.length = result.token_ids.size();
    return result;
}

EncodedPrompt PromptEncoder::truncate(const EncodedPrompt& encoded, int max_length) const {
    if (max_length <= 0 || static_cast<int>(encoded.token_ids.size()) <= max_length) {
        return encoded;
    }
    EncodedPrompt result;
    result.token_ids.assign(
        encoded.token_ids.begin(),
        encoded.token_ids.begin() + max_length);
    result.attention_mask.assign(
        encoded.attention_mask.begin(),
        encoded.attention_mask.begin() + max_length);
    result.position_ids.assign(
        encoded.position_ids.begin(),
        encoded.position_ids.begin() + max_length);
    result.length = static_cast<size_t>(max_length);
    return result;
}

std::vector<EncodedPrompt> PromptEncoder::batch_encode(
    const std::vector<std::string>& prompts,
    int max_length) const
{
    std::vector<EncodedPrompt> results;
    results.reserve(prompts.size());
    for (const auto& prompt : prompts) {
        auto ep = encode(prompt, max_length);
        results.push_back(pad_to_length(ep, max_length));
    }
    return results;
}

std::unordered_map<std::string, int> PromptEncoder::build_vocab(
    const std::vector<std::string>& texts,
    int max_vocab_size)
{
    // Count token frequencies.
    std::unordered_map<std::string, int> freq;
    for (const auto& text : texts) {
        // Re-use split logic inline.
        std::string current;
        for (char ch : text) {
            if (std::isspace(static_cast<unsigned char>(ch))) {
                if (!current.empty()) {
                    freq[current]++;
                    current.clear();
                }
            } else if (std::ispunct(static_cast<unsigned char>(ch))) {
                if (!current.empty()) {
                    freq[current]++;
                    current.clear();
                }
                freq[std::string(1, ch)]++;
            } else {
                current += std::tolower(static_cast<unsigned char>(ch));
            }
        }
        if (!current.empty()) {
            freq[current]++;
        }
    }

    // Sort by descending frequency.
    std::vector<std::pair<std::string, int>> sorted_tokens(freq.begin(), freq.end());
    std::sort(sorted_tokens.begin(), sorted_tokens.end(),
              [](const auto& a, const auto& b) { return a.second > b.second; });

    // Build vocab; id 0 is reserved for UNK.
    std::unordered_map<std::string, int> vocab;
    int id = 1;
    for (const auto& [token, count] : sorted_tokens) {
        if (id >= max_vocab_size) break;
        vocab[token] = id++;
    }
    return vocab;
}

} // namespace llmquant
