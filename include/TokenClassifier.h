#pragma once
#include <string>
#include <vector>
#include <unordered_map>
#include <cstdint>

namespace llmquant {

/// Token type classifications
enum class TokenType : uint8_t {
    Unknown = 0,
    Word,           // alphabetic word
    Number,         // integer or float
    Punctuation,    // . , ! ? ; : etc.
    Whitespace,     // space, tab, newline
    SpecialToken,   // <pad>, <eos>, <bos>, etc.
    CodeSymbol,     // programming operators: + - * / = { } etc.
    Emoji,          // unicode emoji range
    Url,            // http:// or https:// prefix
    Newline,        // \n specifically
};

struct TokenInfo {
    uint32_t token_id;
    std::string text;
    TokenType type;
    bool is_subword;   // starts with ## or Ġ (byte-pair prefix)
    float log_frequency;  // log of training frequency (if known)
};

/// Classify tokens by their surface form
class TokenClassifier {
public:
    explicit TokenClassifier();
    ~TokenClassifier() = default;

    /// Classify a single token by its text
    TokenType classify(const std::string& token_text) const;

    /// Classify token by ID (requires pre-loaded vocabulary)
    TokenType classify_by_id(uint32_t token_id) const;

    /// Build classification cache from vocabulary
    void load_vocabulary(const std::vector<std::pair<uint32_t, std::string>>& vocab);

    /// Classify a sequence of tokens
    std::vector<TokenType> classify_sequence(const std::vector<uint32_t>& token_ids) const;

    /// Count tokens by type in a sequence
    std::unordered_map<TokenType, size_t> type_histogram(const std::vector<uint32_t>& token_ids) const;

    /// Check if a token sequence looks like code (high CodeSymbol ratio)
    bool is_likely_code(const std::vector<uint32_t>& token_ids, float threshold = 0.15f) const;

    /// Check if sequence is numeric-heavy
    bool is_numeric_heavy(const std::vector<uint32_t>& token_ids, float threshold = 0.3f) const;

    /// Statistics about loaded vocabulary
    struct VocabStats {
        size_t total_tokens;
        std::unordered_map<TokenType, size_t> type_counts;
        size_t special_token_count;
    };
    VocabStats vocabulary_stats() const;

private:
    std::unordered_map<uint32_t, TokenInfo> token_cache_;

    bool is_word_char(char c) const;
    bool is_digit(char c) const;
    bool is_punctuation(char c) const;
    bool is_code_symbol(char c) const;
    bool is_special_token(const std::string& text) const;
};

} // namespace llmquant
