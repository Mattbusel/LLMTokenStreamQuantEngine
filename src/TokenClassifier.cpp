#include "TokenClassifier.h"
#include <algorithm>
#include <cctype>

namespace llmquant {

TokenClassifier::TokenClassifier() {}

bool TokenClassifier::is_word_char(char c) const {
    return std::isalpha(static_cast<unsigned char>(c)) || c == '\'';
}

bool TokenClassifier::is_digit(char c) const {
    return std::isdigit(static_cast<unsigned char>(c));
}

bool TokenClassifier::is_punctuation(char c) const {
    static const std::string puncts = ".,!?;:\"'()[]{}";
    return puncts.find(c) != std::string::npos;
}

bool TokenClassifier::is_code_symbol(char c) const {
    static const std::string syms = "+-*/=<>|&^~#@\\%`";
    return syms.find(c) != std::string::npos;
}

bool TokenClassifier::is_special_token(const std::string& text) const {
    return !text.empty() && text.front() == '<' && text.back() == '>';
}

TokenType TokenClassifier::classify(const std::string& token_text) const {
    if (token_text.empty()) return TokenType::Unknown;

    // Check special tokens first
    if (is_special_token(token_text)) return TokenType::SpecialToken;

    // Strip subword prefix (##, Ġ, Ċ etc.)
    std::string text = token_text;
    size_t start = 0;
    if (text.size() >= 2 && text[0] == '#' && text[1] == '#') start = 2;
    else if (!text.empty() && (unsigned char)text[0] >= 0xC0) start = 2; // UTF-8 prefix
    if (start >= text.size()) return TokenType::Whitespace;
    text = text.substr(start);
    if (text.empty()) return TokenType::Whitespace;

    // URL detection
    if (text.find("http") == 0 || text.find("https") == 0) return TokenType::Url;

    // Newline
    if (text == "\n" || text == "\\n" || text == "\r\n") return TokenType::Newline;

    // Whitespace
    bool all_space = std::all_of(text.begin(), text.end(), [](char c) {
        return c == ' ' || c == '\t' || c == '\r' || c == '\n';
    });
    if (all_space) return TokenType::Whitespace;

    // Number
    bool has_digit = false, all_numeric = true;
    for (char c : text) {
        if (is_digit(c) || c == '.' || c == '-' || c == '+' || c == 'e' || c == 'E') {
            if (is_digit(c)) has_digit = true;
        } else { all_numeric = false; }
    }
    if (all_numeric && has_digit) return TokenType::Number;

    // Word
    bool all_word = std::all_of(text.begin(), text.end(), [this](char c) {
        return is_word_char(c) || is_digit(c) || c == '-';
    });
    if (all_word && std::isalpha((unsigned char)text[0])) return TokenType::Word;

    // Code symbol
    bool has_code = std::any_of(text.begin(), text.end(), [this](char c) { return is_code_symbol(c); });
    bool all_code = std::all_of(text.begin(), text.end(), [this](char c) {
        return is_code_symbol(c) || is_digit(c) || c == '_';
    });
    if (all_code && has_code) return TokenType::CodeSymbol;

    // Punctuation
    if (text.size() == 1 && is_punctuation(text[0])) return TokenType::Punctuation;

    return TokenType::Unknown;
}

TokenType TokenClassifier::classify_by_id(uint32_t token_id) const {
    auto it = token_cache_.find(token_id);
    if (it == token_cache_.end()) return TokenType::Unknown;
    return it->second.type;
}

void TokenClassifier::load_vocabulary(const std::vector<std::pair<uint32_t, std::string>>& vocab) {
    token_cache_.clear();
    token_cache_.reserve(vocab.size());
    for (const auto& [id, text] : vocab) {
        bool is_sub = !text.empty() && ((text.size() >= 2 && text[0]=='#' && text[1]=='#') ||
                                         ((unsigned char)text[0] >= 0xC0));
        float log_freq = 0.0f; // unknown
        TokenType type = classify(text);
        token_cache_[id] = TokenInfo{id, text, type, is_sub, log_freq};
    }
}

std::vector<TokenType> TokenClassifier::classify_sequence(const std::vector<uint32_t>& token_ids) const {
    std::vector<TokenType> result;
    result.reserve(token_ids.size());
    for (uint32_t id : token_ids) result.push_back(classify_by_id(id));
    return result;
}

std::unordered_map<TokenType, size_t> TokenClassifier::type_histogram(
    const std::vector<uint32_t>& token_ids) const {
    std::unordered_map<TokenType, size_t> hist;
    for (uint32_t id : token_ids) hist[classify_by_id(id)]++;
    return hist;
}

bool TokenClassifier::is_likely_code(const std::vector<uint32_t>& token_ids, float threshold) const {
    if (token_ids.empty()) return false;
    size_t code_count = 0;
    for (uint32_t id : token_ids) {
        if (classify_by_id(id) == TokenType::CodeSymbol) code_count++;
    }
    return static_cast<float>(code_count) / token_ids.size() >= threshold;
}

bool TokenClassifier::is_numeric_heavy(const std::vector<uint32_t>& token_ids, float threshold) const {
    if (token_ids.empty()) return false;
    size_t num_count = 0;
    for (uint32_t id : token_ids) {
        if (classify_by_id(id) == TokenType::Number) num_count++;
    }
    return static_cast<float>(num_count) / token_ids.size() >= threshold;
}

TokenClassifier::VocabStats TokenClassifier::vocabulary_stats() const {
    VocabStats stats;
    stats.total_tokens = token_cache_.size();
    stats.special_token_count = 0;
    for (const auto& [id, info] : token_cache_) {
        stats.type_counts[info.type]++;
        if (info.type == TokenType::SpecialToken) stats.special_token_count++;
    }
    return stats;
}

} // namespace llmquant
