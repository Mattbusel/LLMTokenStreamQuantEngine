// TokenValidator.cpp — implementation of token validation and sanitization.

#include "TokenValidator.h"

#include <algorithm>
#include <cctype>
#include <sstream>

namespace llmquant {

TokenValidator::TokenValidator(std::size_t max_length)
    : max_length_(max_length), allowed_chars_() {}

bool TokenValidator::is_allowed_char(unsigned char c) const noexcept {
    if (!allowed_chars_.empty()) {
        return allowed_chars_.find(static_cast<char>(c)) != std::string::npos;
    }
    // Default: printable ASCII (0x20–0x7E) plus UTF-8 continuation bytes
    // (0x80–0xBF) and leading bytes (0xC0–0xFF).
    return (c >= 0x20 && c <= 0x7E) || c >= 0x80;
}

std::string TokenValidator::trim_whitespace(const std::string& s) const {
    const auto not_space = [](unsigned char c) { return !std::isspace(c); };
    auto first = std::find_if(s.begin(), s.end(), not_space);
    if (first == s.end()) return {};
    auto last = std::find_if(s.rbegin(), s.rend(), not_space).base();
    return std::string(first, last);
}

ValidationResult TokenValidator::validate(const std::string& token) const {
    ValidationResult result;
    result.sanitized_token = sanitize(token);

    if (token.empty()) {
        result.is_valid = false;
        result.error_message = "token is empty";
        return result;
    }

    if (token.size() > max_length_) {
        result.is_valid = false;
        result.error_message = "token length " + std::to_string(token.size()) +
                               " exceeds maximum " + std::to_string(max_length_);
        return result;
    }

    for (std::size_t i = 0; i < token.size(); ++i) {
        auto c = static_cast<unsigned char>(token[i]);
        if (!is_allowed_char(c)) {
            result.is_valid = false;
            result.error_message = "invalid character at position " + std::to_string(i) +
                                   " (0x" + [&] {
                                       std::ostringstream ss;
                                       ss << std::hex << static_cast<unsigned>(c);
                                       return ss.str();
                                   }() + ")";
            return result;
        }
    }

    result.is_valid = true;
    return result;
}

std::string TokenValidator::sanitize(const std::string& token) const {
    std::string out;
    out.reserve(token.size());
    for (unsigned char c : token) {
        out += is_allowed_char(c) ? static_cast<char>(c) : '?';
    }
    return trim_whitespace(out);
}

std::vector<ValidationResult> TokenValidator::validate_sequence(
    const std::vector<std::string>& tokens) const {
    std::vector<ValidationResult> results;
    results.reserve(tokens.size());
    for (const auto& t : tokens) {
        results.push_back(validate(t));
    }
    return results;
}

std::vector<std::string> TokenValidator::filter_valid(
    const std::vector<std::string>& tokens) const {
    std::vector<std::string> valid;
    for (const auto& t : tokens) {
        if (validate(t).is_valid) {
            valid.push_back(t);
        }
    }
    return valid;
}

void TokenValidator::set_max_length(std::size_t n) {
    max_length_ = n;
}

void TokenValidator::set_allowed_chars(const std::string& chars) {
    allowed_chars_ = chars;
}

}  // namespace llmquant
