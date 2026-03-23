#pragma once
// TokenValidator.h — token validation and sanitization.

#include <cstddef>
#include <string>
#include <vector>

namespace llmquant {

/// Result of validating a single token.
struct ValidationResult {
    bool is_valid{false};
    std::string error_message;
    std::string sanitized_token;
};

/// Validates and sanitizes string tokens for safe downstream processing.
class TokenValidator {
public:
    static constexpr std::size_t kDefaultMaxLength = 512;

    explicit TokenValidator(std::size_t max_length = kDefaultMaxLength);
    ~TokenValidator() = default;

    /// Validate a single token: non-empty, valid characters, length <= max.
    ValidationResult validate(const std::string& token) const;

    /// Replace invalid characters with '?', trim leading/trailing whitespace.
    std::string sanitize(const std::string& token) const;

    /// Validate each token in *tokens*, returning one result per token.
    std::vector<ValidationResult> validate_sequence(
        const std::vector<std::string>& tokens) const;

    /// Return only the tokens that pass validation.
    std::vector<std::string> filter_valid(
        const std::vector<std::string>& tokens) const;

    /// Override the maximum token length (in bytes).
    void set_max_length(std::size_t n);

    /// Override the set of allowed characters.  Empty string = default ASCII.
    void set_allowed_chars(const std::string& chars);

private:
    std::size_t max_length_;
    std::string allowed_chars_;  // empty = use default ASCII range

    bool is_allowed_char(unsigned char c) const noexcept;
    std::string trim_whitespace(const std::string& s) const;
};

}  // namespace llmquant
