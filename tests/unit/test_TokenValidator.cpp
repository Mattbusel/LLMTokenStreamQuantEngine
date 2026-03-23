// test_TokenValidator.cpp — GTest unit tests for TokenValidator.

#include "TokenValidator.h"
#include <gtest/gtest.h>
#include <string>
#include <vector>

using namespace llmquant;

// -------------------------------------------------------------------------
// Tests
// -------------------------------------------------------------------------

TEST(TokenValidator, ValidSimpleToken) {
    TokenValidator tv;
    auto result = tv.validate("hello");
    EXPECT_TRUE(result.is_valid);
    EXPECT_TRUE(result.error_message.empty());
}

TEST(TokenValidator, EmptyTokenIsInvalid) {
    TokenValidator tv;
    auto result = tv.validate("");
    EXPECT_FALSE(result.is_valid);
    EXPECT_FALSE(result.error_message.empty());
}

TEST(TokenValidator, TokenExceedingMaxLengthIsInvalid) {
    TokenValidator tv;
    tv.set_max_length(5);
    auto result = tv.validate("toolong");
    EXPECT_FALSE(result.is_valid);
}

TEST(TokenValidator, TokenAtExactMaxLengthIsValid) {
    TokenValidator tv;
    tv.set_max_length(5);
    auto result = tv.validate("hello");
    EXPECT_TRUE(result.is_valid);
}

TEST(TokenValidator, NullByteIsInvalid) {
    TokenValidator tv;
    std::string bad_token = "ab";
    bad_token += '\x00';
    bad_token += "cd";
    auto result = tv.validate(bad_token);
    EXPECT_FALSE(result.is_valid);
}

TEST(TokenValidator, SanitizeReplacesInvalidChars) {
    TokenValidator tv;
    std::string bad = "ab";
    bad += '\x01';
    bad += "cd";
    std::string sanitized = tv.sanitize(bad);
    EXPECT_EQ(sanitized.find('?'), std::size_t(2));
}

TEST(TokenValidator, SanitizeTrimWhitespace) {
    TokenValidator tv;
    std::string result = tv.sanitize("  hello  ");
    EXPECT_EQ(result, "hello");
}

TEST(TokenValidator, SanitizeEmptyStringReturnsEmpty) {
    TokenValidator tv;
    EXPECT_EQ(tv.sanitize(""), "");
}

TEST(TokenValidator, ValidateSequenceLength) {
    TokenValidator tv;
    std::vector<std::string> tokens{"hello", "", "world"};
    auto results = tv.validate_sequence(tokens);
    EXPECT_EQ(results.size(), 3u);
    EXPECT_TRUE(results[0].is_valid);
    EXPECT_FALSE(results[1].is_valid);
    EXPECT_TRUE(results[2].is_valid);
}

TEST(TokenValidator, FilterValidKeepsOnlyValidTokens) {
    TokenValidator tv;
    std::vector<std::string> tokens{"good", "", "also_good"};
    auto valid = tv.filter_valid(tokens);
    EXPECT_EQ(valid.size(), 2u);
    EXPECT_EQ(valid[0], "good");
    EXPECT_EQ(valid[1], "also_good");
}

TEST(TokenValidator, SetAllowedCharsRestrictsValidation) {
    TokenValidator tv;
    tv.set_allowed_chars("abc");
    EXPECT_TRUE(tv.validate("abc").is_valid);
    EXPECT_FALSE(tv.validate("abcd").is_valid);  // 'd' not allowed
}

TEST(TokenValidator, SetMaxLengthUpdatesLimit) {
    TokenValidator tv;
    tv.set_max_length(3);
    EXPECT_FALSE(tv.validate("four").is_valid);
    EXPECT_TRUE(tv.validate("tri").is_valid);
}

TEST(TokenValidator, SanitizedTokenStoredInResult) {
    TokenValidator tv;
    std::string bad = " \x01hello ";
    auto result = tv.validate(bad);
    // sanitized_token should be produced regardless of validity.
    EXPECT_FALSE(result.sanitized_token.empty());
}
