#include <gtest/gtest.h>
#include "OutputFormatter.h"

#include <unordered_map>
#include <vector>

using namespace llmquant;

// ---------------------------------------------------------------------------
// Helper vocabulary
// ---------------------------------------------------------------------------
static std::unordered_map<int, std::string> make_vocab() {
    return {{1, "Hello"}, {2, "WORLD"}, {3, "foo"}, {4, "bar"}, {5, "baz"}};
}

// ---------------------------------------------------------------------------
// Construction
// ---------------------------------------------------------------------------

TEST(OutputFormatter, DefaultConstructsPlainText) {
    OutputFormatter fmt;
    auto vocab = make_vocab();
    std::string result = fmt.format({1, 2}, vocab);
    EXPECT_FALSE(result.empty());
}

TEST(OutputFormatter, SetFormatChangesOutput) {
    OutputFormatter fmt(OutputFormat::PlainText);
    fmt.set_format(OutputFormat::JSON);
    auto vocab = make_vocab();
    std::string result = fmt.format({1}, vocab);
    EXPECT_NE(result.find("{\"text\""), std::string::npos);
}

// ---------------------------------------------------------------------------
// tokens_to_text
// ---------------------------------------------------------------------------

TEST(OutputFormatter, TokensToTextJoinsWithSpaces) {
    OutputFormatter fmt;
    auto vocab = make_vocab();
    std::string result = fmt.tokens_to_text({3, 4, 5}, vocab);
    EXPECT_NE(result.find("foo"), std::string::npos);
    EXPECT_NE(result.find("bar"), std::string::npos);
}

TEST(OutputFormatter, TokensToTextUnknownIdSkipped) {
    OutputFormatter fmt;
    auto vocab = make_vocab();
    // id 99 not in vocab
    std::string result = fmt.tokens_to_text({1, 99, 3}, vocab);
    EXPECT_NE(result.find("Hello"), std::string::npos);
    EXPECT_NE(result.find("foo"), std::string::npos);
}

TEST(OutputFormatter, TokensToTextEmptyIds) {
    OutputFormatter fmt;
    auto vocab = make_vocab();
    EXPECT_EQ(fmt.tokens_to_text({}, vocab), "");
}

// ---------------------------------------------------------------------------
// Markdown
// ---------------------------------------------------------------------------

TEST(OutputFormatter, ToMarkdownBoldsAllCapsWord) {
    OutputFormatter fmt(OutputFormat::Markdown);
    auto vocab = make_vocab();
    std::string result = fmt.format({1, 2}, vocab);
    // "WORLD" should be bolded
    EXPECT_NE(result.find("**WORLD**"), std::string::npos);
}

TEST(OutputFormatter, ToMarkdownDoesNotBoldMixedCase) {
    OutputFormatter fmt(OutputFormat::Markdown);
    std::string result = fmt.to_markdown("Hello World");
    EXPECT_EQ(result.find("**Hello**"), std::string::npos);
}

// ---------------------------------------------------------------------------
// HTML
// ---------------------------------------------------------------------------

TEST(OutputFormatter, ToHtmlEscapesAmpersand) {
    OutputFormatter fmt;
    std::string result = fmt.escape_html("a & b");
    EXPECT_NE(result.find("&amp;"), std::string::npos);
}

TEST(OutputFormatter, ToHtmlEscapesAngleBrackets) {
    OutputFormatter fmt;
    std::string escaped = fmt.escape_html("<tag>");
    EXPECT_NE(escaped.find("&lt;"), std::string::npos);
    EXPECT_NE(escaped.find("&gt;"), std::string::npos);
}

TEST(OutputFormatter, ToHtmlWrapsInParagraph) {
    OutputFormatter fmt(OutputFormat::HTML);
    auto vocab = make_vocab();
    std::string result = fmt.format({1, 3}, vocab);
    EXPECT_NE(result.find("<p>"), std::string::npos);
    EXPECT_NE(result.find("</p>"), std::string::npos);
}

// ---------------------------------------------------------------------------
// JSON
// ---------------------------------------------------------------------------

TEST(OutputFormatter, ToJsonProducesValidStructure) {
    OutputFormatter fmt(OutputFormat::JSON);
    auto vocab = make_vocab();
    std::string result = fmt.format({1}, vocab);
    EXPECT_EQ(result.substr(0, 9), "{\"text\": ");
    EXPECT_EQ(result.back(), '}');
}

TEST(OutputFormatter, EscapeJsonEscapesQuote) {
    std::string escaped = OutputFormatter::escape_json("say \"hi\"");
    EXPECT_NE(escaped.find("\\\""), std::string::npos);
}

TEST(OutputFormatter, EscapeJsonEscapesNewline) {
    std::string escaped = OutputFormatter::escape_json("line1\nline2");
    EXPECT_NE(escaped.find("\\n"), std::string::npos);
}

// ---------------------------------------------------------------------------
// Code block
// ---------------------------------------------------------------------------

TEST(OutputFormatter, ToCodeBlockWrapsWithFences) {
    OutputFormatter fmt;
    std::string result = fmt.to_code_block("int x = 0;", "cpp");
    EXPECT_EQ(result.substr(0, 6), "```cpp");
    EXPECT_NE(result.rfind("```"), 0u);  // closing fence exists
}

TEST(OutputFormatter, ToCodeBlockEmptyLang) {
    OutputFormatter fmt;
    std::string result = fmt.to_code_block("code", "");
    EXPECT_EQ(result.substr(0, 3), "```");
}
