#pragma once

#include <string>
#include <unordered_map>
#include <vector>

namespace llmquant {

/// Output format options for token sequences.
enum class OutputFormat {
    PlainText,
    Markdown,
    JSON,
    HTML,
    CodeBlock,
};

/**
 * @brief Formats token id sequences into human-readable output strings.
 *
 * Supports plain text, Markdown, JSON, HTML, and code block formats.
 * Uses a vocab map (token_id -> string) for detokenisation.
 */
class OutputFormatter {
public:
    explicit OutputFormatter(OutputFormat format = OutputFormat::PlainText);

    /// Format a sequence of token ids using the given vocabulary.
    std::string format(
        const std::vector<int>& token_ids,
        const std::unordered_map<int, std::string>& vocab) const;

    /// Join token strings, handling leading-space BPE tokens correctly.
    std::string tokens_to_text(
        const std::vector<int>& ids,
        const std::unordered_map<int, std::string>& vocab) const;

    /// Wrap ``` delimited blocks and bold ALL-CAPS words.
    std::string to_markdown(const std::string& text) const;

    /// Escape HTML special chars, wrap paragraphs in <p>, inline code in <code>.
    std::string to_html(const std::string& text) const;

    /// Produce {"text": "<escaped>"} JSON object.
    std::string to_json(const std::string& text) const;

    /// Wrap text in a fenced code block with an optional language tag.
    std::string to_code_block(const std::string& text, const std::string& lang) const;

    void set_format(OutputFormat fmt);

    static std::string escape_json(const std::string& s);
    static std::string escape_html(const std::string& s);

private:
    OutputFormat format_;
};

}  // namespace llmquant
