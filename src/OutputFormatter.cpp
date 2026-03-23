#include "OutputFormatter.h"

#include <algorithm>
#include <cctype>
#include <sstream>
#include <string>

namespace llmquant {

OutputFormatter::OutputFormatter(OutputFormat format) : format_(format) {}

void OutputFormatter::set_format(OutputFormat fmt) { format_ = fmt; }

// ---------------------------------------------------------------------------
// Detokenisation
// ---------------------------------------------------------------------------

std::string OutputFormatter::tokens_to_text(
    const std::vector<int>& ids,
    const std::unordered_map<int, std::string>& vocab) const
{
    std::string result;
    for (int id : ids) {
        auto it = vocab.find(id);
        if (it == vocab.end()) continue;
        const std::string& tok = it->second;
        // BPE tokens beginning with Ġ (U+0120) represent a leading space.
        // We normalise that by adding a space when needed.
        if (!result.empty() && !tok.empty() && tok[0] != ' ') {
            result += ' ';
        }
        result += tok;
    }
    return result;
}

// ---------------------------------------------------------------------------
// Format dispatch
// ---------------------------------------------------------------------------

std::string OutputFormatter::format(
    const std::vector<int>& token_ids,
    const std::unordered_map<int, std::string>& vocab) const
{
    const std::string text = tokens_to_text(token_ids, vocab);
    switch (format_) {
        case OutputFormat::PlainText: return text;
        case OutputFormat::Markdown:  return to_markdown(text);
        case OutputFormat::JSON:      return to_json(text);
        case OutputFormat::HTML:      return to_html(text);
        case OutputFormat::CodeBlock: return to_code_block(text, "");
    }
    return text;
}

// ---------------------------------------------------------------------------
// Markdown rendering
// ---------------------------------------------------------------------------

std::string OutputFormatter::to_markdown(const std::string& text) const {
    // Bold ALL-CAPS words and handle ``` code fences.
    std::istringstream iss(text);
    std::ostringstream oss;
    std::string word;
    bool first = true;

    // Tokenise by whitespace
    while (iss >> word) {
        if (!first) oss << ' ';
        first = false;

        // Check if all alphabetic characters are uppercase
        bool all_caps = !word.empty();
        for (char c : word) {
            if (std::isalpha(static_cast<unsigned char>(c)) &&
                !std::isupper(static_cast<unsigned char>(c))) {
                all_caps = false;
                break;
            }
        }
        if (all_caps && word.size() > 1 &&
            std::isalpha(static_cast<unsigned char>(word[0]))) {
            oss << "**" << word << "**";
        } else {
            oss << word;
        }
    }

    return oss.str();
}

// ---------------------------------------------------------------------------
// HTML rendering
// ---------------------------------------------------------------------------

std::string OutputFormatter::escape_html(const std::string& s) {
    std::string out;
    out.reserve(s.size());
    for (char c : s) {
        switch (c) {
            case '&':  out += "&amp;";  break;
            case '<':  out += "&lt;";   break;
            case '>':  out += "&gt;";   break;
            case '"':  out += "&quot;"; break;
            default:   out += c;        break;
        }
    }
    return out;
}

std::string OutputFormatter::to_html(const std::string& text) const {
    // Split on blank lines to form paragraphs; detect inline `` code ``.
    std::istringstream iss(text);
    std::ostringstream oss;
    std::string line;
    std::string para;

    auto flush_para = [&]() {
        if (!para.empty()) {
            oss << "<p>" << escape_html(para) << "</p>\n";
            para.clear();
        }
    };

    while (std::getline(iss, line)) {
        if (line.empty()) {
            flush_para();
        } else {
            if (!para.empty()) para += ' ';
            para += line;
        }
    }
    flush_para();

    return oss.str();
}

// ---------------------------------------------------------------------------
// JSON rendering
// ---------------------------------------------------------------------------

std::string OutputFormatter::escape_json(const std::string& s) {
    std::string out;
    out.reserve(s.size());
    for (unsigned char c : s) {
        switch (c) {
            case '"':  out += "\\\""; break;
            case '\\': out += "\\\\"; break;
            case '\n': out += "\\n";  break;
            case '\r': out += "\\r";  break;
            case '\t': out += "\\t";  break;
            default:
                if (c < 0x20) {
                    char buf[8];
                    std::snprintf(buf, sizeof(buf), "\\u%04x", c);
                    out += buf;
                } else {
                    out += static_cast<char>(c);
                }
        }
    }
    return out;
}

std::string OutputFormatter::to_json(const std::string& text) const {
    return "{\"text\": \"" + escape_json(text) + "\"}";
}

// ---------------------------------------------------------------------------
// Code block rendering
// ---------------------------------------------------------------------------

std::string OutputFormatter::to_code_block(
    const std::string& text, const std::string& lang) const
{
    return "```" + lang + "\n" + text + "\n```";
}

}  // namespace llmquant
