// GenerationConfig.cpp — Validation and serialisation for GenerationConfig.

#include "GenerationConfig.h"

#include <algorithm>
#include <cctype>
#include <sstream>
#include <stdexcept>
#include <string>

namespace llmquant {

// ---------------------------------------------------------------------------
// Validation
// ---------------------------------------------------------------------------

std::vector<std::string> ConfigValidator::validate(
    const GenerationConfig& cfg) const
{
    std::vector<std::string> errors;

    if (cfg.max_new_tokens == 0) {
        errors.emplace_back("max_new_tokens must be >= 1");
    }
    if (cfg.temperature <= 0.0f) {
        errors.emplace_back("temperature must be > 0.0");
    }
    if (cfg.top_p <= 0.0f || cfg.top_p > 1.0f) {
        errors.emplace_back("top_p must be in (0.0, 1.0]");
    }
    if (cfg.repetition_penalty <= 0.0f) {
        errors.emplace_back("repetition_penalty must be > 0.0");
    }
    if (cfg.num_beams == 0) {
        errors.emplace_back("num_beams must be >= 1");
    }
    if (cfg.num_beams > 1 && cfg.do_sample) {
        errors.emplace_back("do_sample must be false when num_beams > 1");
    }
    if (cfg.pad_token_id == cfg.eos_token_id) {
        errors.emplace_back("pad_token_id and eos_token_id must differ");
    }

    return errors;
}

bool ConfigValidator::is_valid(const GenerationConfig& cfg) const
{
    return validate(cfg).empty();
}

// ---------------------------------------------------------------------------
// with_defaults
// ---------------------------------------------------------------------------

GenerationConfig ConfigValidator::with_defaults()
{
    GenerationConfig cfg;
    cfg.max_new_tokens    = 256;
    cfg.temperature       = 0.7f;
    cfg.top_k             = 50;
    cfg.top_p             = 0.9f;
    cfg.repetition_penalty = 1.1f;
    cfg.do_sample         = true;
    cfg.num_beams         = 1;
    cfg.early_stopping    = false;
    cfg.pad_token_id      = 0;
    cfg.eos_token_id      = 2;
    return cfg;
}

// ---------------------------------------------------------------------------
// from_json  (hand-rolled minimal parser — no external JSON library)
// ---------------------------------------------------------------------------

static std::string trim(const std::string& s)
{
    size_t start = s.find_first_not_of(" \t\r\n");
    size_t end   = s.find_last_not_of(" \t\r\n");
    if (start == std::string::npos) return "";
    return s.substr(start, end - start + 1);
}

static std::string strip_quotes(const std::string& s)
{
    if (s.size() >= 2 && s.front() == '"' && s.back() == '"') {
        return s.substr(1, s.size() - 2);
    }
    return s;
}

GenerationConfig ConfigValidator::from_json(const std::string& json_str)
{
    GenerationConfig cfg;

    // Tokenise key:value pairs separated by commas, ignoring outer braces.
    std::string content = json_str;
    // Remove outer { }
    auto ob = content.find('{');
    auto cb = content.rfind('}');
    if (ob != std::string::npos && cb != std::string::npos && cb > ob) {
        content = content.substr(ob + 1, cb - ob - 1);
    }

    // Split by commas (naive — does not handle nested structures).
    std::istringstream ss(content);
    std::string token;
    while (std::getline(ss, token, ',')) {
        auto colon = token.find(':');
        if (colon == std::string::npos) continue;

        std::string key   = strip_quotes(trim(token.substr(0, colon)));
        std::string value = trim(token.substr(colon + 1));

        // Parse value
        auto is_true  = (value == "true");
        auto is_false = (value == "false");

        try {
            if (key == "max_new_tokens") {
                cfg.max_new_tokens = static_cast<size_t>(std::stoul(value));
            } else if (key == "temperature") {
                cfg.temperature = std::stof(value);
            } else if (key == "top_k") {
                cfg.top_k = static_cast<size_t>(std::stoul(value));
            } else if (key == "top_p") {
                cfg.top_p = std::stof(value);
            } else if (key == "repetition_penalty") {
                cfg.repetition_penalty = std::stof(value);
            } else if (key == "do_sample") {
                if (is_true || is_false) cfg.do_sample = is_true;
            } else if (key == "num_beams") {
                cfg.num_beams = static_cast<size_t>(std::stoul(value));
            } else if (key == "early_stopping") {
                if (is_true || is_false) cfg.early_stopping = is_true;
            } else if (key == "pad_token_id") {
                cfg.pad_token_id = static_cast<size_t>(std::stoul(value));
            } else if (key == "eos_token_id") {
                cfg.eos_token_id = static_cast<size_t>(std::stoul(value));
            }
            // Unknown keys silently ignored.
        } catch (...) {
            // Malformed value — skip.
        }
    }

    return cfg;
}

// ---------------------------------------------------------------------------
// to_json
// ---------------------------------------------------------------------------

std::string ConfigValidator::to_json(const GenerationConfig& cfg)
{
    std::ostringstream out;
    out << "{";
    out << "\"max_new_tokens\":" << cfg.max_new_tokens << ",";
    out << "\"temperature\":"    << cfg.temperature    << ",";
    out << "\"top_k\":"          << cfg.top_k          << ",";
    out << "\"top_p\":"          << cfg.top_p          << ",";
    out << "\"repetition_penalty\":" << cfg.repetition_penalty << ",";
    out << "\"do_sample\":"      << (cfg.do_sample     ? "true" : "false") << ",";
    out << "\"num_beams\":"      << cfg.num_beams      << ",";
    out << "\"early_stopping\":" << (cfg.early_stopping ? "true" : "false") << ",";
    out << "\"pad_token_id\":"   << cfg.pad_token_id   << ",";
    out << "\"eos_token_id\":"   << cfg.eos_token_id;
    out << "}";
    return out.str();
}

}  // namespace llmquant
