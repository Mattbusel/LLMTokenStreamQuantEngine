// SamplerConfig.cpp — Sampler configuration implementation.
#include "SamplerConfig.h"

#include <algorithm>
#include <cctype>
#include <sstream>
#include <stdexcept>
#include <string>

namespace llmquant {

// ---------------------------------------------------------------------------
// Preset factories
// ---------------------------------------------------------------------------

SamplerConfig SamplerConfig::greedy() {
    SamplerConfig cfg;
    cfg.temperature        = 0.0f;   // Will trigger argmax path in sampler
    cfg.top_k              = 1;
    cfg.top_p              = 1.0f;
    cfg.repetition_penalty = 1.0f;
    cfg.min_p              = 0.0f;
    cfg.typical_p          = 1.0f;
    cfg.max_new_tokens     = 256;
    return cfg;
}

SamplerConfig SamplerConfig::creative() {
    SamplerConfig cfg;
    cfg.temperature        = 1.2f;
    cfg.top_k              = 100;
    cfg.top_p              = 0.95f;
    cfg.repetition_penalty = 1.1f;
    cfg.min_p              = 0.0f;
    cfg.typical_p          = 1.0f;
    cfg.max_new_tokens     = 512;
    return cfg;
}

SamplerConfig SamplerConfig::balanced() {
    SamplerConfig cfg;
    cfg.temperature        = 0.8f;
    cfg.top_k              = 50;
    cfg.top_p              = 0.9f;
    cfg.repetition_penalty = 1.05f;
    cfg.min_p              = 0.0f;
    cfg.typical_p          = 1.0f;
    cfg.max_new_tokens     = 256;
    return cfg;
}

SamplerConfig SamplerConfig::precise() {
    SamplerConfig cfg;
    cfg.temperature        = 0.3f;
    cfg.top_k              = 20;
    cfg.top_p              = 0.8f;
    cfg.repetition_penalty = 1.0f;
    cfg.min_p              = 0.01f;
    cfg.typical_p          = 1.0f;
    cfg.max_new_tokens     = 128;
    return cfg;
}

SamplerConfig SamplerConfig::code_generation() {
    SamplerConfig cfg;
    cfg.temperature        = 0.2f;
    cfg.top_k              = 10;
    cfg.top_p              = 0.85f;
    cfg.repetition_penalty = 1.15f;
    cfg.min_p              = 0.02f;
    cfg.typical_p          = 1.0f;
    cfg.max_new_tokens     = 1024;
    cfg.stop_sequences     = {"```", "# end", "// end"};
    return cfg;
}

SamplerConfig SamplerConfig::story_telling() {
    SamplerConfig cfg;
    cfg.temperature        = 0.95f;
    cfg.top_k              = 80;
    cfg.top_p              = 0.92f;
    cfg.repetition_penalty = 1.08f;
    cfg.min_p              = 0.0f;
    cfg.typical_p          = 0.95f;
    cfg.max_new_tokens     = 768;
    return cfg;
}

SamplerConfig SamplerConfig::from_preset(SamplerPreset preset) {
    switch (preset) {
        case SamplerPreset::Greedy:          return greedy();
        case SamplerPreset::Creative:        return creative();
        case SamplerPreset::Balanced:        return balanced();
        case SamplerPreset::Precise:         return precise();
        case SamplerPreset::CodeGeneration:  return code_generation();
        case SamplerPreset::StoryTelling:    return story_telling();
        default:                             return balanced();
    }
}

// ---------------------------------------------------------------------------
// Validation
// ---------------------------------------------------------------------------

bool SamplerConfig::validate() const {
    if (temperature < 0.0f || temperature > 10.0f) return false;
    if (top_k < 0 || top_k > 1000000)              return false;
    if (top_p < 0.0f || top_p > 1.0f)              return false;
    if (repetition_penalty < 0.1f || repetition_penalty > 10.0f) return false;
    if (min_p < 0.0f || min_p > 1.0f)              return false;
    if (typical_p < 0.0f || typical_p > 1.0f)      return false;
    if (max_new_tokens <= 0 || max_new_tokens > 65536) return false;
    return true;
}

// ---------------------------------------------------------------------------
// Serialization
// ---------------------------------------------------------------------------

std::string SamplerConfig::to_json() const {
    std::ostringstream oss;
    oss << "{";
    oss << "\"temperature\":"        << temperature        << ",";
    oss << "\"top_k\":"              << top_k              << ",";
    oss << "\"top_p\":"              << top_p              << ",";
    oss << "\"repetition_penalty\":" << repetition_penalty << ",";
    oss << "\"min_p\":"              << min_p              << ",";
    oss << "\"typical_p\":"          << typical_p          << ",";
    oss << "\"max_new_tokens\":"     << max_new_tokens     << ",";
    oss << "\"stop_sequences\":[";
    for (size_t i = 0; i < stop_sequences.size(); ++i) {
        if (i > 0) oss << ",";
        oss << "\"" << stop_sequences[i] << "\"";
    }
    oss << "]}";
    return oss.str();
}

// Minimal key-value JSON parser — handles flat objects with string/number values.
static std::string trim(const std::string& s) {
    size_t start = s.find_first_not_of(" \t\r\n\"");
    size_t end   = s.find_last_not_of(" \t\r\n\"");
    if (start == std::string::npos) return "";
    return s.substr(start, end - start + 1);
}

SamplerConfig SamplerConfig::from_json(const std::string& json) {
    SamplerConfig cfg = balanced();  // Defaults

    // Very simple tokenizer: extract key:value pairs
    std::string s = json;
    // Remove outer braces
    auto first_brace = s.find('{');
    auto last_brace  = s.rfind('}');
    if (first_brace == std::string::npos) return cfg;
    s = s.substr(first_brace + 1, last_brace - first_brace - 1);

    // Split on commas (naively, won't handle nested arrays well for stop_sequences)
    // Process token by token
    size_t pos = 0;
    while (pos < s.size()) {
        // Find colon
        size_t colon = s.find(':', pos);
        if (colon == std::string::npos) break;
        std::string key = trim(s.substr(pos, colon - pos));

        // Find value end (comma or end of string)
        size_t comma = s.find(',', colon + 1);
        // Skip array values by finding the matching bracket
        size_t val_start = colon + 1;
        // Skip whitespace
        while (val_start < s.size() && std::isspace(static_cast<unsigned char>(s[val_start])))
            ++val_start;

        std::string val_str;
        if (val_start < s.size() && s[val_start] == '[') {
            // Find matching bracket
            int depth = 0;
            size_t i = val_start;
            for (; i < s.size(); ++i) {
                if (s[i] == '[') ++depth;
                else if (s[i] == ']') { --depth; if (depth == 0) break; }
            }
            // skip stop_sequences for now
            pos = (i + 1 < s.size()) ? (s.find(',', i + 1) == std::string::npos ? s.size() : s.find(',', i + 1) + 1) : s.size();
            continue;
        } else {
            size_t val_end = (comma == std::string::npos) ? s.size() : comma;
            val_str = trim(s.substr(colon + 1, val_end - colon - 1));
            pos = (comma == std::string::npos) ? s.size() : comma + 1;
        }

        if (key.empty() || val_str.empty()) continue;

        try {
            if      (key == "temperature")        cfg.temperature        = std::stof(val_str);
            else if (key == "top_k")              cfg.top_k              = std::stoi(val_str);
            else if (key == "top_p")              cfg.top_p              = std::stof(val_str);
            else if (key == "repetition_penalty") cfg.repetition_penalty = std::stof(val_str);
            else if (key == "min_p")              cfg.min_p              = std::stof(val_str);
            else if (key == "typical_p")          cfg.typical_p          = std::stof(val_str);
            else if (key == "max_new_tokens")     cfg.max_new_tokens     = std::stoi(val_str);
        } catch (...) {
            // Skip unparseable values
        }
    }
    return cfg;
}

// ---------------------------------------------------------------------------
// Merge
// ---------------------------------------------------------------------------

SamplerConfig SamplerConfig::merge(const SamplerConfig& other) const {
    SamplerConfig ref = balanced();  // Reference for "default" detection
    SamplerConfig result = *this;

    // Override fields where other differs from the reference default
    if (other.temperature        != ref.temperature)        result.temperature        = other.temperature;
    if (other.top_k              != ref.top_k)              result.top_k              = other.top_k;
    if (other.top_p              != ref.top_p)              result.top_p              = other.top_p;
    if (other.repetition_penalty != ref.repetition_penalty) result.repetition_penalty = other.repetition_penalty;
    if (other.min_p              != ref.min_p)              result.min_p              = other.min_p;
    if (other.typical_p          != ref.typical_p)          result.typical_p          = other.typical_p;
    if (other.max_new_tokens     != ref.max_new_tokens)     result.max_new_tokens     = other.max_new_tokens;
    if (!other.stop_sequences.empty())                      result.stop_sequences     = other.stop_sequences;

    return result;
}

} // namespace llmquant
