#pragma once
// GenerationConfig.h — Configuration and validation for LLM text generation.

#include <cstddef>
#include <string>
#include <vector>

namespace llmquant {

/// All parameters that govern a single generation request.
struct GenerationConfig {
    size_t max_new_tokens    = 128;
    float  temperature       = 1.0f;
    size_t top_k             = 50;
    float  top_p             = 1.0f;
    float  repetition_penalty = 1.0f;
    bool   do_sample         = true;
    size_t num_beams         = 1;
    bool   early_stopping    = false;
    size_t pad_token_id      = 0;
    size_t eos_token_id      = 1;
};

/// Validates and serialises :class:`GenerationConfig` instances.
class ConfigValidator {
public:
    ConfigValidator() = default;

    // ------------------------------------------------------------------
    // Validation
    // ------------------------------------------------------------------

    /// Return a list of human-readable validation errors.
    /// Empty list means the config is valid.
    std::vector<std::string> validate(const GenerationConfig& config) const;

    /// Return true iff validate() returns an empty list.
    bool is_valid(const GenerationConfig& config) const;

    // ------------------------------------------------------------------
    // Factory helpers
    // ------------------------------------------------------------------

    /// Construct a GenerationConfig with conservative, sensible defaults.
    static GenerationConfig with_defaults();

    /// Parse a GenerationConfig from a minimal JSON string.
    /// Supports numeric and boolean values only; no arrays/nested objects.
    /// Unknown keys are silently ignored.
    static GenerationConfig from_json(const std::string& json_str);

    // ------------------------------------------------------------------
    // Serialisation
    // ------------------------------------------------------------------

    /// Serialise a GenerationConfig to a compact JSON string.
    static std::string to_json(const GenerationConfig& config);
};

}  // namespace llmquant
