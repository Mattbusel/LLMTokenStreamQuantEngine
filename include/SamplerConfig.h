#pragma once
// SamplerConfig.h — Unified sampler configuration with presets.
// Encapsulates all sampling hyper-parameters (temperature, top-k, top-p,
// repetition penalty, min-p, typical-p) and provides named preset factories.

#include <string>
#include <vector>

namespace llmquant {

/// Named preset profiles for common generation scenarios.
enum class SamplerPreset {
    Greedy,
    Creative,
    Balanced,
    Precise,
    CodeGeneration,
    StoryTelling
};

/// Complete sampler configuration.
struct SamplerConfig {
    float       temperature{1.0f};
    int         top_k{50};
    float       top_p{0.9f};
    float       repetition_penalty{1.0f};
    float       min_p{0.0f};
    float       typical_p{1.0f};
    int         max_new_tokens{256};
    std::vector<std::string> stop_sequences;

    // ---------------------------------------------------------------------------
    // Static preset factories
    // ---------------------------------------------------------------------------

    /// Greedy decoding — always picks the highest-probability token.
    static SamplerConfig greedy();

    /// High-temperature creative generation.
    static SamplerConfig creative();

    /// Balanced quality / diversity defaults.
    static SamplerConfig balanced();

    /// Low-temperature factual / precise output.
    static SamplerConfig precise();

    /// Optimised for code generation (low temperature, repetition penalty).
    static SamplerConfig code_generation();

    /// Optimised for story-telling (moderate temperature, top-p).
    static SamplerConfig story_telling();

    /// Factory from preset enum.
    static SamplerConfig from_preset(SamplerPreset preset);

    // ---------------------------------------------------------------------------
    // Validation and serialization
    // ---------------------------------------------------------------------------

    /// Returns true if all field values are within valid ranges.
    bool validate() const;

    /// Serialize to a JSON string.
    std::string to_json() const;

    /// Deserialize from a simple key-value JSON string.
    static SamplerConfig from_json(const std::string& json);

    /// Merge: non-default values from other override this config.
    /// "Default" is detected by comparing against the balanced() preset.
    SamplerConfig merge(const SamplerConfig& other) const;
};

} // namespace llmquant
