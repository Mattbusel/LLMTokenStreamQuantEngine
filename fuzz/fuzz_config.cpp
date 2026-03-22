// =============================================================================
// fuzz/fuzz_config.cpp
// LibFuzzer target for the Config YAML parser.
//
// Exercises:
//   1. Config::load_from_string() — YAML parser on arbitrary byte sequences
//   2. Config::validate()         — validation logic on parsed config
//   3. Config::to_summary_string() — serialisation of parsed state
//   4. Config::save()             — round-trip: parse → serialise → parse
//
// Build:
//   cmake -S . -B build-fuzz \
//     -DLLMQUANT_ENABLE_FUZZING=ON \
//     -DCMAKE_CXX_COMPILER=clang++ \
//     -DCMAKE_CXX_FLAGS="-fsanitize=fuzzer-no-link,address,undefined"
//   cmake --build build-fuzz --target fuzz_config
// =============================================================================

#include "Config.h"

#include <cstdint>
#include <string>

using namespace llmquant;

extern "C" int LLVMFuzzerTestOneInput(const uint8_t* data, size_t size) {
    // Feed arbitrary bytes as YAML to the config parser.
    // Config::load_from_string must not crash, throw unhandled exceptions,
    // or invoke undefined behaviour on any input.
    const std::string yaml_str(reinterpret_cast<const char*>(data), size);

    Config cfg;

    // load_from_string may return false on invalid YAML — that is fine.
    bool loaded = cfg.load_from_yaml_string(yaml_str);
    (void)loaded;

    // validate() must not crash regardless of what load_from_yaml_string produced.
    (void)cfg.validate();

    // to_summary_string() and to_yaml_string() must not crash on any parsed state.
    (void)cfg.to_summary_string();

    // In-memory round-trip: serialise to YAML string, then re-parse.
    // Exercises the YAML emitter without file I/O.
    {
        std::string re_serialised = cfg.to_yaml_string();
        Config cfg_rt;
        (void)cfg_rt.load_from_yaml_string(re_serialised);
        (void)cfg_rt.validate();
    }

    // Round-trip: serialise to a temp file and reload.
    // This exercises save() and the YAML emitter path.
    if (loaded) {
        const std::string tmp = "/tmp/fuzz_config_rt.yaml";
        if (cfg.save_to_file(tmp)) {
            Config cfg2;
            (void)cfg2.load_from_file(tmp);
            (void)cfg2.validate();
        }
    }

    return 0;
}
