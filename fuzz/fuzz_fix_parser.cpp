// =============================================================================
// fuzz/fuzz_fix_parser.cpp
// LibFuzzer target for the FIX 4.2 message parser.
//
// Exercises:
//   1. FixOmsAdapter::parse_fix()        — tag=value SOH-delimited parsing
//   2. FixOmsAdapter::fix_message()      — body → framed FIX message builder
//   3. fix_utctime()                     — UTC timestamp formatter (via public proxy)
//
// Both parse_fix() and fix_message() are static methods so no instance is
// required; this lets the fuzzer exercise the network-facing parse surface
// without a live socket.
//
// Build:
//   cmake -S . -B build-fuzz \
//     -DLLMQUANT_ENABLE_FUZZING=ON \
//     -DCMAKE_CXX_COMPILER=clang++ \
//     -DCMAKE_CXX_FLAGS="-fsanitize=fuzzer-no-link,address,undefined"
//   cmake --build build-fuzz --target fuzz_fix_parser
// =============================================================================

#include "FixOmsAdapter.h"

#include <cstdint>
#include <string>

using namespace llmquant;

// Thin subclass that promotes protected parse_fix() and fix_message() to public
// so the fuzzer can call them without a live FIX session.
class FuzzableFixAdapter : public FixOmsAdapter {
public:
    explicit FuzzableFixAdapter(FixOmsAdapter::Config cfg)
        : FixOmsAdapter(std::move(cfg)) {}

    using FixOmsAdapter::parse_fix;
    using FixOmsAdapter::fix_message;
};

// Static instance reused across iterations to exercise stateful paths.
static FuzzableFixAdapter g_adapter = [] {
    FixOmsAdapter::Config cfg;
    cfg.host              = "127.0.0.1";
    cfg.port              = 9999;
    cfg.sender_comp_id    = "FUZZ_SENDER";
    cfg.target_comp_id    = "FUZZ_TARGET";
    cfg.heartbeat_interval_s = 30;
    return FuzzableFixAdapter{cfg};
}();

extern "C" int LLVMFuzzerTestOneInput(const uint8_t* data, size_t size) {
    // Feed arbitrary bytes as a raw FIX wire message.
    // parse_fix() must not crash, throw unhandled exceptions, or invoke UB.
    const std::string raw(reinterpret_cast<const char*>(data), size);

    // 1. Parse the raw bytes as a FIX message.
    auto fields = g_adapter.parse_fix(raw);

    // 2. Round-trip: re-encode the parsed field map through fix_message()
    //    and parse again.  Both directions must not crash.
    if (!fields.empty()) {
        // Build a minimal FIX body from the first few parsed tags.
        std::string body;
        int count = 0;
        for (const auto& kv : fields) {
            body += std::to_string(kv.first) + "=" + kv.second;
            body += '\x01';  // SOH
            if (++count >= 8) break;  // cap to avoid huge messages
        }
        std::string framed = g_adapter.fix_message(body);
        (void)g_adapter.parse_fix(framed);
    }

    return 0;
}
