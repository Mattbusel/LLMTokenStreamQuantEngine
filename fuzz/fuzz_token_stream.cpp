// =============================================================================
// fuzz/fuzz_token_stream.cpp
// LibFuzzer target for the TokenStreamSimulator and the full token parsing path.
//
// Exercises:
//   1. TokenStreamSimulator::load_tokens_from_memory() with arbitrary token strings
//   2. TokenStreamSimulator start/stop lifecycle with a consumer callback
//   3. Token text content routed through LLMAdapter::map_token_to_weight()
//   4. Deduplicator::check() on fuzz-derived token strings
//   5. TradeSignalEngine::process_semantic_weight() on the resulting weights
//
// Build:
//   cmake -S . -B build-fuzz \
//     -DLLMQUANT_ENABLE_FUZZING=ON \
//     -DCMAKE_CXX_COMPILER=clang++ \
//     -DCMAKE_CXX_FLAGS="-fsanitize=fuzzer-no-link,address,undefined"
//   cmake --build build-fuzz --target fuzz_token_stream

#include "Deduplicator.h"
#include "LLMAdapter.h"
#include "TokenStreamSimulator.h"
#include "TradeSignalEngine.h"

#include <atomic>
#include <chrono>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

using namespace llmquant;

// Static objects shared across fuzzing iterations to exercise stateful paths.
static LLMAdapter g_adapter;
static TradeSignalEngine g_engine = [] {
    TradeSignalEngine::Config cfg;
    cfg.bias_sensitivity       = 1.0;
    cfg.volatility_sensitivity = 1.0;
    cfg.signal_decay_rate      = 0.95;
    cfg.signal_cooldown        = std::chrono::microseconds{0};
    return TradeSignalEngine{cfg};
}();

namespace {
struct Init {
    Init() { g_engine.set_backtest_mode(true); }
};
Init g_init;
} // namespace

extern "C" int LLVMFuzzerTestOneInput(const uint8_t* data, size_t size) {
    if (size == 0) return 0;

    // Split the fuzz buffer on null bytes to produce a sequence of token strings.
    // Cap total byte consumption to 1 KiB to avoid OOM.
    const size_t capped = (size > 1024) ? 1024u : size;

    std::vector<std::string> tokens;
    tokens.reserve(16);

    const char* ptr = reinterpret_cast<const char*>(data);
    const char* end = ptr + capped;
    const char* seg = ptr;

    while (ptr < end) {
        if (*ptr == '\0') {
            tokens.emplace_back(seg, static_cast<size_t>(ptr - seg));
            seg = ptr + 1;
        }
        ++ptr;
    }
    // Remaining bytes after the last null (or the whole buffer if no nulls).
    if (seg < end) {
        tokens.emplace_back(seg, static_cast<size_t>(end - seg));
    }

    if (tokens.empty()) return 0;

    // --- Deduplication layer ---------------------------------------------------
    auto backend = std::make_shared<InProcessDeduplicator>();
    Deduplicator dedup(backend, std::chrono::milliseconds{200});

    // --- TokenStreamSimulator --------------------------------------------------
    TokenStreamSimulator::Config sim_cfg;
    sim_cfg.use_memory_stream = true;
    sim_cfg.token_interval    = std::chrono::microseconds{0}; // no sleep in fuzzing
    sim_cfg.buffer_size       = tokens.size() + 1;

    std::atomic<uint64_t> processed{0};

    {
        TokenStreamSimulator sim(sim_cfg);
        sim.load_tokens_from_memory(tokens);
        sim.set_token_callback([&](const Token& tok) {
            // Stage 1: deduplication
            if (dedup.check(tok.text) == DedupResult::Duplicate) return;

            // Stage 2: semantic weight mapping
            const SemanticWeight w = g_adapter.map_token_to_weight(tok.text);

            // Stage 3: signal generation
            g_engine.process_semantic_weight(w);

            ++processed;
        });
        sim.start();
        // Give the worker thread at most 5 ms to drain the token buffer.
        std::this_thread::sleep_for(std::chrono::milliseconds{5});
        sim.stop();
    }

    // Verify stats are accessible after stop.
    (void)backend->size();

    return 0;
}
