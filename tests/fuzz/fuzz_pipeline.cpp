// LibFuzzer target for the full 5-stage token processing pipeline.
//
// Stage 1: Deduplicator::check        (FNV-1a hash + TTL table)
// Stage 2: LatencyController::start   (timestamp capture)
// Stage 3: LLMAdapter::map_token_to_weight (unordered_map lookup)
// Stage 4: TradeSignalEngine::process_semantic_weight (CAS accumulators)
// Stage 5: RiskManager::evaluate      (5-gate cascade)
//
// A single fuzz input drives all five stages so the fuzzer can discover
// interactions across stage boundaries (e.g. dedup short-circuits before
// the adapter, extreme weights that saturate the risk gate).
//
// Build:
//   cmake -S . -B build-fuzz \
//     -DLLMQUANT_ENABLE_FUZZING=ON \
//     -DCMAKE_CXX_COMPILER=clang++ \
//     -DCMAKE_CXX_FLAGS="-fsanitize=fuzzer,address,undefined"
//   cmake --build build-fuzz --target fuzz_pipeline

#include "Deduplicator.h"
#include "LatencyController.h"
#include "LLMAdapter.h"
#include "RiskManager.h"
#include "TradeSignalEngine.h"

#include <chrono>
#include <cstdint>
#include <memory>

using namespace llmquant;

// The pipeline objects are created once per fuzzing process (not per input)
// to exercise the stateful accumulator and rate-limit paths across calls.
static LLMAdapter         g_adapter;
static TradeSignalEngine  g_engine = [] {
    TradeSignalEngine::Config cfg;
    cfg.bias_sensitivity       = 1.0;
    cfg.volatility_sensitivity = 1.0;
    cfg.signal_decay_rate      = 0.95;
    cfg.signal_cooldown        = std::chrono::microseconds{0};
    return TradeSignalEngine{cfg};
}();
static RiskManager g_risk = [] {
    RiskManager::Config cfg;
    cfg.max_bias_magnitude      = 2.0;
    cfg.max_volatility_magnitude = 2.0;
    cfg.max_spread_magnitude    = 1.0;
    cfg.min_confidence          = 0.0;
    cfg.max_signals_per_second  = 10000;
    cfg.max_drawdown            = 100.0;
    return RiskManager{cfg};
}();

namespace {
    // One-time initialisation: register callbacks that must not throw.
    struct Init {
        Init() {
            g_engine.set_backtest_mode(true);
            g_engine.set_signal_callback([](const TradeSignal& s) {
                (void)g_risk.evaluate(s);
            });
            g_risk.set_alert_callback([](const std::string&, const TradeSignal&) {});
        }
    };
    Init g_init;
} // namespace

extern "C" int LLVMFuzzerTestOneInput(const uint8_t* data, size_t size) {
    if (size == 0) return 0;

    // Interpret the fuzz buffer as a null-terminated token string (clamped to
    // 512 bytes so the fuzzer does not exhaust memory on huge inputs).
    const size_t token_len = (size > 512) ? 512 : size;
    const std::string token(reinterpret_cast<const char*>(data), token_len);

    // Stage 1: deduplication
    static auto s_backend = std::make_shared<InProcessDeduplicator>();
    static Deduplicator s_dedup(s_backend, std::chrono::milliseconds{100});
    if (s_dedup.check(token) == DedupResult::Duplicate) return 0;

    // Stage 2: latency measurement (start only; end is implicit)
    LatencyController::Config lc_cfg;
    lc_cfg.target_latency   = std::chrono::microseconds{50};
    lc_cfg.sample_window    = 64;
    lc_cfg.enable_profiling = false;
    static LatencyController s_lc(lc_cfg);
    s_lc.start_measurement();

    // Stage 3: semantic weight mapping
    const SemanticWeight w = g_adapter.map_token_to_weight(token);

    // Stage 4: signal generation + Stage 5: risk gate (via callback above)
    g_engine.process_semantic_weight(w);

    s_lc.end_measurement();

    return 0;
}
