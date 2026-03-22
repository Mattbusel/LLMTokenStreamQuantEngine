// =============================================================================
// fuzz/fuzz_pipeline.cpp
// LibFuzzer target for the full signal-processing pipeline.
//
// Exercises:
//   1. LLMAdapter::map_token_to_weight()      — token → semantic weight
//   2. RiskManager::evaluate()                — risk gate evaluation
//   3. TradeSignalEngine::process_semantic_weight() — weight → signal
//   4. LatencyController::record_latency()    — latency stats accumulation
//   5. Pipeline wiring with arbitrary token text and fuzz-derived risk params
//
// The fuzzer drives the token text and risk manager config values from the
// input buffer so that LibFuzzer can explore edge cases in all layers.
//
// Build:
//   cmake -S . -B build-fuzz \
//     -DLLMQUANT_ENABLE_FUZZING=ON \
//     -DCMAKE_CXX_COMPILER=clang++ \
//     -DCMAKE_CXX_FLAGS="-fsanitize=fuzzer-no-link,address,undefined"
//   cmake --build build-fuzz --target fuzz_pipeline
// =============================================================================

#include "LLMAdapter.h"
#include "LatencyController.h"
#include "RiskManager.h"
#include "TradeSignalEngine.h"

#include <chrono>
#include <cstdint>
#include <cstring>
#include <string>

using namespace llmquant;

// Static objects reused across iterations to exercise stateful accumulation.
static LLMAdapter g_adapter;

extern "C" int LLVMFuzzerTestOneInput(const uint8_t* data, size_t size) {
    if (size < 4) return 0;

    // First 4 bytes steer the risk config; the remainder is the token text.
    // Byte 0: magnitude_threshold   [0.01 .. 2.56] (value/100.0)
    // Byte 1: confidence_threshold  [0.01 .. 1.00] (value/255.0)
    // Byte 2: max_position          [1.0 .. 256.0]  (value + 1.0)
    // Byte 3: max_drawdown          [0.01 .. 2.56] (value/100.0)
    const double magnitude_thresh  = (data[0] == 0 ? 1 : data[0]) / 100.0;
    const double confidence_thresh = (data[1] == 0 ? 1 : data[1]) / 255.0;
    const double max_position      = static_cast<double>(data[2]) + 1.0;
    const double max_drawdown      = (data[3] == 0 ? 1 : data[3]) / 100.0;

    const std::string token_text(reinterpret_cast<const char*>(data + 4), size - 4);

    // --- RiskManager setup ---------------------------------------------------
    RiskManager::Config risk_cfg;
    risk_cfg.magnitude_threshold  = magnitude_thresh;
    risk_cfg.confidence_threshold = confidence_thresh;
    risk_cfg.max_position         = max_position;
    risk_cfg.max_drawdown         = max_drawdown;
    risk_cfg.max_rate_per_second  = 1000.0;  // generous — not under test here

    RiskManager risk;
    try {
        risk.update_config(risk_cfg);
    } catch (...) {
        // Fuzz may supply extreme values; update_config validates them.
        return 0;
    }

    // --- TradeSignalEngine setup ---------------------------------------------
    TradeSignalEngine::Config engine_cfg;
    engine_cfg.bias_sensitivity       = 1.0;
    engine_cfg.volatility_sensitivity = 1.0;
    engine_cfg.signal_decay_rate      = 0.95;
    engine_cfg.signal_cooldown        = std::chrono::microseconds{0};

    TradeSignalEngine engine(engine_cfg);
    engine.set_backtest_mode(true);

    // --- LatencyController setup ---------------------------------------------
    LatencyController::Config lat_config;
    lat_config.enable_profiling = true;
    lat_config.sample_window    = 16;
    LatencyController lat_ctrl(lat_config);

    // --- Pipeline execution --------------------------------------------------
    // 1. Map the fuzz-derived token text to a semantic weight.
    const SemanticWeight weight = g_adapter.map_token_to_weight(token_text);

    // 2. Run the risk gate.
    const bool allowed = risk.evaluate(weight);

    if (allowed) {
        // 3. Feed weight into the signal engine under latency measurement.
        lat_ctrl.start_measurement();
        engine.process_semantic_weight(weight);
        lat_ctrl.end_measurement();
    }

    // 4. Drain stats — must not crash on any accumulated state.
    (void)lat_ctrl.get_stats();
    (void)engine.get_current_signal();
    (void)risk.get_stats();

    return 0;
}
