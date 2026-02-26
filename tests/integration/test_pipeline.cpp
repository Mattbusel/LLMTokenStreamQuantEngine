#include "gtest/gtest.h"

#include "TokenStreamSimulator.h"
#include "LLMAdapter.h"
#include "TradeSignalEngine.h"
#include "LatencyController.h"
#include "OutputSink.h"

#include <atomic>
#include <chrono>
#include <cmath>
#include <thread>
#include <vector>

namespace llmquant {
namespace {

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

static TokenStreamSimulator::Config sim_config(int interval_us = 500) {
    TokenStreamSimulator::Config cfg;
    cfg.token_interval    = std::chrono::microseconds{interval_us};
    cfg.buffer_size       = 64;
    cfg.use_memory_stream = true;
    return cfg;
}

static TradeSignalEngine::Config engine_config_backtest() {
    TradeSignalEngine::Config cfg;
    cfg.bias_sensitivity       = 1.0;
    cfg.volatility_sensitivity = 1.0;
    cfg.signal_decay_rate      = 0.95;
    cfg.signal_cooldown        = std::chrono::microseconds{0};
    return cfg;
}

// ---------------------------------------------------------------------------
// Integration tests
// ---------------------------------------------------------------------------

TEST(PipelineIntegration, test_pipeline_end_to_end_tokens_produce_signals) {
    // Wire up the full pipeline: Simulator -> LLMAdapter -> TradeSignalEngine
    // -> MemoryOutputSink.  Run for 300 ms and assert signals were emitted.

    LLMAdapter adapter;
    TradeSignalEngine engine(engine_config_backtest());
    engine.set_backtest_mode(true);

    MemoryOutputSink sink;
    engine.set_signal_callback([&sink](const TradeSignal& s) { sink.emit(s); });

    TokenStreamSimulator sim(sim_config(500 /*0.5 ms*/));
    // Use fear tokens that have strong mappings in the default dictionary.
    sim.load_tokens_from_memory({"crash", "panic", "bearish", "plunge"});

    sim.set_token_callback([&adapter, &engine](const Token& tok) {
        SemanticWeight w = adapter.map_token_to_weight(tok.text);
        engine.process_semantic_weight(w);
    });

    sim.start();
    std::this_thread::sleep_for(std::chrono::milliseconds{300});
    sim.stop();

    EXPECT_GT(sink.get_signals().size(), 0u)
        << "Full pipeline must have produced at least one signal in 300 ms";

    // All signals from fear tokens must have negative bias.
    for (const auto& s : sink.get_signals()) {
        EXPECT_LT(s.delta_bias_shift, 0.01)
            << "Fear tokens should not produce strongly positive bias";
    }
}

TEST(PipelineIntegration, test_pipeline_latency_under_10us_for_single_token) {
    // Directly time one token -> weight -> signal cycle.
    LLMAdapter adapter;

    TradeSignalEngine::Config cfg = engine_config_backtest();
    TradeSignalEngine engine(cfg);
    engine.set_backtest_mode(true);
    engine.set_signal_callback([](const TradeSignal&) {});

    SemanticWeight w = adapter.map_token_to_weight("bullish");

    auto t0 = std::chrono::high_resolution_clock::now();
    engine.process_semantic_weight(w);
    auto t1 = std::chrono::high_resolution_clock::now();

    auto elapsed_us = std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count();

    // 10 μs is the declared budget.  On a loaded CI machine we allow 5x headroom.
    EXPECT_LT(elapsed_us, 50)
        << "Single token->signal cycle took " << elapsed_us << " μs, expected < 50 μs";
}

TEST(PipelineIntegration, test_pipeline_multiple_tokens_accumulate_correctly) {
    // 10 consecutive bullish tokens must drive the accumulated bias positive.
    LLMAdapter adapter;
    TradeSignalEngine engine(engine_config_backtest());
    engine.set_backtest_mode(true);

    TradeSignal last_signal;
    engine.set_signal_callback([&last_signal](const TradeSignal& s) { last_signal = s; });

    SemanticWeight bullish = adapter.map_token_to_weight("bullish");
    for (int i = 0; i < 10; ++i) {
        engine.process_semantic_weight(bullish);
    }

    EXPECT_GT(last_signal.delta_bias_shift, 0.0)
        << "10 bullish tokens must yield a positive accumulated bias";
}

TEST(PipelineIntegration, test_pipeline_latency_controller_records_end_to_end_latency) {
    // Exercise LatencyController alongside the engine.
    LatencyController::Config lc_cfg;
    lc_cfg.target_latency   = std::chrono::microseconds{10};
    lc_cfg.sample_window    = 200;
    lc_cfg.enable_profiling = true;

    LatencyController lc(lc_cfg);
    LLMAdapter adapter;
    TradeSignalEngine engine(engine_config_backtest());
    engine.set_backtest_mode(true);
    engine.set_signal_callback([](const TradeSignal&) {});

    SemanticWeight w = adapter.map_token_to_weight("rally");

    for (int i = 0; i < 50; ++i) {
        lc.start_measurement();
        engine.process_semantic_weight(w);
        lc.end_measurement();
    }

    auto stats = lc.get_stats();
    EXPECT_EQ(stats.measurements, 50u);
    EXPECT_GT(stats.max_latency.count(), 0);
}

TEST(PipelineIntegration, test_pipeline_output_sink_captures_all_signals) {
    // Verify that the number of signals captured by MemoryOutputSink matches
    // the engine's own signals_generated counter.
    LLMAdapter adapter;
    TradeSignalEngine engine(engine_config_backtest());
    engine.set_backtest_mode(true);

    MemoryOutputSink sink;
    engine.set_signal_callback([&sink](const TradeSignal& s) { sink.emit(s); });

    SemanticWeight w{0.5, 0.7, 0.4, 0.5};
    const int N = 20;
    for (int i = 0; i < N; ++i) {
        engine.process_semantic_weight(w);
    }

    EXPECT_EQ(static_cast<uint64_t>(sink.get_signals().size()),
              engine.get_stats().signals_generated.load())
        << "Sink and engine signal counts must match";
}

} // namespace
} // namespace llmquant
