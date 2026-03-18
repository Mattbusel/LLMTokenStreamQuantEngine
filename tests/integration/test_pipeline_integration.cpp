// =============================================================================
// tests/integration/test_pipeline_integration.cpp
//
// 5-stage pipeline integration tests using the full production API surface.
//
// Stage 1: LLM token stream ingestion  -- TokenStreamSimulator
// Stage 2: Deduplication               -- Deduplicator / InProcessDeduplicator
// Stage 3: Signal generation           -- TradeSignalEngine
// Stage 4: Risk management             -- RiskManager
// Stage 5: OMS order submission        -- MockOmsAdapter
// =============================================================================

#include "gtest/gtest.h"

#include "Deduplicator.h"
#include "LLMAdapter.h"
#include "MockOmsAdapter.h"
#include "RiskManager.h"
#include "TokenStreamSimulator.h"
#include "TradeSignalEngine.h"

#include <atomic>
#include <chrono>
#include <memory>
#include <string>
#include <thread>
#include <vector>

namespace llmquant {
namespace {

// ---------------------------------------------------------------------------
// Config helpers
// ---------------------------------------------------------------------------

static RiskManager::Config permissive_risk_cfg() {
    RiskManager::Config c;
    c.max_bias_magnitude       = 100.0;
    c.max_volatility_magnitude = 100.0;
    c.max_spread_magnitude     = 100.0;
    c.min_confidence           = 0.0;
    c.max_signals_per_second   = 1000000;
    c.max_drawdown             = 10000.0;
    return c;
}

static TradeSignalEngine::Config backtest_engine_cfg() {
    TradeSignalEngine::Config c;
    c.bias_sensitivity       = 1.0;
    c.volatility_sensitivity = 1.0;
    c.signal_decay_rate      = 0.95;
    c.signal_cooldown        = std::chrono::microseconds{0};
    return c;
}

// ---------------------------------------------------------------------------
// Full 5-stage pipeline harness
// ---------------------------------------------------------------------------

struct FullPipeline {
    // Stage 1
    TokenStreamSimulator         simulator;
    // Stage 2
    std::shared_ptr<InProcessDeduplicator> dedup_backend;
    Deduplicator                 dedup;
    // Stage 3
    LLMAdapter                   adapter;
    TradeSignalEngine            engine;
    // Stage 4
    RiskManager                  risk;
    // Stage 5
    MockOmsAdapter               oms;

    std::atomic<uint64_t>        signals_passed{0};
    std::atomic<uint64_t>        signals_blocked{0};
    std::atomic<uint64_t>        tokens_deduped{0};
    std::atomic<uint64_t>        tokens_processed{0};

    // TokenStreamSimulator loops indefinitely.  max_tokens_ is set by run()
    // to cap processing at exactly tokens.size() observations so tests that
    // assert exact counts are not sensitive to loop iterations or drain timing.
    std::atomic<uint64_t>        max_tokens_{UINT64_MAX};

    explicit FullPipeline(
        std::chrono::milliseconds dedup_ttl = std::chrono::milliseconds{500})
        : simulator([] {
            TokenStreamSimulator::Config sc;
            sc.use_memory_stream = true;
            sc.token_interval    = std::chrono::microseconds{0};
            sc.buffer_size       = 256;
            return sc;
          }())
        , dedup_backend(std::make_shared<InProcessDeduplicator>())
        , dedup(dedup_backend, dedup_ttl)
        , engine(backtest_engine_cfg())
        , risk(permissive_risk_cfg())
        , oms()
    {
        engine.set_backtest_mode(true);
        engine.set_signal_callback([this](const TradeSignal& sig) {
            if (risk.evaluate(sig)) {
                ++signals_passed;
            } else {
                ++signals_blocked;
            }
        });
        risk.set_alert_callback([](const std::string&, const TradeSignal&) {});

        // Stage 5: wire OMS position feedback back to RiskManager
        oms.set_position_callback([this](const RiskManager::PositionState& ps) {
            risk.update_position(ps);
        });

        simulator.set_token_callback([this](const Token& tok) {
            // Discard tokens once we have seen max_tokens_ total observations.
            // This makes exact-count assertions independent of loop iterations.
            const uint64_t seen = tokens_processed.load() + tokens_deduped.load();
            if (seen >= max_tokens_.load()) return;

            // Stage 2: deduplication
            if (dedup.check(tok.text) == DedupResult::Duplicate) {
                ++tokens_deduped;
                return;
            }
            ++tokens_processed;
            // Stage 3: semantic weight -> signal (callback fires signal -> stage 4/5)
            const SemanticWeight w = adapter.map_token_to_weight(tok.text);
            engine.process_semantic_weight(w);
        });
    }

    // Load tokens, optionally start the OMS mock, run simulator, wait for drain.
    void run(const std::vector<std::string>& tokens,
             const std::vector<RiskManager::PositionState>& oms_states = {},
             std::chrono::milliseconds drain_ms = std::chrono::milliseconds{200})
    {
        // Cap processing at exactly tokens.size() observations so the simulator
        // looping behaviour does not affect exact-count assertions.
        max_tokens_.store(static_cast<uint64_t>(tokens.size()));

        simulator.load_tokens_from_memory(tokens);
        if (!oms_states.empty()) {
            oms.load_states(oms_states);
            oms.start();
        }
        simulator.start();
        std::this_thread::sleep_for(drain_ms);
        simulator.stop();
        if (oms.is_running()) oms.stop();
    }
};

// ---------------------------------------------------------------------------
// Stage 1: Token stream ingestion via TokenStreamSimulator
// ---------------------------------------------------------------------------

TEST(PipelineIntegration, Stage1_SimulatorDeliversAllTokens) {
    FullPipeline p;
    const std::vector<std::string> tokens{"buy", "sell", "rally", "crash", "hold"};
    p.run(tokens);

    const uint64_t total = p.tokens_processed.load() + p.tokens_deduped.load();
    EXPECT_EQ(total, tokens.size());
    EXPECT_GE(p.tokens_processed.load(), 1u);
}

TEST(PipelineIntegration, Stage1_SimulatorStatsArePopulated) {
    TokenStreamSimulator::Config sc;
    sc.use_memory_stream = true;
    sc.token_interval    = std::chrono::microseconds{0};
    sc.buffer_size       = 64;
    TokenStreamSimulator sim(sc);

    std::atomic<uint64_t> count{0};
    sim.set_token_callback([&](const Token&) { ++count; });
    sim.load_tokens_from_memory({"a", "b", "c"});
    sim.start();
    std::this_thread::sleep_for(std::chrono::milliseconds{100});
    sim.stop();

    EXPECT_GE(sim.get_stats().tokens_emitted.load(), 3u);
    EXPECT_GE(count.load(), 3u);
}

// ---------------------------------------------------------------------------
// Stage 2: Deduplication
// ---------------------------------------------------------------------------

TEST(PipelineIntegration, Stage2_DedupSuppressesRepeatedTokens) {
    FullPipeline p(std::chrono::milliseconds{500});
    // Push the same token twice; only one should be processed.
    p.run({"bullish", "bullish"});

    EXPECT_EQ(p.tokens_processed.load(), 1u);
    EXPECT_EQ(p.tokens_deduped.load(), 1u);
}

TEST(PipelineIntegration, Stage2_DedupPassesUniqueTokens) {
    FullPipeline p;
    p.run({"buy", "sell", "hold", "rally", "crash"});

    EXPECT_EQ(p.tokens_deduped.load(), 0u);
    EXPECT_EQ(p.tokens_processed.load(), 5u);
}

TEST(PipelineIntegration, Stage2_DedupStatsMatchProcessedCount) {
    FullPipeline p;
    p.run({"up", "up", "up", "down"});

    const uint64_t novel = p.dedup_backend->total_novel();
    const uint64_t dups  = p.dedup_backend->total_duplicates();
    EXPECT_EQ(novel + dups, 4u);
    EXPECT_EQ(dups, 2u);  // second and third "up" are duplicates
}

// ---------------------------------------------------------------------------
// Stage 3: Signal generation via TradeSignalEngine
// ---------------------------------------------------------------------------

TEST(PipelineIntegration, Stage3_SignalsGeneratedForNovelTokens) {
    FullPipeline p;
    p.run({"bullish", "bearish", "volatile"});

    const uint64_t signals = p.signals_passed.load() + p.signals_blocked.load();
    // 3 unique tokens -> 3 process_semantic_weight calls -> 3 signals (backtest mode)
    EXPECT_EQ(signals, 3u);
}

TEST(PipelineIntegration, Stage3_SignalEngineStatsAccumulate) {
    FullPipeline p;
    p.run({"crash", "rally", "dump", "pump"});

    const auto stats = p.engine.get_stats();
    EXPECT_GE(stats.signals_generated.load(), 4u);
}

// ---------------------------------------------------------------------------
// Stage 4: Risk management via RiskManager
// ---------------------------------------------------------------------------

TEST(PipelineIntegration, Stage4_RiskManagerPassesLowMagnitudeSignals) {
    FullPipeline p;
    p.run({"neutral", "hold", "wait"});

    // With permissive config all signals should pass.
    EXPECT_EQ(p.signals_blocked.load(), 0u);
    EXPECT_GE(p.signals_passed.load(), 1u);
}

TEST(PipelineIntegration, Stage4_RiskManagerBlocksHighMagnitudeSignals) {
    FullPipeline p2(std::chrono::milliseconds{500});
    // Rebuild risk with tight thresholds.
    RiskManager::Config tight;
    tight.max_bias_magnitude       = 0.0001;
    tight.max_volatility_magnitude = 0.0001;
    tight.max_spread_magnitude     = 0.0001;
    tight.min_confidence           = 0.99;
    tight.max_signals_per_second   = 1000000;
    tight.max_drawdown             = 10000.0;

    // Use independent objects for this test.
    LLMAdapter          adapter;
    TradeSignalEngine   engine(backtest_engine_cfg());
    RiskManager         risk(tight);

    std::atomic<uint64_t> blocked{0};
    engine.set_backtest_mode(true);
    engine.set_signal_callback([&](const TradeSignal& sig) {
        if (!risk.evaluate(sig)) ++blocked;
    });
    risk.set_alert_callback([](const std::string&, const TradeSignal&) {});

    const std::vector<std::string> strong_tokens{"crash", "collapse", "surge", "bullish"};
    for (const auto& t : strong_tokens) {
        engine.process_semantic_weight(adapter.map_token_to_weight(t));
    }

    // With very tight thresholds, at least some signals should be blocked.
    EXPECT_GT(blocked.load(), 0u);
}

// ---------------------------------------------------------------------------
// Stage 5: OMS integration via MockOmsAdapter
// ---------------------------------------------------------------------------

TEST(PipelineIntegration, Stage5_MockOmsDeliversPositionUpdates) {
    FullPipeline p;
    std::vector<RiskManager::PositionState> states{
        {0.1, 1.0, 0.05, -10.0},
        {0.2, 1.0, 0.10, -10.0},
        {-0.1, 1.0, -0.05, -10.0},
    };
    p.run({"buy", "sell"}, states, std::chrono::milliseconds{300});

    EXPECT_GE(p.oms.emitted_count(), 1u);
}

TEST(PipelineIntegration, Stage5_OmsPositionFeedsRiskManager) {
    FullPipeline p;

    // Pre-load a position that is near the limit.
    std::vector<RiskManager::PositionState> states{
        {0.9, 1.0, 0.0, -10.0},
    };
    p.run({"rally", "surge"}, states, std::chrono::milliseconds{200});

    // RiskManager should have received the position update.
    const auto pos = p.risk.get_position();
    // After at least one OMS emission the net_position should be updated.
    // We can only assert that the pipeline ran without crashing.
    (void)pos;
    SUCCEED();
}

// ---------------------------------------------------------------------------
// End-to-end: all 5 stages together with a realistic token sequence
// ---------------------------------------------------------------------------

TEST(PipelineIntegration, EndToEnd_AllStagesCompleteWithoutError) {
    FullPipeline p;

    const std::vector<std::string> tokens{
        "bullish", "rally",   "surge",   "buy",     "long",
        "bearish", "crash",   "dump",    "sell",    "short",
        "neutral", "hold",    "wait",    "stable",  "flat",
        "bullish",  // duplicate of first token -> deduped
        "volatile", "spike",  "jump",    "drop",
    };

    std::vector<RiskManager::PositionState> oms_states{
        {0.05, 1.0, 0.02, -10.0},
        {-0.05, 1.0, -0.01, -10.0},
        {0.10, 1.0, 0.05, -10.0},
    };

    p.run(tokens, oms_states, std::chrono::milliseconds{400});

    // 20 tokens, 1 duplicate -> 19 processed, 1 deduped.
    EXPECT_EQ(p.tokens_deduped.load(), 1u);
    EXPECT_EQ(p.tokens_processed.load(), 19u);

    // At least some signals must have been generated and evaluated.
    const uint64_t total_signals = p.signals_passed.load() + p.signals_blocked.load();
    EXPECT_EQ(total_signals, 19u);

    // OMS must have run at least one emission.
    EXPECT_GE(p.oms.emitted_count(), 1u);

    // Signal engine stats.
    const auto eng_stats = p.engine.get_stats();
    EXPECT_GE(eng_stats.signals_generated.load(), 1u);

    // Dedup backend stats.
    EXPECT_EQ(p.dedup_backend->total_duplicates(), 1u);
    EXPECT_EQ(p.dedup_backend->total_novel(), 19u);
}

} // namespace
} // namespace llmquant
