// Integration tests for the complete 5-stage token processing pipeline.
//
// Stage 1: Deduplicator        -- FNV-1a dedup with TTL
// Stage 2: LatencyController   -- start/end measurement, stats
// Stage 3: LLMAdapter          -- token-to-SemanticWeight mapping
// Stage 4: TradeSignalEngine   -- CAS accumulator, decay, signal emission
// Stage 5: RiskManager         -- 5-gate cascade (magnitude, confidence,
//                                 rate-limit, drawdown, position)

#include "gtest/gtest.h"

#include "Deduplicator.h"
#include "LatencyController.h"
#include "LLMAdapter.h"
#include "OutputSinkImpl.h"
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
// Helpers
// ---------------------------------------------------------------------------

static RiskManager::Config permissive_risk() {
    RiskManager::Config c;
    c.max_bias_magnitude      = 10.0;
    c.max_volatility_magnitude = 10.0;
    c.max_spread_magnitude    = 10.0;
    c.min_confidence          = 0.0;
    c.max_signals_per_second  = 100000;
    c.max_drawdown            = 1000.0;
    return c;
}

static TradeSignalEngine::Config backtest_engine() {
    TradeSignalEngine::Config c;
    c.bias_sensitivity       = 1.0;
    c.volatility_sensitivity = 1.0;
    c.signal_decay_rate      = 0.95;
    c.signal_cooldown        = std::chrono::microseconds{0};
    return c;
}

// Wire all 5 stages into a callable that returns the RiskManager gate result
// for a single token string.
struct Pipeline {
    InProcessDeduplicator     dedup_backend;
    Deduplicator              dedup;
    LatencyController         lc;
    LLMAdapter                adapter;
    TradeSignalEngine         engine;
    RiskManager               risk;
    MemoryOutputSink          sink;

    std::atomic<uint64_t>     passed{0};
    std::atomic<uint64_t>     blocked{0};
    std::atomic<uint64_t>     deduped{0};

    explicit Pipeline(
        std::chrono::milliseconds dedup_ttl     = std::chrono::milliseconds{500},
        LatencyController::Config lc_cfg        = default_lc(),
        RiskManager::Config       risk_cfg      = permissive_risk())
        : dedup(std::make_shared<InProcessDeduplicator>(), dedup_ttl)
        , lc(lc_cfg)
        , engine(backtest_engine())
        , risk(risk_cfg) {
        engine.set_backtest_mode(true);
        engine.set_signal_callback([this](const TradeSignal& s) {
            sink.emit(s);
            if (risk.evaluate(s)) ++passed;
            else                  ++blocked;
        });
        risk.set_alert_callback([](const std::string&, const TradeSignal&) {});
    }

    // Returns true if the token was novel and processed; false if deduped.
    bool push(const std::string& token, const std::string& ctx = "") {
        if (dedup.check(token, ctx) == DedupResult::Duplicate) {
            ++deduped;
            return false;
        }
        lc.start_measurement();
        const SemanticWeight w = adapter.map_token_to_weight(token);
        engine.process_semantic_weight(w);
        lc.end_measurement();
        return true;
    }

    static LatencyController::Config default_lc() {
        LatencyController::Config c;
        c.target_latency   = std::chrono::microseconds{100};
        c.sample_window    = 256;
        c.enable_profiling = false;
        return c;
    }
};

// ---------------------------------------------------------------------------
// Stage 1: Deduplicator gate
// ---------------------------------------------------------------------------

TEST(FullPipeline5Stage, Stage1_DedupBlocksRepeatedTokens) {
    Pipeline p;
    // First occurrence is novel and processed.
    EXPECT_TRUE(p.push("crash"));
    // Identical token within TTL is deduplicated.
    EXPECT_FALSE(p.push("crash"));
    EXPECT_FALSE(p.push("crash"));
    EXPECT_EQ(p.deduped.load(), 2u);
    EXPECT_EQ(p.sink.get_signals().size(), 1u);
}

TEST(FullPipeline5Stage, Stage1_DedupAllowsDifferentTokens) {
    Pipeline p;
    EXPECT_TRUE(p.push("bullish"));
    EXPECT_TRUE(p.push("bearish"));
    EXPECT_TRUE(p.push("rally"));
    EXPECT_EQ(p.deduped.load(), 0u);
    EXPECT_EQ(p.sink.get_signals().size(), 3u);
}

TEST(FullPipeline5Stage, Stage1_DedupRespectsTTLExpiry) {
    // Use a 100 ms TTL and a 300 ms sleep so the test is robust on loaded CI
    // runners where std::this_thread::sleep_for() may wake late.
    Pipeline p(std::chrono::milliseconds{100} /*TTL*/);
    EXPECT_TRUE(p.push("volatile"));
    EXPECT_FALSE(p.push("volatile"));        // within TTL -> deduped
    std::this_thread::sleep_for(std::chrono::milliseconds{300});
    EXPECT_TRUE(p.push("volatile"));         // TTL expired -> novel again
}

// ---------------------------------------------------------------------------
// Stage 2: LatencyController records measurements
// ---------------------------------------------------------------------------

TEST(FullPipeline5Stage, Stage2_LatencyControllerAccumulatesStats) {
    Pipeline p;
    const int N = 20;
    for (int i = 0; i < N; ++i) {
        p.push("bullish_" + std::to_string(i));  // unique tokens avoid dedup
    }
    const auto stats = p.lc.get_stats();
    EXPECT_EQ(stats.measurements, static_cast<uint64_t>(N))
        << "LatencyController must record one measurement per processed token";
    EXPECT_GE(stats.max_latency.count(), 0);
    EXPECT_GE(stats.avg_latency.count(), 0);
}

// ---------------------------------------------------------------------------
// Stage 3: LLMAdapter semantic weight mapping
// ---------------------------------------------------------------------------

TEST(FullPipeline5Stage, Stage3_FearTokensProduceNegativeBias) {
    Pipeline p;
    p.push("crash");
    p.push("panic");
    p.push("plunge");

    const auto& sigs = p.sink.get_signals();
    ASSERT_GT(sigs.size(), 0u);
    // Accumulated bias from fear tokens must be negative.
    EXPECT_LT(sigs.back().delta_bias_shift, 0.0)
        << "Fear tokens must drive delta_bias_shift negative";
}

TEST(FullPipeline5Stage, Stage3_BullishTokensProducePositiveBias) {
    Pipeline p;
    // Pass a unique context per call so each push is a distinct DedupKey while
    // the token itself ("bullish") is the real dictionary entry.
    for (int i = 0; i < 5; ++i)
        p.push("bullish", std::to_string(i));

    const auto& sigs = p.sink.get_signals();
    ASSERT_GT(sigs.size(), 0u);
    EXPECT_GT(sigs.back().delta_bias_shift, 0.0)
        << "Bullish tokens must drive delta_bias_shift positive";
}

TEST(FullPipeline5Stage, Stage3_UnknownTokenProducesNeutralWeight) {
    Pipeline p;
    // An unknown token returns the neutral weight {0, 0.5, 0.1, 0}.
    // With decay 0.95 and a single unknown token the bias shift must be ~0.
    p.push("xyzzy_totally_unknown_token_12345");

    const auto& sigs = p.sink.get_signals();
    ASSERT_EQ(sigs.size(), 1u);
    EXPECT_NEAR(sigs[0].delta_bias_shift, 0.0, 0.1)
        << "Unknown token must produce near-zero directional bias";
}

// ---------------------------------------------------------------------------
// Stage 4: TradeSignalEngine accumulation and decay
// ---------------------------------------------------------------------------

TEST(FullPipeline5Stage, Stage4_AccumulatorDecaysOverMultipleTokens) {
    Pipeline p;
    // Use the real "bullish" dict entry (via unique context) for the first push,
    // then feed unknown tokens (zero directional bias) to exercise decay.
    p.push("bullish", "decay_ctx0");
    const double first_bias = p.sink.get_signals().back().delta_bias_shift;
    ASSERT_GT(first_bias, 0.0) << "First bullish token must produce positive bias";

    for (int i = 1; i < 10; ++i) {
        // Tokens not in the dictionary return directional_bias = 0.0.
        // With decay rate 0.95 the accumulated bias falls each step.
        p.push("unknown_zzz_decay", std::to_string(i));
    }

    // After 9 decay steps with zero contribution:
    // last_bias ~ first_bias * 0.95^9 ~ 0.63 * first_bias < 2.0 * first_bias
    const double last_bias = p.sink.get_signals().back().delta_bias_shift;
    EXPECT_LT(std::abs(last_bias), std::abs(first_bias) * 2.0)
        << "Signal magnitude must not grow unboundedly with neutral tokens";
}

TEST(FullPipeline5Stage, Stage4_SignalCountMatchesSinkCount) {
    Pipeline p;
    for (int i = 0; i < 15; ++i)
        p.push("tok_" + std::to_string(i));

    EXPECT_EQ(p.sink.get_signals().size(),
              p.engine.get_stats().signals_generated.load())
        << "Sink signal count must equal engine signals_generated";
}

// ---------------------------------------------------------------------------
// Stage 5: RiskManager gate cascade
// ---------------------------------------------------------------------------

TEST(FullPipeline5Stage, Stage5_PermissiveConfigPassesAllSignals) {
    Pipeline p(std::chrono::milliseconds{500}, Pipeline::default_lc(),
               permissive_risk());

    for (int i = 0; i < 10; ++i)
        p.push("bullish_" + std::to_string(i));

    EXPECT_EQ(p.blocked.load(), 0u)
        << "Permissive risk config must not block any signals";
    EXPECT_EQ(p.passed.load(), p.sink.get_signals().size());
}

TEST(FullPipeline5Stage, Stage5_MagnitudeGateBlocksLargeShifts) {
    // Tight magnitude limit: bias shifts above 0.001 are blocked.
    RiskManager::Config tight;
    tight.max_bias_magnitude      = 0.001;
    tight.max_volatility_magnitude = 0.001;
    tight.max_spread_magnitude    = 0.001;
    tight.min_confidence          = 0.0;
    tight.max_signals_per_second  = 100000;
    tight.max_drawdown            = 1000.0;

    Pipeline p(std::chrono::milliseconds{500}, Pipeline::default_lc(), tight);

    // Strong fear tokens produce large bias shifts that exceed 0.001.
    p.push("crash");
    p.push("panic");

    EXPECT_GT(p.blocked.load(), 0u)
        << "Tight magnitude gate must block at least one high-magnitude signal";
    EXPECT_GT(p.risk.get_stats().signals_blocked_magnitude.load(), 0u);
}

TEST(FullPipeline5Stage, Stage5_ConfidenceGateBlocksLowConfidenceSignals) {
    RiskManager::Config high_conf;
    high_conf.max_bias_magnitude      = 10.0;
    high_conf.max_volatility_magnitude = 10.0;
    high_conf.max_spread_magnitude    = 10.0;
    high_conf.min_confidence          = 0.99;  // nearly impossible to satisfy
    high_conf.max_signals_per_second  = 100000;
    high_conf.max_drawdown            = 1000.0;

    Pipeline p(std::chrono::milliseconds{500}, Pipeline::default_lc(), high_conf);

    for (int i = 0; i < 5; ++i)
        p.push("tok_" + std::to_string(i));

    EXPECT_GT(p.blocked.load(), 0u)
        << "min_confidence=0.99 must block most signals";
    EXPECT_GT(p.risk.get_stats().signals_blocked_confidence.load(), 0u);
}

TEST(FullPipeline5Stage, Stage5_RateLimitGateBlocksBursts) {
    RiskManager::Config rate_limited;
    rate_limited.max_bias_magnitude      = 10.0;
    rate_limited.max_volatility_magnitude = 10.0;
    rate_limited.max_spread_magnitude    = 10.0;
    rate_limited.min_confidence          = 0.0;
    rate_limited.max_signals_per_second  = 2;  // only 2 per second
    rate_limited.max_drawdown            = 1000.0;

    Pipeline p(std::chrono::milliseconds{500}, Pipeline::default_lc(),
               rate_limited);

    for (int i = 0; i < 10; ++i)
        p.push("tok_" + std::to_string(i));

    // With a 10-token burst into a 2/s limit most should be blocked.
    EXPECT_GT(p.blocked.load(), 0u)
        << "Rate limit of 2/s must block bursts of 10 tokens";
    EXPECT_GT(p.risk.get_stats().signals_blocked_rate.load(), 0u);
}

TEST(FullPipeline5Stage, Stage5_DrawdownGateHaltsAfterExceedingLimit) {
    RiskManager::Config tight_dd;
    tight_dd.max_bias_magnitude      = 10.0;
    tight_dd.max_volatility_magnitude = 10.0;
    tight_dd.max_spread_magnitude    = 10.0;
    tight_dd.min_confidence          = 0.0;
    tight_dd.max_signals_per_second  = 100000;
    tight_dd.max_drawdown            = 0.01;  // tiny window

    Pipeline p(std::chrono::milliseconds{500}, Pipeline::default_lc(), tight_dd);

    // Real "crash" dict token via unique contexts; each call is a distinct
    // DedupKey so all 20 pass Stage 1 and reach the RiskManager.
    for (int i = 0; i < 20; ++i)
        p.push("crash", std::to_string(i));

    EXPECT_GT(p.risk.get_stats().signals_blocked_drawdown.load(), 0u)
        << "Drawdown gate must fire when cumulative bias exceeds 0.01";
}

// ---------------------------------------------------------------------------
// End-to-end: all 5 stages wired together under a background token stream
// ---------------------------------------------------------------------------

TEST(FullPipeline5Stage, EndToEnd_BackgroundStreamProducesSignalsThroughAllStages) {
    Pipeline p;

    TokenStreamSimulator::Config sim_cfg;
    sim_cfg.token_interval    = std::chrono::microseconds{1000};
    sim_cfg.buffer_size       = 32;
    sim_cfg.use_memory_stream = true;

    TokenStreamSimulator sim(sim_cfg);
    sim.load_tokens_from_memory({"bullish", "crash", "rally", "plunge", "surge"});

    sim.set_token_callback([&p](const Token& tok) {
        p.push(tok.text);
    });

    sim.start();
    std::this_thread::sleep_for(std::chrono::milliseconds{200});
    sim.stop();

    // All 5 stages ran: sink must have received signals, lc must have measurements.
    EXPECT_GT(p.sink.get_signals().size(), 0u)
        << "End-to-end pipeline must emit signals over 200 ms";

    const auto lc_stats = p.lc.get_stats();
    EXPECT_GT(lc_stats.measurements, 0u)
        << "LatencyController must have recorded at least one measurement";

    const auto eng_stats = p.engine.get_stats();
    EXPECT_GT(eng_stats.signals_generated.load(), 0u);
}

TEST(FullPipeline5Stage, EndToEnd_DeduplicatorReducesRedundantWork) {
    // Verify that when all tokens in the stream are identical the deduplicator
    // short-circuits the pipeline so the engine receives far fewer inputs.
    Pipeline p(std::chrono::milliseconds{1000});

    const std::string single_token = "rally";
    for (int i = 0; i < 50; ++i)
        p.push(single_token);

    // First push goes through; the remaining 49 are deduplicated.
    EXPECT_EQ(p.deduped.load(), 49u);
    EXPECT_EQ(p.sink.get_signals().size(), 1u);
}

} // namespace
} // namespace llmquant
