/// Property / invariant tests for algorithmic components.
///
/// These tests enforce determinism and boundary invariants that must hold for
/// all inputs, mirroring the role of property-based testing without an
/// external proptest library.  A hand-crafted vocabulary and parameter sweep
/// is used instead of a random generator.

#include "gtest/gtest.h"
#include "LLMAdapter.h"
#include "Deduplicator.h"
#include "RiskManager.h"
#include "TradeSignalEngine.h"
#include "LatencyController.h"

#include <cmath>
#include <limits>
#include <memory>
#include <string>
#include <vector>

using namespace llmquant;

// ---------------------------------------------------------------------------
// Helper: build a signal with explicit fields.
// ---------------------------------------------------------------------------

namespace {

static TradeSignal make_signal(double bias, double vol, double spread, double conf,
                               uint64_t ts = 1) {
    TradeSignal s;
    s.delta_bias_shift      = bias;
    s.volatility_adjustment = vol;
    s.spread_modifier       = spread;
    s.confidence            = conf;
    s.timestamp_ns          = ts;
    return s;
}

[[maybe_unused]] static RiskManager::Config permissive_config() {
    RiskManager::Config cfg;
    cfg.max_bias_magnitude       = 100.0;
    cfg.max_volatility_magnitude = 100.0;
    cfg.max_spread_magnitude     = 100.0;
    cfg.min_confidence           = 0.0;
    cfg.max_signals_per_second   = 1000000;
    cfg.max_drawdown             = 10000.0;
    cfg.drawdown_window          = std::chrono::seconds{3600};
    cfg.position_warn_fraction   = 0.8;
    return cfg;
}

static TradeSignalEngine::Config backtest_cfg() {
    TradeSignalEngine::Config cfg;
    cfg.bias_sensitivity       = 1.0;
    cfg.volatility_sensitivity = 1.0;
    cfg.signal_decay_rate      = 0.95;
    cfg.signal_cooldown        = std::chrono::microseconds{0};
    return cfg;
}

} // namespace

// ---------------------------------------------------------------------------
// Test 1: DedupKey is deterministic for every token in a 30-token vocabulary.
// ---------------------------------------------------------------------------

TEST(Invariants, test_invariant_dedup_key_deterministic_for_all_vocab) {
    // 30-token vocabulary covering diverse strings including Unicode-adjacent
    // ASCII, numbers, and known financial tokens.
    static const std::vector<std::string> vocab = {
        "bullish", "bearish", "crash", "rally", "surge", "panic",
        "neutral", "stable", "volatile", "plunge", "moon", "dump",
        "buy",     "sell",   "hold",   "long",   "short", "hedge",
        "alpha",   "beta",   "gamma",  "delta",  "theta", "vega",
        "0",       "1",      "99",     "3.14",   " ",     ""
    };

    for (const auto& token : vocab) {
        DedupKey k1 = DedupKey::from_token(token);
        DedupKey k2 = DedupKey::from_token(token);

        EXPECT_EQ(k1.value, k2.value)
            << "DedupKey must be deterministic for token='" << token << "'";

        // Key with context must also be deterministic.
        DedupKey kc1 = DedupKey::from_token(token, "ctx");
        DedupKey kc2 = DedupKey::from_token(token, "ctx");
        EXPECT_EQ(kc1.value, kc2.value)
            << "DedupKey with context must be deterministic for token='"
            << token << "'";

        // Token-only key and token+context key must differ (context changes hash).
        if (!token.empty()) {
            EXPECT_NE(k1.value, kc1.value)
                << "Key with empty context must differ from key with 'ctx'";
        }
    }
}

// ---------------------------------------------------------------------------
// Test 2: LLMAdapter sentiment sign invariants for known token categories.
// ---------------------------------------------------------------------------

TEST(Invariants, test_invariant_llm_adapter_known_tokens_sentiment_sign) {
    LLMAdapter adapter;

    // Fear / bearish tokens must map to negative sentiment and negative bias.
    static const std::vector<std::string> fear_tokens = {
        "crash", "panic", "bearish", "plunge"
    };
    for (const auto& tok : fear_tokens) {
        SemanticWeight w = adapter.map_token_to_weight(tok);
        EXPECT_LT(w.sentiment_score, 0.0)
            << "Fear token '" << tok << "' must have negative sentiment_score";
        EXPECT_LT(w.directional_bias, 0.0)
            << "Fear token '" << tok << "' must have negative directional_bias";
    }

    // Bullish tokens must map to positive directional_bias.
    static const std::vector<std::string> bullish_tokens = {
        "bullish", "rally", "surge"
    };
    for (const auto& tok : bullish_tokens) {
        SemanticWeight w = adapter.map_token_to_weight(tok);
        EXPECT_GT(w.directional_bias, 0.0)
            << "Bullish token '" << tok << "' must have positive directional_bias";
    }

    // All weights must have confidence_score in [0, 1].
    static const std::vector<std::string> all_known = {
        "crash", "panic", "bearish", "plunge", "bullish", "rally", "surge",
        "neutral", "stable", "volatile"
    };
    for (const auto& tok : all_known) {
        SemanticWeight w = adapter.map_token_to_weight(tok);
        EXPECT_GE(w.confidence_score, 0.0)
            << "Token '" << tok << "' confidence_score must be >= 0";
        EXPECT_LE(w.confidence_score, 1.0)
            << "Token '" << tok << "' confidence_score must be <= 1";
    }
}

// ---------------------------------------------------------------------------
// Test 3: passed + all_blocked == total_calls for RiskManager.
// ---------------------------------------------------------------------------

TEST(Invariants, test_invariant_risk_manager_passed_plus_blocked_equals_total) {
    // Use a mix of passing and blocking signals over 50 calls.
    RiskManager::Config cfg;
    cfg.max_bias_magnitude       = 0.5;   // tight — some signals will be blocked
    cfg.max_volatility_magnitude = 1.0;
    cfg.max_spread_magnitude     = 0.5;
    cfg.min_confidence           = 0.2;   // some low-confidence signals blocked
    cfg.max_signals_per_second   = 1000000;
    cfg.max_drawdown             = 100.0;
    cfg.drawdown_window          = std::chrono::seconds{3600};
    cfg.position_warn_fraction   = 0.8;

    RiskManager rm(cfg);

    // Mix of signals: varying bias and confidence to exercise multiple paths.
    const std::vector<TradeSignal> signals = {
        make_signal( 0.1, 0.1, 0.1,  0.8),   // passes all
        make_signal( 0.1, 0.1, 0.1,  0.8),
        make_signal( 0.1, 0.1, 0.1,  0.8),
        make_signal( 0.8, 0.1, 0.1,  0.8),   // bias blocked
        make_signal( 0.8, 0.1, 0.1,  0.8),
        make_signal( 0.1, 0.1, 0.1,  0.05),  // confidence blocked
        make_signal( 0.1, 0.1, 0.1,  0.05),
        make_signal( 0.1, 0.1, 0.1,  0.8),
        make_signal(-0.8, 0.1, 0.1,  0.8),   // negative bias blocked
        make_signal( 0.1, 0.1, 0.7,  0.8),   // spread blocked
    };

    const uint64_t total_calls = static_cast<uint64_t>(signals.size());
    for (const auto& s : signals) {
        rm.evaluate(s);
    }

    const auto& st = rm.get_stats();
    uint64_t total_accounted =
        st.signals_passed.load()           +
        st.signals_blocked_magnitude.load() +
        st.signals_blocked_confidence.load() +
        st.signals_blocked_rate.load()      +
        st.signals_blocked_drawdown.load()  +
        st.signals_blocked_position.load();

    EXPECT_EQ(total_accounted, total_calls)
        << "passed + all blocked counters must equal total calls to evaluate()";
}

// ---------------------------------------------------------------------------
// Test 4: LatencyController avg is always between min and max.
// ---------------------------------------------------------------------------

TEST(Invariants, test_invariant_latency_controller_avg_between_min_and_max) {
    LatencyController::Config cfg;
    cfg.target_latency   = std::chrono::microseconds{100};
    cfg.sample_window    = 200;
    cfg.enable_profiling = true;
    LatencyController lc(cfg);

    // Record 100 latency values spanning a range.
    const std::vector<int64_t> latencies_us = {
         1,  2,  3,  4,  5,  6,  7,  8,  9, 10,
        11, 12, 13, 14, 15, 16, 17, 18, 19, 20,
        21, 22, 23, 24, 25, 26, 27, 28, 29, 30,
        31, 32, 33, 34, 35, 36, 37, 38, 39, 40,
        41, 42, 43, 44, 45, 46, 47, 48, 49, 50,
        51, 52, 53, 54, 55, 56, 57, 58, 59, 60,
        61, 62, 63, 64, 65, 66, 67, 68, 69, 70,
        71, 72, 73, 74, 75, 76, 77, 78, 79, 80,
        81, 82, 83, 84, 85, 86, 87, 88, 89, 90,
        91, 92, 93, 94, 95, 96, 97, 98, 99, 100
    };

    for (auto us : latencies_us) {
        lc.record_latency(std::chrono::microseconds{us});
    }

    auto stats = lc.get_stats();

    EXPECT_EQ(stats.measurements, static_cast<uint64_t>(latencies_us.size()));
    EXPECT_GE(stats.avg_latency.count(), stats.min_latency.count())
        << "avg must be >= min";
    EXPECT_LE(stats.avg_latency.count(), stats.max_latency.count())
        << "avg must be <= max";
    EXPECT_GE(stats.min_latency.count(), 0)
        << "min latency must be non-negative";
}

// ---------------------------------------------------------------------------
// Test 5: all emitted TradeSignal confidence values are in [0.0, 1.0].
// ---------------------------------------------------------------------------

TEST(Invariants, test_invariant_trade_signal_confidence_in_unit_interval) {
    LLMAdapter adapter;
    TradeSignalEngine engine(backtest_cfg());
    engine.set_backtest_mode(true);

    std::vector<TradeSignal> signals;
    engine.set_signal_callback([&](const TradeSignal& s) {
        signals.push_back(s);
    });

    // Process 20 varied tokens from the known dictionary.
    const std::vector<std::string> tokens = {
        "crash", "bullish", "panic", "rally", "neutral",
        "surge", "plunge",  "stable", "volatile", "bearish",
        "crash", "bullish", "panic", "rally", "neutral",
        "surge", "plunge",  "stable", "volatile", "bearish"
    };

    for (const auto& tok : tokens) {
        SemanticWeight w = adapter.map_token_to_weight(tok);
        engine.process_semantic_weight(w);
    }

    ASSERT_GT(signals.size(), 0u)
        << "Engine must emit at least one signal in backtest mode for 20 tokens";

    for (const auto& sig : signals) {
        EXPECT_GE(sig.confidence, 0.0)
            << "TradeSignal::confidence must be >= 0.0";
        EXPECT_LE(sig.confidence, 1.0)
            << "TradeSignal::confidence must be <= 1.0";
    }
}

// ---------------------------------------------------------------------------
// Test 6: total_novel + total_duplicates == total_checks for Deduplicator.
// ---------------------------------------------------------------------------

TEST(Invariants, test_invariant_dedup_total_novel_plus_duplicate_equals_total_checks) {
    auto backend = std::make_shared<InProcessDeduplicator>();
    Deduplicator dedup(backend, std::chrono::milliseconds{5000});

    // 50 unique tokens followed by 50 repeats of the same 50 tokens.
    std::vector<std::string> unique_tokens;
    for (int i = 0; i < 50; ++i) {
        unique_tokens.push_back("token_" + std::to_string(i));
    }

    // First pass: all 50 must be novel.
    for (const auto& tok : unique_tokens) {
        dedup.check(tok);
    }

    // Second pass: all 50 must be duplicates (TTL = 5 s, checks are immediate).
    for (const auto& tok : unique_tokens) {
        dedup.check(tok);
    }

    const uint64_t total_checks = 100;
    uint64_t novel      = backend->total_novel();
    uint64_t duplicates = backend->total_duplicates();

    EXPECT_EQ(novel + duplicates, total_checks)
        << "total_novel + total_duplicates must equal 100";
    EXPECT_EQ(novel, 50u)
        << "First 50 unique checks must all be novel";
    EXPECT_EQ(duplicates, 50u)
        << "Second 50 repeated checks must all be duplicates";
}