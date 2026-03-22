#include <gtest/gtest.h>

#ifdef LLMQUANT_CAUSAL_TRACER_ENABLED

#include "SignalCausalChainTracer.h"
#include <atomic>
#include <cmath>
#include <thread>
#include <vector>

using namespace llmquant;

namespace {

TEST(SignalCausalChainTracer, DefaultState) {
    SignalCausalChainTracer tracer;
    EXPECT_EQ(tracer.total_tokens(), 0u);
    EXPECT_EQ(tracer.strong_token_events(), 0u);
    EXPECT_DOUBLE_EQ(tracer.max_contribution(), 0.0);
    EXPECT_TRUE(tracer.trace().empty());
}

// Total tokens increments correctly.
TEST(SignalCausalChainTracer, TotalTokensIncrements) {
    SignalCausalChainTracer tracer;
    tracer.record_token("bullish", 0.1);
    tracer.record_token("sell",   -0.2);
    tracer.record_token("hold",    0.0);
    EXPECT_EQ(tracer.total_tokens(), 3u);
}

// trace() returns top-N by absolute contribution.
TEST(SignalCausalChainTracer, TraceReturnsTopN) {
    SignalCausalChainTracer::Config cfg;
    cfg.window = 16;
    cfg.strong_threshold = 10.0;  // suppress strong-token events
    SignalCausalChainTracer tracer(cfg);

    tracer.record_token("a", 0.1);
    tracer.record_token("b", 0.5);
    tracer.record_token("c", 0.3);
    tracer.record_token("d", 0.9);
    tracer.record_token("e", 0.2);

    auto top2 = tracer.trace(2);
    ASSERT_EQ(top2.size(), 2u);
    EXPECT_EQ(top2[0].token, "d");  // 0.9 is largest
    EXPECT_EQ(top2[1].token, "b");  // 0.5 is second
}

// trace() handles negative contributions via absolute value.
TEST(SignalCausalChainTracer, TraceRankedByAbsoluteValue) {
    SignalCausalChainTracer::Config cfg;
    cfg.window = 8;
    cfg.strong_threshold = 10.0;
    SignalCausalChainTracer tracer(cfg);

    tracer.record_token("pos", 0.3);
    tracer.record_token("neg", -0.8);

    auto top1 = tracer.trace(1);
    ASSERT_EQ(top1.size(), 1u);
    EXPECT_EQ(top1[0].token, "neg");  // |-0.8| > |0.3|
}

// Window eviction: oldest tokens are discarded when window fills.
TEST(SignalCausalChainTracer, WindowEvictsOldTokens) {
    SignalCausalChainTracer::Config cfg;
    cfg.window = 4;
    cfg.strong_threshold = 10.0;
    SignalCausalChainTracer tracer(cfg);

    // Fill window with high contributions.
    for (int i = 0; i < 4; ++i)
        tracer.record_token("old", 0.9);
    // Overwrite with low contributions.
    for (int i = 0; i < 4; ++i)
        tracer.record_token("new", 0.01);

    auto snap = tracer.window_snapshot();
    for (const auto& entry : snap)
        EXPECT_EQ(entry.token, "new");
}

// on_strong_token fires when |contribution| >= strong_threshold.
TEST(SignalCausalChainTracer, StrongTokenCallbackFires) {
    SignalCausalChainTracer::Config cfg;
    cfg.strong_threshold = 0.5;
    std::atomic<int> fires{0};
    cfg.on_strong_token = [&](const std::string&, double) { ++fires; };
    SignalCausalChainTracer tracer(cfg);

    tracer.record_token("weak",   0.1);   // no fire
    tracer.record_token("strong", 0.6);   // fires
    tracer.record_token("strong", -0.8);  // fires

    EXPECT_EQ(fires.load(), 2);
    EXPECT_EQ(tracer.strong_token_events(), 2u);
}

// max_contribution tracks the largest |contribution| in the current window.
TEST(SignalCausalChainTracer, MaxContributionTracked) {
    SignalCausalChainTracer::Config cfg;
    cfg.window = 8;
    cfg.strong_threshold = 10.0;
    SignalCausalChainTracer tracer(cfg);

    tracer.record_token("a", 0.3);
    tracer.record_token("b", 0.7);
    tracer.record_token("c", -0.5);

    EXPECT_NEAR(tracer.max_contribution(), 0.7, 1e-9);
}

// window_snapshot returns temporal order.
TEST(SignalCausalChainTracer, WindowSnapshotTemporalOrder) {
    SignalCausalChainTracer::Config cfg;
    cfg.window = 4;
    cfg.strong_threshold = 10.0;
    SignalCausalChainTracer tracer(cfg);

    tracer.record_token("first",  0.1);
    tracer.record_token("second", 0.2);
    tracer.record_token("third",  0.3);

    auto snap = tracer.window_snapshot();
    ASSERT_EQ(snap.size(), 3u);
    EXPECT_EQ(snap[0].token, "first");
    EXPECT_EQ(snap[1].token, "second");
    EXPECT_EQ(snap[2].token, "third");
}

// Reset clears all state.
TEST(SignalCausalChainTracer, ResetClearsState) {
    SignalCausalChainTracer tracer;
    tracer.record_token("x", 0.9);
    tracer.reset();
    EXPECT_EQ(tracer.total_tokens(), 0u);
    EXPECT_EQ(tracer.strong_token_events(), 0u);
    EXPECT_DOUBLE_EQ(tracer.max_contribution(), 0.0);
    EXPECT_TRUE(tracer.trace().empty());
}

// update_config resets state.
TEST(SignalCausalChainTracer, UpdateConfigResetsState) {
    SignalCausalChainTracer tracer;
    tracer.record_token("x", 0.5);
    SignalCausalChainTracer::Config cfg2;
    cfg2.window = 8;
    tracer.update_config(cfg2);
    EXPECT_EQ(tracer.total_tokens(), 0u);
    EXPECT_TRUE(tracer.trace().empty());
}

// to_stats_json contains expected fields.
TEST(SignalCausalChainTracer, StatsJsonContainsFields) {
    SignalCausalChainTracer tracer;
    tracer.record_token("tok", 0.3);
    std::string j = tracer.to_stats_json();
    EXPECT_NE(j.find("total_tokens"),        std::string::npos);
    EXPECT_NE(j.find("strong_token_events"), std::string::npos);
    EXPECT_NE(j.find("max_contribution"),    std::string::npos);
    EXPECT_NE(j.find("top_tokens"),          std::string::npos);
}

// Concurrent record_token() is thread-safe.
TEST(SignalCausalChainTracer, ConcurrentRecordSafe) {
    SignalCausalChainTracer::Config cfg;
    cfg.window = 64;
    cfg.strong_threshold = 0.5;
    SignalCausalChainTracer tracer(cfg);

    std::atomic<bool> go{false};
    auto worker = [&](int id) {
        std::string tok = "t" + std::to_string(id);
        while (!go.load()) {}
        for (int i = 0; i < 250; ++i)
            tracer.record_token(tok, static_cast<double>(i % 7) * 0.1 - 0.3);
    };
    std::vector<std::thread> threads;
    for (int t = 0; t < 4; ++t) threads.emplace_back(worker, t);
    go.store(true);
    for (auto& t : threads) t.join();

    EXPECT_EQ(tracer.total_tokens(), 1000u);
}

}  // namespace

#else

TEST(SignalCausalChainTracer, DisabledAtBuildTime) { SUCCEED(); }

#endif  // LLMQUANT_CAUSAL_TRACER_ENABLED
