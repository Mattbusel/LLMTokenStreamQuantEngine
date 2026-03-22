#include <gtest/gtest.h>

#ifdef LLMQUANT_BIAS_HYSTERESIS_ENABLED

#include "BiasHysteresisGate.h"
#include <atomic>
#include <thread>
#include <vector>

using namespace llmquant;

namespace {

TEST(BiasHysteresisGate, DefaultState) {
    BiasHysteresisGate g;
    EXPECT_FALSE(g.is_open());
    EXPECT_EQ(g.consecutive_ticks(), 0);
    EXPECT_EQ(g.total_evaluated(), 0u);
    EXPECT_EQ(g.gate_open_events(), 0u);
    EXPECT_EQ(g.gate_close_events(), 0u);
}

// Gate opens after required_ticks consecutive exceedances.
TEST(BiasHysteresisGate, OpensAfterRequiredTicks) {
    BiasHysteresisGate::Config cfg;
    cfg.enter_threshold = 0.1;
    cfg.exit_threshold  = 0.05;
    cfg.required_ticks  = 3;
    BiasHysteresisGate g(cfg);

    EXPECT_FALSE(g.evaluate(0.2));  // tick 1 — not open yet
    EXPECT_FALSE(g.evaluate(0.2));  // tick 2 — not open yet
    EXPECT_TRUE(g.evaluate(0.2));   // tick 3 — gate opens
    EXPECT_TRUE(g.is_open());
    EXPECT_EQ(g.gate_open_events(), 1u);
}

// A dip below threshold before reaching required_ticks resets the counter.
TEST(BiasHysteresisGate, DipResetsCounter) {
    BiasHysteresisGate::Config cfg;
    cfg.enter_threshold = 0.1;
    cfg.exit_threshold  = 0.05;
    cfg.required_ticks  = 3;
    BiasHysteresisGate g(cfg);

    g.evaluate(0.2);   // tick 1
    g.evaluate(0.2);   // tick 2
    g.evaluate(0.01);  // dip — resets counter
    EXPECT_EQ(g.consecutive_ticks(), 0);
    EXPECT_FALSE(g.is_open());
    g.evaluate(0.2);   // tick 1 again
    EXPECT_FALSE(g.is_open());
}

// Once open, gate stays open until bias drops below exit_threshold.
TEST(BiasHysteresisGate, HysteresisKeepsGateOpen) {
    BiasHysteresisGate::Config cfg;
    cfg.enter_threshold = 0.1;
    cfg.exit_threshold  = 0.05;
    cfg.required_ticks  = 2;
    BiasHysteresisGate g(cfg);

    g.evaluate(0.2); g.evaluate(0.2);  // open
    ASSERT_TRUE(g.is_open());

    // Bias between exit and enter thresholds — gate stays open.
    EXPECT_TRUE(g.evaluate(0.07));
    EXPECT_TRUE(g.evaluate(0.08));
    EXPECT_TRUE(g.is_open());
}

// Gate closes when bias drops below exit_threshold.
TEST(BiasHysteresisGate, ClosesWhenBelowExitThreshold) {
    BiasHysteresisGate::Config cfg;
    cfg.enter_threshold = 0.1;
    cfg.exit_threshold  = 0.05;
    cfg.required_ticks  = 2;
    std::atomic<int> closes{0};
    cfg.on_gate_close = [&](double) { ++closes; };
    BiasHysteresisGate g(cfg);

    g.evaluate(0.2); g.evaluate(0.2);  // open
    ASSERT_TRUE(g.is_open());

    EXPECT_FALSE(g.evaluate(0.02));  // |0.02| < 0.05 — gate closes
    EXPECT_FALSE(g.is_open());
    EXPECT_EQ(g.gate_close_events(), 1u);
    EXPECT_GT(closes.load(), 0);
}

// on_gate_open callback fires exactly once when gate opens.
TEST(BiasHysteresisGate, OpenCallbackFires) {
    BiasHysteresisGate::Config cfg;
    cfg.enter_threshold = 0.1;
    cfg.required_ticks  = 1;
    std::atomic<int> fires{0};
    cfg.on_gate_open = [&](double) { ++fires; };
    BiasHysteresisGate g(cfg);

    g.evaluate(0.5);
    EXPECT_EQ(fires.load(), 1);
    g.evaluate(0.5);  // already open — no second fire
    EXPECT_EQ(fires.load(), 1);
}

// Negative bias is handled by absolute value.
TEST(BiasHysteresisGate, NegativeBiasTreatedByAbsValue) {
    BiasHysteresisGate::Config cfg;
    cfg.enter_threshold = 0.1;
    cfg.required_ticks  = 2;
    BiasHysteresisGate g(cfg);

    g.evaluate(-0.5);  // |-0.5| >= 0.1 — tick 1
    EXPECT_TRUE(g.evaluate(-0.5));  // tick 2 — opens
}

// Reset clears all state.
TEST(BiasHysteresisGate, ResetClearsState) {
    BiasHysteresisGate::Config cfg;
    cfg.enter_threshold = 0.1;
    cfg.required_ticks  = 1;
    BiasHysteresisGate g(cfg);
    g.evaluate(0.5);
    ASSERT_TRUE(g.is_open());

    g.reset();
    EXPECT_FALSE(g.is_open());
    EXPECT_EQ(g.consecutive_ticks(), 0);
    EXPECT_EQ(g.total_evaluated(), 0u);
    EXPECT_EQ(g.gate_open_events(), 0u);
}

// update_config resets state.
TEST(BiasHysteresisGate, UpdateConfigResetsState) {
    BiasHysteresisGate::Config cfg;
    cfg.enter_threshold = 0.1;
    cfg.required_ticks  = 1;
    BiasHysteresisGate g(cfg);
    g.evaluate(0.5);
    ASSERT_TRUE(g.is_open());

    BiasHysteresisGate::Config cfg2;
    cfg2.required_ticks = 5;
    g.update_config(cfg2);
    EXPECT_FALSE(g.is_open());
    EXPECT_EQ(g.total_evaluated(), 0u);
}

// to_stats_json contains expected fields.
TEST(BiasHysteresisGate, StatsJsonContainsFields) {
    BiasHysteresisGate g;
    g.evaluate(0.5);
    std::string j = g.to_stats_json();
    EXPECT_NE(j.find("is_open"),          std::string::npos);
    EXPECT_NE(j.find("consecutive_ticks"), std::string::npos);
    EXPECT_NE(j.find("total_evaluated"),   std::string::npos);
    EXPECT_NE(j.find("gate_open_events"),  std::string::npos);
}

// Gate can open, close, and open again.
TEST(BiasHysteresisGate, MultipleOpenCloseEvents) {
    BiasHysteresisGate::Config cfg;
    cfg.enter_threshold = 0.1;
    cfg.exit_threshold  = 0.02;
    cfg.required_ticks  = 1;
    BiasHysteresisGate g(cfg);

    g.evaluate(0.5);   // open
    g.evaluate(0.01);  // close
    g.evaluate(0.5);   // open again
    g.evaluate(0.01);  // close again

    EXPECT_EQ(g.gate_open_events(),  2u);
    EXPECT_EQ(g.gate_close_events(), 2u);
}

// Thread-safe concurrent evaluate().
TEST(BiasHysteresisGate, ConcurrentEvaluateSafe) {
    BiasHysteresisGate::Config cfg;
    cfg.enter_threshold = 0.05;
    cfg.required_ticks  = 1;
    BiasHysteresisGate g(cfg);

    std::atomic<bool> go{false};
    auto worker = [&] {
        while (!go.load()) {}
        for (int i = 0; i < 500; ++i)
            (void)g.evaluate(static_cast<double>(i % 3) * 0.1 - 0.05);
    };
    std::vector<std::thread> threads;
    for (int t = 0; t < 4; ++t) threads.emplace_back(worker);
    go.store(true);
    for (auto& t : threads) t.join();

    EXPECT_EQ(g.total_evaluated(), 2000u);
}

}  // namespace

#else

TEST(BiasHysteresisGate, DisabledAtBuildTime) { SUCCEED(); }

#endif  // LLMQUANT_BIAS_HYSTERESIS_ENABLED
