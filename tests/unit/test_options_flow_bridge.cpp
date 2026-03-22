#include <gtest/gtest.h>

#ifdef LLMQUANT_OPTIONS_FLOW_BRIDGE_ENABLED
#include "OptionsFlowSentimentBridge.h"
#include <atomic>
#include <thread>
using namespace llmquant;

TEST(OptionsFlowBridgeTest, InitialState) {
    OptionsFlowSentimentBridge b;
    EXPECT_NEAR(b.sentiment_velocity_ema(), 0.0, 1e-9);
    EXPECT_NEAR(b.skew_ema(), 0.0, 1e-9);
    EXPECT_NEAR(b.divergence_score(), 0.0, 1e-9);
    EXPECT_EQ(b.divergence_count(), 0u);
    EXPECT_EQ(b.last_divergence(), OptionsFlowSentimentBridge::DivergenceKind::None);
}

TEST(OptionsFlowBridgeTest, SkewEmaUpdates) {
    OptionsFlowSentimentBridge b;
    b.record_skew(0.30, 0.20);  // skew = 0.10
    EXPECT_GT(b.skew_ema(), 0.0);
    b.record_skew(0.20, 0.30);  // skew = -0.10
    // After negative skew, EMA should decrease
    EXPECT_LT(b.skew_ema(), 0.02);
}

TEST(OptionsFlowBridgeTest, VelocityEmaUpdates) {
    OptionsFlowSentimentBridge::Config cfg;
    cfg.min_warmup = 2;
    OptionsFlowSentimentBridge b(cfg);
    b.record_bias(0.0, 1.0);
    b.record_bias(1.0, 1.0);  // large upward move
    EXPECT_GT(b.sentiment_velocity_ema(), 0.0);
}

TEST(OptionsFlowBridgeTest, BearDivergenceDetected) {
    OptionsFlowSentimentBridge::Config cfg;
    cfg.min_warmup    = 3;
    cfg.div_threshold = 0.001;
    cfg.velocity_alpha = 0.9;
    cfg.skew_alpha     = 0.9;
    std::atomic<int> fires{0};
    OptionsFlowSentimentBridge::DivergenceKind last_kind =
        OptionsFlowSentimentBridge::DivergenceKind::None;
    cfg.on_divergence = [&](OptionsFlowSentimentBridge::DivergenceKind k,
                             double, double, double) {
        if (k != OptionsFlowSentimentBridge::DivergenceKind::None) {
            ++fires;
            last_kind = k;
        }
    };
    OptionsFlowSentimentBridge b(cfg);

    // Rapidly increasing bias (bullish sentiment velocity)
    b.record_bias(0.0, 1.0);
    b.record_skew(0.40, 0.10); // wide put premium → bearish hedge
    b.record_bias(0.5, 1.0);
    b.record_skew(0.45, 0.10);
    b.record_bias(1.0, 1.0);
    b.record_skew(0.50, 0.10);
    // Drive velocity up more
    for (int i = 0; i < 10; ++i) {
        b.record_bias(1.0 + i * 0.2, 1.0);
        b.record_skew(0.50, 0.05);
    }
    EXPECT_GE(fires.load(), 1);
    EXPECT_EQ(last_kind, OptionsFlowSentimentBridge::DivergenceKind::SmartMoneyBear);
}

TEST(OptionsFlowBridgeTest, BullDivergenceDetected) {
    OptionsFlowSentimentBridge::Config cfg;
    cfg.min_warmup    = 3;
    cfg.div_threshold = 0.001;
    cfg.velocity_alpha = 0.9;
    cfg.skew_alpha     = 0.9;
    std::atomic<int> fires{0};
    OptionsFlowSentimentBridge::DivergenceKind last_kind =
        OptionsFlowSentimentBridge::DivergenceKind::None;
    cfg.on_divergence = [&](OptionsFlowSentimentBridge::DivergenceKind k,
                             double, double, double) {
        if (k == OptionsFlowSentimentBridge::DivergenceKind::SmartMoneyBull) ++fires;
        last_kind = k;
    };
    OptionsFlowSentimentBridge b(cfg);

    // Decreasing bias (bearish velocity), calls bid (negative skew)
    b.record_bias(1.0, 1.0);
    b.record_skew(0.10, 0.40); // negative skew
    b.record_bias(0.5, 1.0);
    b.record_skew(0.05, 0.45);
    b.record_bias(0.0, 1.0);
    b.record_skew(0.05, 0.50);
    for (int i = 0; i < 10; ++i) {
        b.record_bias(-0.2 * i, 1.0);
        b.record_skew(0.05, 0.50);
    }
    EXPECT_GE(fires.load(), 1);
}

TEST(OptionsFlowBridgeTest, ResetClearsState) {
    OptionsFlowSentimentBridge b;
    b.record_bias(0.5, 1.0);
    b.record_skew(0.3, 0.1);
    b.reset();
    EXPECT_NEAR(b.skew_ema(), 0.0, 1e-9);
    EXPECT_EQ(b.divergence_count(), 0u);
}

TEST(OptionsFlowBridgeTest, StatsJsonNonEmpty) {
    OptionsFlowSentimentBridge b;
    b.record_bias(0.1, 1.0);
    std::string j = b.to_stats_json();
    EXPECT_NE(j.find("sentiment_velocity_ema"), std::string::npos);
    EXPECT_NE(j.find("divergence_count"), std::string::npos);
}

TEST(OptionsFlowBridgeTest, ZeroDtIgnored) {
    OptionsFlowSentimentBridge b;
    b.record_bias(0.0, 0.0);  // dt = 0, should not crash or update velocity
    b.record_bias(1.0, 0.0);
    EXPECT_NEAR(b.sentiment_velocity_ema(), 0.0, 1e-9);
}

TEST(OptionsFlowBridgeTest, ConcurrentSafe) {
    OptionsFlowSentimentBridge b;
    std::atomic<bool> go{false};
    auto worker_bias = [&] {
        while (!go.load()) {}
        for (int i = 0; i < 200; ++i) b.record_bias(i * 0.01, 0.1);
    };
    auto worker_skew = [&] {
        while (!go.load()) {}
        for (int i = 0; i < 200; ++i) b.record_skew(0.2 + i * 0.001, 0.1);
    };
    std::thread t1(worker_bias), t2(worker_skew);
    go.store(true);
    t1.join(); t2.join();
    // No crash = success
}

#else
TEST(OptionsFlowBridgeTest, DisabledAtBuildTime) { SUCCEED(); }
#endif
