#include <gtest/gtest.h>

#ifdef LLMQUANT_ORDER_FLOW_IMBALANCE_ENABLED

#include "OrderFlowImbalanceDetector.h"
#include <atomic>
#include <cmath>
#include <string>

using namespace llmquant;

namespace {

// ============================================================
// Default construction
// ============================================================

TEST(OrderFlowImbalanceDetector, DefaultConstructs) {
    OrderFlowImbalanceDetector det;
    EXPECT_NEAR(det.imbalance(),    0.0, 1e-9);
    EXPECT_EQ(det.total_tokens(),   0u);
    EXPECT_EQ(det.imbalance_events(), 0u);
    EXPECT_FALSE(det.is_buy_imbalance());
    EXPECT_FALSE(det.is_sell_imbalance());
}

// ============================================================
// Buy tokens drive imbalance positive
// ============================================================

TEST(OrderFlowImbalanceDetector, BuyTokensDrivePositiveImbalance) {
    OrderFlowImbalanceDetector::Config cfg;
    cfg.ema_alpha   = 0.5;  // fast response
    cfg.min_tokens  = 1;
    OrderFlowImbalanceDetector det(cfg);

    for (int i = 0; i < 20; ++i) det.record("buyers");

    EXPECT_GT(det.imbalance(), 0.0);
}

// ============================================================
// Sell tokens drive imbalance negative
// ============================================================

TEST(OrderFlowImbalanceDetector, SellTokensDriveNegativeImbalance) {
    OrderFlowImbalanceDetector::Config cfg;
    cfg.ema_alpha  = 0.5;
    cfg.min_tokens = 1;
    OrderFlowImbalanceDetector det(cfg);

    for (int i = 0; i < 20; ++i) det.record("sellers");

    EXPECT_LT(det.imbalance(), 0.0);
}

// ============================================================
// Neutral tokens don't push imbalance
// ============================================================

TEST(OrderFlowImbalanceDetector, NeutralTokensNoEffect) {
    OrderFlowImbalanceDetector det;
    for (int i = 0; i < 50; ++i) det.record("the");
    // Neutral token → imbalance stays at 0
    EXPECT_NEAR(det.imbalance(), 0.0, 1e-9);
}

// ============================================================
// is_buy_imbalance fires at threshold
// ============================================================

TEST(OrderFlowImbalanceDetector, IsBuyImbalanceAboveThreshold) {
    OrderFlowImbalanceDetector::Config cfg;
    cfg.ema_alpha            = 0.9;
    cfg.imbalance_threshold  = 0.5;
    cfg.min_tokens           = 1;
    OrderFlowImbalanceDetector det(cfg);

    // Record enough buys to saturate EMA > 0.5
    for (int i = 0; i < 30; ++i) det.record("buyers");
    EXPECT_TRUE(det.is_buy_imbalance());
    EXPECT_FALSE(det.is_sell_imbalance());
}

// ============================================================
// is_sell_imbalance fires at threshold
// ============================================================

TEST(OrderFlowImbalanceDetector, IsSellImbalanceBelowThreshold) {
    OrderFlowImbalanceDetector::Config cfg;
    cfg.ema_alpha            = 0.9;
    cfg.imbalance_threshold  = 0.5;
    cfg.min_tokens           = 1;
    OrderFlowImbalanceDetector det(cfg);

    for (int i = 0; i < 30; ++i) det.record("sellers");
    EXPECT_TRUE(det.is_sell_imbalance());
    EXPECT_FALSE(det.is_buy_imbalance());
}

// ============================================================
// Callback fires when threshold exceeded
// ============================================================

TEST(OrderFlowImbalanceDetector, CallbackFiresAtThreshold) {
    OrderFlowImbalanceDetector::Config cfg;
    cfg.ema_alpha           = 0.9;
    cfg.imbalance_threshold = 0.5;
    cfg.min_tokens          = 1;
    std::atomic<int> fires{0};
    cfg.on_imbalance = [&](double /*imb*/, bool /*buy*/) { ++fires; };
    OrderFlowImbalanceDetector det(cfg);

    for (int i = 0; i < 20; ++i) det.record("buyers");
    EXPECT_GT(fires.load(), 0);
}

// ============================================================
// record_pressure with explicit weight
// ============================================================

TEST(OrderFlowImbalanceDetector, RecordPressureDirectly) {
    OrderFlowImbalanceDetector::Config cfg;
    cfg.ema_alpha  = 0.5;
    cfg.min_tokens = 1;
    OrderFlowImbalanceDetector det(cfg);

    // Drive imbalance negative with explicit sell pressure.
    for (int i = 0; i < 20; ++i) det.record_pressure(-1.0);
    EXPECT_LT(det.imbalance(), 0.0);
}

// ============================================================
// Custom token weight
// ============================================================

TEST(OrderFlowImbalanceDetector, CustomTokenWeight) {
    OrderFlowImbalanceDetector::Config cfg;
    cfg.ema_alpha  = 0.9;
    cfg.min_tokens = 1;
    OrderFlowImbalanceDetector det(cfg);

    det.add_token_weight("gamma", 1.0);  // custom buy keyword
    for (int i = 0; i < 20; ++i) det.record("gamma");
    EXPECT_GT(det.imbalance(), 0.0);
}

// ============================================================
// total_tokens increments
// ============================================================

TEST(OrderFlowImbalanceDetector, TotalTokensIncrements) {
    OrderFlowImbalanceDetector det;
    det.record("bullish");
    det.record("the");
    det.record("fear");
    EXPECT_EQ(det.total_tokens(), 3u);
}

// ============================================================
// reset clears state
// ============================================================

TEST(OrderFlowImbalanceDetector, ResetClearsState) {
    OrderFlowImbalanceDetector::Config cfg;
    cfg.ema_alpha  = 0.5;
    cfg.min_tokens = 1;
    OrderFlowImbalanceDetector det(cfg);

    for (int i = 0; i < 30; ++i) det.record("buyers");
    det.reset();

    EXPECT_NEAR(det.imbalance(), 0.0, 1e-9);
    EXPECT_EQ(det.total_tokens(), 0u);
}

// ============================================================
// to_stats_json contains expected keys
// ============================================================

TEST(OrderFlowImbalanceDetector, ToStatsJsonContainsKeys) {
    OrderFlowImbalanceDetector det;
    det.record("bullish");
    auto j = det.to_stats_json();
    EXPECT_NE(j.find("imbalance"),       std::string::npos);
    EXPECT_NE(j.find("total_tokens"),    std::string::npos);
    EXPECT_NE(j.find("is_buy"),          std::string::npos);
    EXPECT_NE(j.find("is_sell"),         std::string::npos);
}

// ============================================================
// update_config changes threshold without resetting EMA
// ============================================================

TEST(OrderFlowImbalanceDetector, UpdateConfigChangesThreshold) {
    OrderFlowImbalanceDetector::Config cfg;
    cfg.ema_alpha  = 0.5;
    cfg.min_tokens = 1;
    OrderFlowImbalanceDetector det(cfg);

    for (int i = 0; i < 20; ++i) det.record("buyers");
    double imb = det.imbalance();
    EXPECT_GT(imb, 0.0);

    // Lower threshold so the existing imbalance qualifies.
    cfg.imbalance_threshold = 0.01;
    det.update_config(cfg);
    EXPECT_TRUE(det.is_buy_imbalance());
}

// ============================================================
// Case-insensitive keyword matching
// ============================================================

TEST(OrderFlowImbalanceDetector, CaseInsensitiveMatch) {
    OrderFlowImbalanceDetector::Config cfg;
    cfg.ema_alpha  = 0.9;
    cfg.min_tokens = 1;
    OrderFlowImbalanceDetector det(cfg);

    // "BUYERS" should match the "buyers" dictionary entry.
    for (int i = 0; i < 20; ++i) det.record("BUYERS");
    EXPECT_GT(det.imbalance(), 0.0);
}

// ============================================================
// Mixed buy/sell tokens approach zero
// ============================================================

TEST(OrderFlowImbalanceDetector, MixedTokensApproachZero) {
    OrderFlowImbalanceDetector::Config cfg;
    cfg.ema_alpha  = 0.1;
    cfg.min_tokens = 1;
    OrderFlowImbalanceDetector det(cfg);

    for (int i = 0; i < 200; ++i)
        det.record(i % 2 == 0 ? "buyers" : "sellers");

    // After equal buy/sell, EMA should be near 0.
    EXPECT_NEAR(det.imbalance(), 0.0, 0.1);
}

} // namespace

#else

TEST(OrderFlowImbalanceDetector, DisabledAtBuildTime) { SUCCEED(); }

#endif  // LLMQUANT_ORDER_FLOW_IMBALANCE_ENABLED
