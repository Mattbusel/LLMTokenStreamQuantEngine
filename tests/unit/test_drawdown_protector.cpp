#include <gtest/gtest.h>

#ifdef LLMQUANT_DRAWDOWN_PROTECTOR_ENABLED

#include "DrawdownProtector.h"

#include <string>

using namespace llmquant;

namespace {

// ============================================================
// Construction
// ============================================================

TEST(DrawdownProtector, ConstructsOk) {
    DrawdownProtector dp;
    EXPECT_EQ(dp.current_tier(), 0);
    EXPECT_EQ(dp.tier_transitions(), 0u);
    EXPECT_NEAR(dp.cumulative_pnl(), 0.0, 1e-12);
    EXPECT_NEAR(dp.high_water_mark(), 0.0, 1e-12);
}

// ============================================================
// No drawdown — stays tier 0
// ============================================================

TEST(DrawdownProtector, ProfitableHistoryStaysTier0) {
    DrawdownProtector dp;
    dp.record_pnl(1.0);
    dp.record_pnl(1.0);
    dp.record_pnl(1.0);
    EXPECT_EQ(dp.current_tier(), 0);
    EXPECT_NEAR(dp.current_drawdown_pct(), 0.0, 1e-9);
}

// ============================================================
// high_water_mark tracking
// ============================================================

TEST(DrawdownProtector, HighWaterMarkTracksMax) {
    DrawdownProtector dp;
    dp.record_pnl(5.0);
    dp.record_pnl(3.0);
    dp.record_pnl(-4.0);
    EXPECT_NEAR(dp.high_water_mark(), 8.0, 1e-9);
    EXPECT_NEAR(dp.cumulative_pnl(), 4.0, 1e-9);
}

// ============================================================
// Tier 1 (caution) — 5–10% drawdown
// ============================================================

TEST(DrawdownProtector, Tier1ActivatesAt5PctDrawdown) {
    DrawdownProtector::Config cfg;
    cfg.tier1 = {0.05, 0.7};
    cfg.tier2 = {0.10, 0.4};
    cfg.tier3 = {0.20, 0.1};
    DrawdownProtector dp(cfg);

    dp.record_pnl(100.0);  // HWM = 100
    dp.record_pnl(-6.0);   // drawdown = 6/100 = 6% => tier 1
    EXPECT_EQ(dp.current_tier(), 1);
    EXPECT_NEAR(dp.current_drawdown_pct(), 0.06, 1e-9);
}

// ============================================================
// Tier 2 (warning) — 10–20% drawdown
// ============================================================

TEST(DrawdownProtector, Tier2ActivatesAt10PctDrawdown) {
    DrawdownProtector::Config cfg;
    cfg.tier1 = {0.05, 0.7};
    cfg.tier2 = {0.10, 0.4};
    cfg.tier3 = {0.20, 0.1};
    DrawdownProtector dp(cfg);

    dp.record_pnl(100.0);
    dp.record_pnl(-11.0);  // 11% drawdown => tier 2
    EXPECT_EQ(dp.current_tier(), 2);
    auto [cs, bs] = dp.current_scale();
    EXPECT_NEAR(cs, 0.4, 1e-9);
    EXPECT_NEAR(bs, 0.4, 1e-9);
}

// ============================================================
// Tier 3 (critical) — ≥20% drawdown
// ============================================================

TEST(DrawdownProtector, Tier3ActivatesAt20PctDrawdown) {
    DrawdownProtector::Config cfg;
    cfg.tier1 = {0.05, 0.7};
    cfg.tier2 = {0.10, 0.4};
    cfg.tier3 = {0.20, 0.1};
    DrawdownProtector dp(cfg);

    dp.record_pnl(100.0);
    dp.record_pnl(-25.0);  // 25% drawdown => tier 3
    EXPECT_EQ(dp.current_tier(), 3);
    auto [cs, bs] = dp.current_scale();
    EXPECT_NEAR(cs, 0.1, 1e-9);
}

// ============================================================
// Tier callback fires
// ============================================================

TEST(DrawdownProtector, TierCallbackFires) {
    DrawdownProtector::Config cfg;
    cfg.tier1 = {0.05, 0.7};
    cfg.tier2 = {0.10, 0.4};
    cfg.tier3 = {0.20, 0.1};
    int fired_tier = -1;
    cfg.on_tier_change = [&](int t, double, double) { fired_tier = t; };
    DrawdownProtector dp(cfg);

    dp.record_pnl(100.0);
    dp.record_pnl(-6.0);  // tier 0 -> 1
    EXPECT_EQ(fired_tier, 1);
}

// ============================================================
// Tier transitions counter
// ============================================================

TEST(DrawdownProtector, TierTransitionsCountedCorrectly) {
    DrawdownProtector::Config cfg;
    cfg.tier1 = {0.05, 0.7};
    cfg.tier2 = {0.10, 0.4};
    cfg.tier3 = {0.20, 0.1};
    DrawdownProtector dp(cfg);

    dp.record_pnl(100.0);   // HWM = 100
    dp.record_pnl(-6.0);    // tier 0 → 1  (+1)
    dp.record_pnl(-5.0);    // tier 1 → 2  (+1)
    dp.record_pnl(-10.0);   // tier 2 → 3  (+1)
    EXPECT_GE(dp.tier_transitions(), 3u);
}

// ============================================================
// No drawdown before profit — HWM = 0 branch
// ============================================================

TEST(DrawdownProtector, NoDrawdownWhenPnlNeverPositive) {
    DrawdownProtector dp;
    dp.record_pnl(-5.0);  // HWM stays 0; drawdown = 0
    EXPECT_EQ(dp.current_tier(), 0);
    EXPECT_NEAR(dp.current_drawdown_pct(), 0.0, 1e-9);
}

// ============================================================
// reset()
// ============================================================

TEST(DrawdownProtector, ResetClearsPnlAndTier) {
    DrawdownProtector::Config cfg;
    cfg.tier1 = {0.05, 0.7};
    DrawdownProtector dp(cfg);
    dp.record_pnl(100.0);
    dp.record_pnl(-6.0);
    EXPECT_EQ(dp.current_tier(), 1);
    dp.reset();
    EXPECT_EQ(dp.current_tier(), 0);
    EXPECT_NEAR(dp.cumulative_pnl(), 0.0, 1e-12);
    EXPECT_NEAR(dp.high_water_mark(), 0.0, 1e-12);
}

// ============================================================
// update_config
// ============================================================

TEST(DrawdownProtector, UpdateConfigDoesNotCrash) {
    DrawdownProtector dp;
    DrawdownProtector::Config cfg2;
    cfg2.tier1 = {0.02, 0.8};
    cfg2.tier2 = {0.05, 0.5};
    cfg2.tier3 = {0.15, 0.05};
    dp.update_config(cfg2);
    dp.record_pnl(1.0);
}

// ============================================================
// current_scale — tier 0 returns 1.0
// ============================================================

TEST(DrawdownProtector, Tier0ScaleIsOne) {
    DrawdownProtector dp;
    auto [cs, bs] = dp.current_scale();
    EXPECT_NEAR(cs, 1.0, 1e-9);
    EXPECT_NEAR(bs, 1.0, 1e-9);
}

// ============================================================
// to_stats_json
// ============================================================

TEST(DrawdownProtector, StatsJsonContainsRequiredKeys) {
    DrawdownProtector dp;
    dp.record_pnl(10.0);
    std::string js = dp.to_stats_json();
    EXPECT_NE(js.find("current_tier"),      std::string::npos);
    EXPECT_NE(js.find("current_drawdown"),  std::string::npos);
    EXPECT_NE(js.find("cumulative_pnl"),    std::string::npos);
    EXPECT_NE(js.find("high_water_mark"),   std::string::npos);
    EXPECT_NE(js.find("tier_transitions"),  std::string::npos);
}

} // namespace

#else  // LLMQUANT_DRAWDOWN_PROTECTOR_ENABLED not defined

TEST(DrawdownProtector, DisabledAtBuildTime) {
    SUCCEED();
}

#endif  // LLMQUANT_DRAWDOWN_PROTECTOR_ENABLED
