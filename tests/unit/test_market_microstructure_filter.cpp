#include <gtest/gtest.h>

#ifdef LLMQUANT_MARKET_MICROSTRUCTURE_ENABLED

#include "MarketMicrostructureFilter.h"

using namespace llmquant;

namespace {

// ============================================================
// Construction
// ============================================================

TEST(MarketMicrostructureFilter, DefaultConstructs) {
    MarketMicrostructureFilter f;
    EXPECT_GT(f.estimated_half_spread(), 0.0);
    EXPECT_EQ(f.total_fills(), 0u);
    EXPECT_EQ(f.total_passed(), 0u);
    EXPECT_EQ(f.total_blocked(), 0u);
}

// ============================================================
// should_trade passes with sufficient edge
// ============================================================

TEST(MarketMicrostructureFilter, PassesWithLargeEdge) {
    MarketMicrostructureFilter::Config cfg;
    cfg.initial_half_spread  = 0.01;
    cfg.impact_fraction      = 0.0;
    cfg.min_edge_multiplier  = 1.5;
    MarketMicrostructureFilter f(cfg);

    // cost = 0.01, need edge >= 0.015
    auto dec = f.should_trade(0.02, 0.0);
    EXPECT_EQ(dec, MarketMicrostructureFilter::Decision::Pass);
    EXPECT_EQ(f.total_passed(), 1u);
}

// ============================================================
// should_trade blocks with insufficient edge
// ============================================================

TEST(MarketMicrostructureFilter, BlocksWithSmallEdge) {
    MarketMicrostructureFilter::Config cfg;
    cfg.initial_half_spread  = 0.10;
    cfg.impact_fraction      = 0.0;
    cfg.min_edge_multiplier  = 1.5;
    MarketMicrostructureFilter f(cfg);

    // cost = 0.10, need edge >= 0.15; providing only 0.05
    auto dec = f.should_trade(0.05, 0.0);
    EXPECT_EQ(dec, MarketMicrostructureFilter::Decision::Block);
    EXPECT_EQ(f.total_blocked(), 1u);
}

// ============================================================
// record_fill updates spread downward
// ============================================================

TEST(MarketMicrostructureFilter, RecordFillReducesSpreadOnTightFill) {
    MarketMicrostructureFilter::Config cfg;
    cfg.initial_half_spread = 1.0;  // large initial spread
    cfg.ewma_decay          = 0.9;
    MarketMicrostructureFilter f(cfg);

    // Feed tight fills (arrival ≈ fill) → spread should decrease.
    for (int i = 0; i < 20; ++i)
        f.record_fill(100.0, 100.001, true);  // tiny slippage

    EXPECT_LT(f.estimated_half_spread(), 1.0);
    EXPECT_EQ(f.total_fills(), 20u);
}

// ============================================================
// record_fill updates spread upward on wide slippage
// ============================================================

TEST(MarketMicrostructureFilter, RecordFillIncreasesSpreadOnWideFill) {
    MarketMicrostructureFilter::Config cfg;
    cfg.initial_half_spread = 0.001;
    cfg.ewma_decay          = 0.5;
    MarketMicrostructureFilter f(cfg);

    f.record_fill(100.0, 102.0, true);  // 2% slippage
    EXPECT_GT(f.estimated_half_spread(), 0.001);
}

// ============================================================
// Sell fill contributes positive half-spread
// ============================================================

TEST(MarketMicrostructureFilter, SellFillSlippagePositive) {
    MarketMicrostructureFilter::Config cfg;
    cfg.initial_half_spread = 0.001;
    cfg.ewma_decay          = 0.5;
    MarketMicrostructureFilter f(cfg);

    // For a sell: fill_price < arrival_price → slippage = arrival - fill
    f.record_fill(100.0, 98.0, false);  // 2 point slippage
    EXPECT_GT(f.estimated_half_spread(), 0.001);
}

// ============================================================
// Impact fraction contributes to cost
// ============================================================

TEST(MarketMicrostructureFilter, ImpactFractionAddsToCost) {
    MarketMicrostructureFilter::Config cfg;
    cfg.initial_half_spread  = 0.01;
    cfg.impact_fraction      = 0.01;  // 1% of trade value
    cfg.min_edge_multiplier  = 1.0;
    MarketMicrostructureFilter f(cfg);

    // cost = 0.01 + 0.01 * 1.0 = 0.02; edge = 0.015 < 0.02 → block
    auto dec = f.should_trade(0.015, 1.0);
    EXPECT_EQ(dec, MarketMicrostructureFilter::Decision::Block);
}

// ============================================================
// estimated_cost
// ============================================================

TEST(MarketMicrostructureFilter, EstimatedCostFormula) {
    MarketMicrostructureFilter::Config cfg;
    cfg.initial_half_spread = 0.05;
    cfg.impact_fraction     = 0.001;
    MarketMicrostructureFilter f(cfg);

    double cost = f.estimated_cost(100.0);
    EXPECT_NEAR(cost, 0.05 + 0.001 * 100.0, 1e-9);
}

// ============================================================
// on_blocked callback
// ============================================================

TEST(MarketMicrostructureFilter, BlockedCallbackFires) {
    MarketMicrostructureFilter::Config cfg;
    cfg.initial_half_spread  = 0.10;
    cfg.impact_fraction      = 0.0;
    cfg.min_edge_multiplier  = 2.0;
    int fires = 0;
    cfg.on_blocked = [&](double, double) { ++fires; };
    MarketMicrostructureFilter f(cfg);

    f.should_trade(0.01, 0.0);  // blocks
    f.should_trade(0.01, 0.0);  // blocks again
    EXPECT_EQ(fires, 2);
}

// ============================================================
// reset
// ============================================================

TEST(MarketMicrostructureFilter, ResetRestoresInitialSpread) {
    MarketMicrostructureFilter::Config cfg;
    cfg.initial_half_spread = 0.05;
    cfg.ewma_decay          = 0.5;
    MarketMicrostructureFilter f(cfg);

    f.record_fill(100.0, 101.0, true);
    double after_fill = f.estimated_half_spread();
    f.reset();
    EXPECT_NEAR(f.estimated_half_spread(), 0.05, 1e-9);
    (void)after_fill;
}

// ============================================================
// update_config resets spread
// ============================================================

TEST(MarketMicrostructureFilter, UpdateConfigResetsSpread) {
    MarketMicrostructureFilter::Config cfg;
    cfg.initial_half_spread = 0.02;
    cfg.ewma_decay          = 0.5;
    MarketMicrostructureFilter f(cfg);

    f.record_fill(100.0, 110.0, true);  // drive spread high

    MarketMicrostructureFilter::Config cfg2;
    cfg2.initial_half_spread = 0.001;
    f.update_config(cfg2);
    EXPECT_NEAR(f.estimated_half_spread(), 0.001, 1e-9);
}

// ============================================================
// to_stats_json contains required keys
// ============================================================

TEST(MarketMicrostructureFilter, ToStatsJsonContainsKeys) {
    MarketMicrostructureFilter f;
    f.record_fill(100.0, 100.05, true);
    (void)f.should_trade(0.5, 10.0);
    std::string json = f.to_stats_json();
    EXPECT_NE(json.find("estimated_half_spread"), std::string::npos);
    EXPECT_NE(json.find("total_fills"),           std::string::npos);
    EXPECT_NE(json.find("total_passed"),          std::string::npos);
    EXPECT_NE(json.find("total_blocked"),         std::string::npos);
    EXPECT_NE(json.find("min_edge_multiplier"),   std::string::npos);
}

} // namespace

#else  // LLMQUANT_MICROSTRUCTURE_FILTER_ENABLED not defined

TEST(MarketMicrostructureFilter, DisabledAtBuildTime) {
    SUCCEED();
}

#endif  // LLMQUANT_MARKET_MICROSTRUCTURE_ENABLED
