#include <gtest/gtest.h>

#ifdef LLMQUANT_ORDER_BOOK_SIM_ENABLED
#include "OrderBookSimulator.h"
#include <cmath>

using namespace llmquant;

TEST(OrderBookSimulatorTest, ConstructsOk) {
    OrderBookSimulator lob;
    EXPECT_EQ(lob.total_updates(), 0u);
}

TEST(OrderBookSimulatorTest, InitialMidPriceEqualsRefPrice) {
    OrderBookSimulator::Config cfg;
    cfg.ref_price = 200.0;
    OrderBookSimulator lob(cfg);
    EXPECT_DOUBLE_EQ(lob.mid_price(), 200.0);
}

TEST(OrderBookSimulatorTest, PositiveBiasRaisesMidPrice) {
    OrderBookSimulator lob;
    double before = lob.mid_price();
    lob.update_bias(1.0);
    EXPECT_GT(lob.mid_price(), before);
}

TEST(OrderBookSimulatorTest, NegativeBiasLowersMidPrice) {
    OrderBookSimulator lob;
    double before = lob.mid_price();
    lob.update_bias(-1.0);
    EXPECT_LT(lob.mid_price(), before);
}

TEST(OrderBookSimulatorTest, HalfSpreadIncreaseWithOrderSize) {
    OrderBookSimulator lob;
    double hs_small = lob.half_spread(100.0);
    double hs_large = lob.half_spread(10000.0);
    EXPECT_GT(hs_large, hs_small);
}

TEST(OrderBookSimulatorTest, BuyFillAboveMid) {
    OrderBookSimulator lob;
    double mid  = lob.mid_price();
    double fill = lob.estimated_fill_price(OrderBookSimulator::Side::Buy, 1000.0);
    EXPECT_GT(fill, mid);
}

TEST(OrderBookSimulatorTest, SellFillBelowMid) {
    OrderBookSimulator lob;
    double mid  = lob.mid_price();
    double fill = lob.estimated_fill_price(OrderBookSimulator::Side::Sell, 1000.0);
    EXPECT_LT(fill, mid);
}

TEST(OrderBookSimulatorTest, ResetZerosBias) {
    OrderBookSimulator lob;
    for (int i = 0; i < 5; ++i) lob.update_bias(1.0);
    lob.reset();
    EXPECT_NEAR(lob.cumulative_bias(), 0.0, 1e-9);
    EXPECT_NEAR(lob.mid_price(), 100.0, 1e-6);
}

TEST(OrderBookSimulatorTest, TotalUpdatesIncrement) {
    OrderBookSimulator lob;
    lob.update_bias(0.1);
    lob.update_bias(0.2);
    EXPECT_EQ(lob.total_updates(), 2u);
}

TEST(OrderBookSimulatorTest, BiasDecayConvergesNotToRefPrice) {
    // With 0 < bias_decay < 1 and repeated small positive updates,
    // cumulative bias saturates at delta / bias_decay.
    OrderBookSimulator::Config cfg;
    cfg.bias_decay       = 0.1;
    cfg.bias_sensitivity = 0.01;
    OrderBookSimulator lob(cfg);
    for (int i = 0; i < 200; ++i) lob.update_bias(0.01);
    // Expected saturation ≈ 0.01 / 0.1 = 0.1
    EXPECT_NEAR(lob.cumulative_bias(), 0.1, 0.01);
}

TEST(OrderBookSimulatorTest, UpdateConfigResetsBias) {
    OrderBookSimulator lob;
    for (int i = 0; i < 10; ++i) lob.update_bias(0.5);
    lob.update_config(OrderBookSimulator::Config{});
    EXPECT_NEAR(lob.cumulative_bias(), 0.0, 1e-9);
}

TEST(OrderBookSimulatorTest, StatsJsonContainsRequiredKeys) {
    OrderBookSimulator lob;
    lob.update_bias(0.1);
    std::string json = lob.to_stats_json();
    EXPECT_NE(json.find("mid_price"),        std::string::npos);
    EXPECT_NE(json.find("cumulative_bias"),  std::string::npos);
    EXPECT_NE(json.find("total_updates"),    std::string::npos);
}

#else
TEST(OrderBookSimulatorTest, FeatureFlagDisabled) { SUCCEED(); }
#endif
