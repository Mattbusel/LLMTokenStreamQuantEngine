#include <gtest/gtest.h>
#include <thread>
#include <vector>

#ifdef LLMQUANT_ADAPTIVE_THRESHOLD_ENABLED
#include "AdaptiveThresholdController.h"
using namespace llmquant;

TEST(AdaptiveThresholdController, ConstructDefault) {
    AdaptiveThresholdController atc;
    EXPECT_DOUBLE_EQ(atc.overbought_threshold(), 70.0);
    EXPECT_DOUBLE_EQ(atc.oversold_threshold(),   30.0);
    EXPECT_DOUBLE_EQ(atc.rolling_sigma(),         0.0);
    EXPECT_EQ(atc.change_events(),  0u);
    EXPECT_EQ(atc.total_records(),  0u);
}

TEST(AdaptiveThresholdController, TotalRecordsIncrement) {
    AdaptiveThresholdController atc;
    atc.record(50.0);
    atc.record(60.0);
    EXPECT_EQ(atc.total_records(), 2u);
}

TEST(AdaptiveThresholdController, ThresholdsWithinClampRange) {
    AdaptiveThresholdController atc;
    for (int i = 0; i < 100; ++i) atc.record(50.0 + 20.0 * (i % 3 - 1));
    double ob = atc.overbought_threshold();
    double os = atc.oversold_threshold();
    EXPECT_GE(ob, 55.0);
    EXPECT_LE(ob, 95.0);
    EXPECT_GE(os,  5.0);
    EXPECT_LE(os, 45.0);
}

TEST(AdaptiveThresholdController, HighVolatilityWidensOverbought) {
    AdaptiveThresholdController low_vol, high_vol;
    // Low vol: tiny swings
    for (int i = 0; i < 40; ++i) low_vol.record(50.0 + (i % 2) * 0.01);
    // High vol: large swings
    for (int i = 0; i < 40; ++i) high_vol.record(50.0 + (i % 2) * 20.0);
    EXPECT_GE(high_vol.overbought_threshold(), low_vol.overbought_threshold());
}

TEST(AdaptiveThresholdController, HighVolatilityNarrowsOversold) {
    AdaptiveThresholdController low_vol, high_vol;
    for (int i = 0; i < 40; ++i) low_vol.record(50.0 + (i % 2) * 0.01);
    for (int i = 0; i < 40; ++i) high_vol.record(50.0 + (i % 2) * 20.0);
    EXPECT_LE(high_vol.oversold_threshold(), low_vol.oversold_threshold());
}

TEST(AdaptiveThresholdController, SigmaIncreasesAfterLargeMove) {
    AdaptiveThresholdController atc;
    atc.record(50.0);
    atc.record(70.0);   // Δ = 20
    EXPECT_GT(atc.rolling_sigma(), 0.0);
}

TEST(AdaptiveThresholdController, SigmaDecaysTowardZero) {
    AdaptiveThresholdController atc;
    // Warm up sigma
    for (int i = 0; i < 20; ++i) atc.record(50.0 + (i % 2) * 10.0);
    double sigma_warm = atc.rolling_sigma();
    // Feed constant signal — sigma should decay
    for (int i = 0; i < 200; ++i) atc.record(50.0);
    EXPECT_LT(atc.rolling_sigma(), sigma_warm);
}

TEST(AdaptiveThresholdController, CallbackFiredOnChange) {
    AdaptiveThresholdController::Config cfg;
    cfg.change_threshold = 0.0;   // fire on every change
    int fired = 0;
    cfg.on_threshold_change = [&](double, double) { ++fired; };
    AdaptiveThresholdController atc(cfg);
    for (int i = 0; i < 20; ++i) atc.record(50.0 + i * 3.0);
    EXPECT_GT(fired, 0);
}

TEST(AdaptiveThresholdController, CallbackReceivesClampedValues) {
    AdaptiveThresholdController::Config cfg;
    cfg.change_threshold = 0.0;
    double last_ob = -1.0, last_os = -1.0;
    cfg.on_threshold_change = [&](double ob, double os) {
        last_ob = ob;
        last_os = os;
    };
    AdaptiveThresholdController atc(cfg);
    for (int i = 0; i < 30; ++i) atc.record(50.0 + (i % 4) * 10.0);
    EXPECT_GE(last_ob, 55.0);
    EXPECT_LE(last_ob, 95.0);
    EXPECT_GE(last_os,  5.0);
    EXPECT_LE(last_os, 45.0);
}

TEST(AdaptiveThresholdController, ChangeEventsAccumulate) {
    AdaptiveThresholdController::Config cfg;
    cfg.change_threshold = 0.0;
    AdaptiveThresholdController atc(cfg);
    for (int i = 0; i < 20; ++i) atc.record(50.0 + i * 2.0);
    EXPECT_GT(atc.change_events(), 0u);
}

TEST(AdaptiveThresholdController, Reset) {
    AdaptiveThresholdController atc;
    for (int i = 0; i < 30; ++i) atc.record(50.0 + i);
    atc.reset();
    EXPECT_DOUBLE_EQ(atc.overbought_threshold(), 70.0);
    EXPECT_DOUBLE_EQ(atc.oversold_threshold(),   30.0);
    EXPECT_DOUBLE_EQ(atc.rolling_sigma(),         0.0);
    EXPECT_EQ(atc.change_events(), 0u);
    EXPECT_EQ(atc.total_records(), 0u);
}

TEST(AdaptiveThresholdController, UpdateConfigAppliesNewBase) {
    AdaptiveThresholdController atc;
    for (int i = 0; i < 20; ++i) atc.record(60.0);
    AdaptiveThresholdController::Config cfg;
    cfg.base_overbought = 80.0;
    cfg.base_oversold   = 20.0;
    atc.update_config(cfg);
    EXPECT_DOUBLE_EQ(atc.overbought_threshold(), 80.0);
    EXPECT_DOUBLE_EQ(atc.oversold_threshold(),   20.0);
}

TEST(AdaptiveThresholdController, ToStatsJsonFields) {
    AdaptiveThresholdController atc;
    for (int i = 0; i < 10; ++i) atc.record(50.0 + i);
    auto js = atc.to_stats_json();
    EXPECT_NE(js.find("overbought_threshold"), std::string::npos);
    EXPECT_NE(js.find("oversold_threshold"),   std::string::npos);
    EXPECT_NE(js.find("rolling_sigma"),        std::string::npos);
    EXPECT_NE(js.find("change_events"),        std::string::npos);
    EXPECT_NE(js.find("total_records"),        std::string::npos);
}

TEST(AdaptiveThresholdController, ConcurrentRecord) {
    AdaptiveThresholdController atc;
    std::vector<std::thread> threads;
    for (int t = 0; t < 4; ++t)
        threads.emplace_back([&] {
            for (int i = 0; i < 50; ++i) atc.record(50.0 + i);
        });
    for (auto& th : threads) th.join();
    EXPECT_GE(atc.total_records(), 100u);
}

#else
TEST(AdaptiveThresholdController, DisabledAtBuildTime) { SUCCEED(); }
#endif
