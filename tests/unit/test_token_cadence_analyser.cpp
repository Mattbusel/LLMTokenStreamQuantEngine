#include <gtest/gtest.h>
#include "TokenCadenceAnalyser.h"
#include <thread>
#include <chrono>

namespace llmquant::tests {

TEST(TokenCadenceAnalyser, DefaultConstruction) {
    TokenCadenceAnalyser c;
    EXPECT_EQ(c.total_records(), 0u);
    EXPECT_DOUBLE_EQ(c.mean_iai_ms(), 0.0);
}

TEST(TokenCadenceAnalyser, RecordsAccumulate) {
    TokenCadenceAnalyser c;
    for (int i = 0; i < 10; ++i) c.record(0.0);
    EXPECT_EQ(c.total_records(), 10u);
}

TEST(TokenCadenceAnalyser, MeanIAINonNegative) {
    TokenCadenceAnalyser c;
    for (int i = 0; i < 10; ++i) {
        c.record(0.0);
        std::this_thread::sleep_for(std::chrono::microseconds(100));
    }
    EXPECT_GE(c.mean_iai_ms(), 0.0);
}

TEST(TokenCadenceAnalyser, LastIAINonNegative) {
    TokenCadenceAnalyser c;
    for (int i = 0; i < 5; ++i) {
        c.record(0.0);
        std::this_thread::sleep_for(std::chrono::microseconds(100));
    }
    EXPECT_GE(c.last_iai_ms(), 0.0);
}

TEST(TokenCadenceAnalyser, GapCallback) {
    TokenCadenceAnalyser::Config cfg;
    cfg.gap_threshold_ms  = 0.001; // very small → any IAI fires gap
    cfg.min_samples       = 2;
    int gaps = 0;
    cfg.on_gap = [&](double) { ++gaps; };
    TokenCadenceAnalyser c(cfg);
    for (int i = 0; i < 10; ++i) {
        c.record(0.0);
        std::this_thread::sleep_for(std::chrono::microseconds(100));
    }
    EXPECT_GT(gaps, 0);
}

TEST(TokenCadenceAnalyser, CVNonNegative) {
    TokenCadenceAnalyser c;
    for (int i = 0; i < 15; ++i) {
        c.record(0.0);
        std::this_thread::sleep_for(std::chrono::microseconds(50));
    }
    EXPECT_GE(c.cv_iai(), 0.0);
}

TEST(TokenCadenceAnalyser, Reset) {
    TokenCadenceAnalyser c;
    for (int i = 0; i < 10; ++i) c.record(0.0);
    c.reset();
    EXPECT_EQ(c.total_records(), 0u);
    EXPECT_DOUBLE_EQ(c.mean_iai_ms(), 0.0);
    EXPECT_EQ(c.burst_events(), 0u);
    EXPECT_EQ(c.gap_events(), 0u);
}

TEST(TokenCadenceAnalyser, ToStatsJson) {
    TokenCadenceAnalyser c;
    for (int i = 0; i < 10; ++i) c.record(0.0);
    auto json = c.to_stats_json();
    EXPECT_NE(json.find("mean_iai_ms"), std::string::npos);
    EXPECT_NE(json.find("burst_events"), std::string::npos);
}

} // namespace llmquant::tests
