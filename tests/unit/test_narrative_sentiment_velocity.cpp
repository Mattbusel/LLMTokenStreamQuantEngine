#include <gtest/gtest.h>
#include "NarrativeSentimentVelocity.h"

using namespace llmquant;

TEST(NarrativeSentimentVelocity, DefaultConstruction) {
    NarrativeSentimentVelocity v(NarrativeSentimentVelocity::Config{});
    EXPECT_EQ(v.total_records(), 0u);
    EXPECT_DOUBLE_EQ(v.ema(), 0.0);
    EXPECT_DOUBLE_EQ(v.velocity(), 0.0);
}

TEST(NarrativeSentimentVelocity, EmaTracksPositive) {
    NarrativeSentimentVelocity::Config cfg;
    cfg.alpha = 0.5;
    NarrativeSentimentVelocity v(cfg);
    for (int i = 0; i < 10; ++i) v.record(1.0);
    EXPECT_GT(v.ema(), 0.0);
}

TEST(NarrativeSentimentVelocity, VelocityPositiveForUptrend) {
    NarrativeSentimentVelocity::Config cfg;
    cfg.alpha = 0.5;
    cfg.min_samples = 2;
    NarrativeSentimentVelocity v(cfg);
    v.record(0.0);
    v.record(0.5);
    v.record(1.0);
    EXPECT_GT(v.velocity(), 0.0);
}

TEST(NarrativeSentimentVelocity, PositiveSurgeCallback) {
    NarrativeSentimentVelocity::Config cfg;
    cfg.alpha = 0.5;
    cfg.surge_threshold = 0.001;
    cfg.min_samples = 2;
    int surges = 0;
    cfg.on_positive_surge = [&](double) { ++surges; };
    NarrativeSentimentVelocity v(cfg);
    for (int i = 0; i < 5; ++i) v.record(1.0);
    EXPECT_GE(surges, 1);
}

TEST(NarrativeSentimentVelocity, NegativeSurgeCallback) {
    NarrativeSentimentVelocity::Config cfg;
    cfg.alpha = 0.5;
    cfg.surge_threshold = 0.001;
    cfg.min_samples = 2;
    int surges = 0;
    cfg.on_negative_surge = [&](double) { ++surges; };
    NarrativeSentimentVelocity v(cfg);
    for (int i = 0; i < 5; ++i) v.record(-1.0);
    EXPECT_GE(surges, 1);
}

TEST(NarrativeSentimentVelocity, Reset) {
    NarrativeSentimentVelocity v(NarrativeSentimentVelocity::Config{});
    for (int i = 0; i < 5; ++i) v.record(0.1);
    v.reset();
    EXPECT_EQ(v.total_records(), 0u);
    EXPECT_DOUBLE_EQ(v.ema(), 0.0);
    EXPECT_DOUBLE_EQ(v.velocity(), 0.0);
    EXPECT_EQ(v.positive_surges(), 0u);
    EXPECT_EQ(v.negative_surges(), 0u);
}

TEST(NarrativeSentimentVelocity, ToStatsJson) {
    NarrativeSentimentVelocity v(NarrativeSentimentVelocity::Config{});
    for (int i = 0; i < 5; ++i) v.record(0.05);
    std::string j = v.to_stats_json();
    EXPECT_NE(j.find("velocity"), std::string::npos);
    EXPECT_NE(j.find("acceleration"), std::string::npos);
    EXPECT_NE(j.find("total_records"), std::string::npos);
}
