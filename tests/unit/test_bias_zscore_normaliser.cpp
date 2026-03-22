#include <gtest/gtest.h>
#include "BiasZScoreNormaliser.h"

using namespace llmquant;

TEST(BiasZScoreNormaliser, DefaultConstruction) {
    BiasZScoreNormaliser b(BiasZScoreNormaliser::Config{});
    EXPECT_EQ(b.total_records(), 0u);
    EXPECT_DOUBLE_EQ(b.z_score(), 0.0);
}

TEST(BiasZScoreNormaliser, ZeroForConstantSeries) {
    BiasZScoreNormaliser::Config cfg;
    cfg.window = 5;
    cfg.min_samples = 3;
    BiasZScoreNormaliser b(cfg);
    for (int i = 0; i < 5; ++i) b.record(0.5);
    EXPECT_NEAR(b.z_score(), 0.0, 1e-6);
}

TEST(BiasZScoreNormaliser, MeanApproachesInput) {
    BiasZScoreNormaliser::Config cfg;
    cfg.window = 5;
    cfg.min_samples = 3;
    BiasZScoreNormaliser b(cfg);
    for (int i = 0; i < 5; ++i) b.record(1.0);
    EXPECT_NEAR(b.rolling_mean(), 1.0, 1e-6);
}

TEST(BiasZScoreNormaliser, PositiveExtremeCallback) {
    BiasZScoreNormaliser::Config cfg;
    cfg.window = 10;
    cfg.min_samples = 5;
    cfg.extreme_threshold = 1.5;
    int pos_count = 0;
    cfg.on_positive_extreme = [&](double) { ++pos_count; };
    BiasZScoreNormaliser b(cfg);
    for (int i = 0; i < 8; ++i) b.record(0.0);
    b.record(100.0); // extreme outlier
    EXPECT_GE(pos_count, 1);
}

TEST(BiasZScoreNormaliser, NegativeExtremeCallback) {
    BiasZScoreNormaliser::Config cfg;
    cfg.window = 10;
    cfg.min_samples = 5;
    cfg.extreme_threshold = 1.5;
    int neg_count = 0;
    cfg.on_negative_extreme = [&](double) { ++neg_count; };
    BiasZScoreNormaliser b(cfg);
    for (int i = 0; i < 8; ++i) b.record(0.0);
    b.record(-100.0);
    EXPECT_GE(neg_count, 1);
}

TEST(BiasZScoreNormaliser, StdNonNegative) {
    BiasZScoreNormaliser::Config cfg;
    cfg.window = 5;
    cfg.min_samples = 3;
    BiasZScoreNormaliser b(cfg);
    for (int i = 0; i < 5; ++i) b.record(0.01 * i);
    EXPECT_GE(b.rolling_std(), 0.0);
}

TEST(BiasZScoreNormaliser, Reset) {
    BiasZScoreNormaliser b(BiasZScoreNormaliser::Config{});
    for (int i = 0; i < 10; ++i) b.record(0.05);
    b.reset();
    EXPECT_EQ(b.total_records(), 0u);
    EXPECT_DOUBLE_EQ(b.z_score(), 0.0);
    EXPECT_EQ(b.positive_extremes(), 0u);
    EXPECT_EQ(b.negative_extremes(), 0u);
}

TEST(BiasZScoreNormaliser, ToStatsJson) {
    BiasZScoreNormaliser b(BiasZScoreNormaliser::Config{});
    for (int i = 0; i < 5; ++i) b.record(0.05);
    std::string j = b.to_stats_json();
    EXPECT_NE(j.find("z_score"), std::string::npos);
    EXPECT_NE(j.find("rolling_mean"), std::string::npos);
    EXPECT_NE(j.find("total_records"), std::string::npos);
}
