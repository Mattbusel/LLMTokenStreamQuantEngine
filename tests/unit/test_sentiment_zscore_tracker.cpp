#include <gtest/gtest.h>
#include <cmath>

#ifdef LLMQUANT_ZSCORE_TRACKER_ENABLED
#include "SentimentZScoreTracker.h"
using namespace llmquant;

TEST(SentimentZScoreTrackerTest, InitialState) {
    SentimentZScoreTracker t;
    EXPECT_EQ(t.observation_count(), 0u);
    EXPECT_FALSE(t.is_extreme());
}

TEST(SentimentZScoreTrackerTest, ObservationCountIncrement) {
    SentimentZScoreTracker t;
    t.record(0.1);
    t.record(0.2);
    EXPECT_EQ(t.observation_count(), 2u);
}

TEST(SentimentZScoreTrackerTest, ZScoreNearZeroForConstantSeries) {
    SentimentZScoreTracker t;
    for (int i = 0; i < 30; ++i) t.record(0.5);
    // All values equal → sigma→0 → z=0
    EXPECT_NEAR(t.z_score(), 0.0, 1e-9);
}

TEST(SentimentZScoreTrackerTest, ExtremeDetected) {
    SentimentZScoreTracker::Config cfg;
    cfg.window = 20;
    cfg.extreme_threshold = 2.0;
    SentimentZScoreTracker t(cfg);
    // Feed 19 zeros then a spike
    for (int i = 0; i < 19; ++i) t.record(0.0);
    t.record(5.0); // should be extreme
    EXPECT_TRUE(t.is_extreme());
    EXPECT_GT(std::abs(t.z_score()), 2.0);
}

TEST(SentimentZScoreTrackerTest, ExtremeCallback) {
    SentimentZScoreTracker::Config cfg;
    cfg.window = 20;
    cfg.extreme_threshold = 1.5;
    int fires = 0;
    cfg.on_extreme = [&](double) { ++fires; };
    SentimentZScoreTracker t(cfg);
    for (int i = 0; i < 19; ++i) t.record(0.0);
    t.record(5.0);
    EXPECT_GE(fires, 1);
}

TEST(SentimentZScoreTrackerTest, RollingMeanValid) {
    SentimentZScoreTracker t;
    for (int i = 0; i < 50; ++i) t.record(0.4);
    EXPECT_NEAR(t.rolling_mean(), 0.4, 1e-9);
}

TEST(SentimentZScoreTrackerTest, Reset) {
    SentimentZScoreTracker t;
    for (int i = 0; i < 20; ++i) t.record(0.5);
    t.reset();
    EXPECT_EQ(t.observation_count(), 0u);
    EXPECT_EQ(t.z_score(), 0.0);
}

TEST(SentimentZScoreTrackerTest, StatsJson) {
    SentimentZScoreTracker t;
    t.record(0.3);
    std::string json = t.to_stats_json();
    EXPECT_NE(json.find("z_score"), std::string::npos);
    EXPECT_NE(json.find("mean"), std::string::npos);
    EXPECT_NE(json.find("sigma"), std::string::npos);
}

#else
TEST(SentimentZScoreTrackerTest, Disabled) { SUCCEED(); }
#endif
