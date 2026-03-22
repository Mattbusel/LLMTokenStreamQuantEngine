#include <gtest/gtest.h>

#ifdef LLMQUANT_SIGNAL_PERSISTENCE_ENABLED
#include "SignalPersistenceTracker.h"

using namespace llmquant;
using Dir = SignalPersistenceTracker::Direction;

TEST(SignalPersistenceTrackerTest, ConstructsOk) {
    SignalPersistenceTracker t;
    EXPECT_EQ(t.current_streak(), 0);
    EXPECT_EQ(t.total_signals(), 0u);
}

TEST(SignalPersistenceTrackerTest, FirstSignalStartsStreak) {
    SignalPersistenceTracker t;
    int s = t.record(Dir::Bullish);
    EXPECT_EQ(s, 1);
    EXPECT_EQ(t.current_direction(), Dir::Bullish);
}

TEST(SignalPersistenceTrackerTest, SameDirectionExtendsStreak) {
    SignalPersistenceTracker t;
    t.record(Dir::Bullish);
    t.record(Dir::Bullish);
    EXPECT_EQ(t.current_streak(), 2);
}

TEST(SignalPersistenceTrackerTest, NeutralDoesNotBreakOrExtend) {
    SignalPersistenceTracker t;
    t.record(Dir::Bullish);
    t.record(Dir::Bullish);
    t.record(Dir::Neutral);
    EXPECT_EQ(t.current_streak(), 2);
    EXPECT_EQ(t.current_direction(), Dir::Bullish);
}

TEST(SignalPersistenceTrackerTest, ReversalResetsStreak) {
    SignalPersistenceTracker t;
    t.record(Dir::Bullish);
    t.record(Dir::Bullish);
    t.record(Dir::Bearish);
    EXPECT_EQ(t.current_streak(), 1);
    EXPECT_EQ(t.current_direction(), Dir::Bearish);
}

TEST(SignalPersistenceTrackerTest, ReversalCallbackFires) {
    SignalPersistenceTracker::Config cfg;
    int fires = 0;
    cfg.on_reversal = [&](int streak, Dir) {
        EXPECT_EQ(streak, 3);
        ++fires;
    };
    SignalPersistenceTracker t(cfg);
    t.record(Dir::Bullish);
    t.record(Dir::Bullish);
    t.record(Dir::Bullish);
    t.record(Dir::Bearish);
    EXPECT_EQ(fires, 1);
    EXPECT_EQ(t.total_reversals(), 1u);
}

TEST(SignalPersistenceTrackerTest, ConvictionCallbackEdgeTriggered) {
    SignalPersistenceTracker::Config cfg;
    cfg.conviction_threshold = 3;
    int fires = 0;
    cfg.on_conviction = [&](int) { ++fires; };
    SignalPersistenceTracker t(cfg);

    // 5 bullish — conviction fires once at streak=3, not again at 4 or 5.
    for (int i = 0; i < 5; ++i) t.record(Dir::Bullish);
    EXPECT_EQ(fires, 1);
    EXPECT_EQ(t.conviction_events(), 1u);
}

TEST(SignalPersistenceTrackerTest, ConvictionFiresAgainAfterReversal) {
    SignalPersistenceTracker::Config cfg;
    cfg.conviction_threshold = 3;
    int fires = 0;
    cfg.on_conviction = [&](int) { ++fires; };
    SignalPersistenceTracker t(cfg);

    for (int i = 0; i < 4; ++i) t.record(Dir::Bullish);
    t.record(Dir::Bearish);
    for (int i = 0; i < 4; ++i) t.record(Dir::Bearish);
    EXPECT_EQ(fires, 2);
}

TEST(SignalPersistenceTrackerTest, ConvictionScaleAtMinimum) {
    SignalPersistenceTracker t;
    EXPECT_NEAR(t.conviction_scale(), 1.0, 0.01); // zero streak → scale = 1
}

TEST(SignalPersistenceTrackerTest, ConvictionScaleGrowsWithStreak) {
    SignalPersistenceTracker::Config cfg;
    cfg.conviction_threshold = 5;
    cfg.max_scale            = 3.0;
    SignalPersistenceTracker t(cfg);
    for (int i = 0; i < 5; ++i) t.record(Dir::Bullish);
    // scale = 1 + 5/5 = 2.0
    EXPECT_NEAR(t.conviction_scale(), 2.0, 0.01);
}

TEST(SignalPersistenceTrackerTest, ConvictionScaleCappedAtMaxScale) {
    SignalPersistenceTracker::Config cfg;
    cfg.conviction_threshold = 5;
    cfg.max_scale            = 1.5;
    SignalPersistenceTracker t(cfg);
    for (int i = 0; i < 20; ++i) t.record(Dir::Bullish);
    EXPECT_NEAR(t.conviction_scale(), 1.5, 0.01);
}

TEST(SignalPersistenceTrackerTest, RecordBiasPositiveIsBullish) {
    SignalPersistenceTracker t;
    t.record_bias(0.5);
    EXPECT_EQ(t.current_direction(), Dir::Bullish);
}

TEST(SignalPersistenceTrackerTest, RecordBiasNegativeIsBearish) {
    SignalPersistenceTracker t;
    t.record_bias(-0.5);
    EXPECT_EQ(t.current_direction(), Dir::Bearish);
}

TEST(SignalPersistenceTrackerTest, RecordBiasWithinDeadbandIsNeutral) {
    SignalPersistenceTracker t;
    t.record_bias(0.0005, 0.001); // within deadband
    EXPECT_EQ(t.current_streak(), 0);
}

TEST(SignalPersistenceTrackerTest, ResetClearsStreak) {
    SignalPersistenceTracker t;
    for (int i = 0; i < 5; ++i) t.record(Dir::Bullish);
    t.reset();
    EXPECT_EQ(t.current_streak(), 0);
    EXPECT_EQ(t.current_direction(), Dir::Neutral);
}

TEST(SignalPersistenceTrackerTest, StatsJsonContainsRequiredKeys) {
    SignalPersistenceTracker t;
    t.record(Dir::Bullish);
    std::string json = t.to_stats_json();
    EXPECT_NE(json.find("current_streak"),    std::string::npos);
    EXPECT_NE(json.find("direction"),         std::string::npos);
    EXPECT_NE(json.find("conviction_scale"),  std::string::npos);
    EXPECT_NE(json.find("total_signals"),     std::string::npos);
    EXPECT_NE(json.find("total_reversals"),   std::string::npos);
}

#else
TEST(SignalPersistenceTrackerTest, FeatureFlagDisabled) { SUCCEED(); }
#endif
