#include <gtest/gtest.h>

#ifdef LLMQUANT_SIGNAL_FATIGUE_ENABLED
#include "SignalFatigueMeter.h"
using namespace llmquant;

TEST(SignalFatigueMeterTest, InitialState) {
    SignalFatigueMeter m;
    EXPECT_EQ(m.streak(), 0);
    EXPECT_EQ(m.direction(), 0);
    EXPECT_EQ(m.fatigue_score(), 0.0);
    EXPECT_FALSE(m.is_fatigued());
    EXPECT_EQ(m.observation_count(), 0u);
}

TEST(SignalFatigueMeterTest, ObservationCountIncrement) {
    SignalFatigueMeter m;
    m.record(0.1);
    m.record(0.2);
    m.record(-0.1);
    EXPECT_EQ(m.observation_count(), 3u);
}

TEST(SignalFatigueMeterTest, BullishStreakBuilds) {
    SignalFatigueMeter m;
    for (int i = 0; i < 6; ++i) m.record(0.5);
    EXPECT_EQ(m.streak(), 6);
    EXPECT_EQ(m.direction(), 1);
    EXPECT_TRUE(m.is_fatigued());
}

TEST(SignalFatigueMeterTest, BearishStreakBuilds) {
    SignalFatigueMeter m;
    for (int i = 0; i < 6; ++i) m.record(-0.3);
    EXPECT_EQ(m.streak(), 6);
    EXPECT_EQ(m.direction(), -1);
    EXPECT_TRUE(m.is_fatigued());
}

TEST(SignalFatigueMeterTest, DirectionFlipResetsStreak) {
    SignalFatigueMeter m;
    for (int i = 0; i < 6; ++i) m.record(0.5);
    EXPECT_TRUE(m.is_fatigued());
    m.record(-0.3); // flip
    EXPECT_EQ(m.streak(), 1);
    EXPECT_EQ(m.direction(), -1);
    EXPECT_FALSE(m.is_fatigued());
}

TEST(SignalFatigueMeterTest, NeutralBreaksStreak) {
    SignalFatigueMeter m;
    for (int i = 0; i < 6; ++i) m.record(0.5);
    m.record(0.0); // neutral
    EXPECT_EQ(m.streak(), 0);
    EXPECT_EQ(m.direction(), 0);
    EXPECT_FALSE(m.is_fatigued());
}

TEST(SignalFatigueMeterTest, FatigueCallback) {
    SignalFatigueMeter::Config cfg;
    cfg.fatigue_threshold  = 4;
    cfg.saturation_streak  = 8;
    int fires = 0;
    cfg.on_fatigue = [&](int, double) { ++fires; };
    SignalFatigueMeter m(cfg);
    for (int i = 0; i < 8; ++i) m.record(0.5);
    EXPECT_GE(fires, 1);
}

TEST(SignalFatigueMeterTest, FatigueScoreRange) {
    SignalFatigueMeter m;
    for (int i = 0; i < 20; ++i) m.record(0.5);
    EXPECT_GE(m.fatigue_score(), 0.0);
    EXPECT_LE(m.fatigue_score(), 1.0);
}

TEST(SignalFatigueMeterTest, Reset) {
    SignalFatigueMeter m;
    for (int i = 0; i < 10; ++i) m.record(0.5);
    m.reset();
    EXPECT_EQ(m.streak(), 0);
    EXPECT_EQ(m.observation_count(), 0u);
    EXPECT_FALSE(m.is_fatigued());
}

TEST(SignalFatigueMeterTest, StatsJson) {
    SignalFatigueMeter m;
    m.record(0.5);
    std::string json = m.to_stats_json();
    EXPECT_NE(json.find("fatigue_score"), std::string::npos);
}

#else
TEST(SignalFatigueMeterTest, Disabled) { SUCCEED(); }
#endif
