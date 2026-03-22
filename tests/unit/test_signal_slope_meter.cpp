#include <gtest/gtest.h>

#ifdef LLMQUANT_SIGNAL_SLOPE_ENABLED
#include "SignalSlopeMeter.h"
using namespace llmquant;

TEST(SignalSlopeMeterTest, InitialState) {
    SignalSlopeMeter m;
    EXPECT_EQ(m.slope(), 0.0);
    EXPECT_EQ(m.slope_score(), 0.0);
    EXPECT_FALSE(m.is_accelerating());
    EXPECT_EQ(m.observation_count(), 0u);
}

TEST(SignalSlopeMeterTest, ObservationCountIncrement) {
    SignalSlopeMeter m;
    m.record(0.1);
    m.record(0.2);
    m.record(0.3);
    EXPECT_EQ(m.observation_count(), 3u);
}

TEST(SignalSlopeMeterTest, RisingSlopePositive) {
    SignalSlopeMeter m;
    // Feed strictly increasing values — slope should be positive.
    for (int i = 0; i < 20; ++i) m.record(static_cast<double>(i) * 0.01);
    EXPECT_GT(m.slope(), 0.0);
}

TEST(SignalSlopeMeterTest, FallingSlopeNegative) {
    SignalSlopeMeter m;
    for (int i = 0; i < 20; ++i) m.record(-static_cast<double>(i) * 0.01);
    EXPECT_LT(m.slope(), 0.0);
}

TEST(SignalSlopeMeterTest, FlatSeriesZeroSlope) {
    SignalSlopeMeter m;
    for (int i = 0; i < 20; ++i) m.record(0.5);
    EXPECT_NEAR(m.slope(), 0.0, 1e-9);
}

TEST(SignalSlopeMeterTest, SlopeScoreClampedToOne) {
    SignalSlopeMeter::Config cfg;
    cfg.window              = 10;
    cfg.saturation_slope    = 0.01; // low saturation
    cfg.acceleration_threshold = 0.005;
    SignalSlopeMeter m(cfg);
    // Very steep rise — score should saturate at 1.
    for (int i = 0; i < 10; ++i) m.record(static_cast<double>(i) * 1.0);
    EXPECT_LE(m.slope_score(), 1.0);
    EXPECT_GE(m.slope_score(), -1.0);
}

TEST(SignalSlopeMeterTest, AccelerationDetected) {
    SignalSlopeMeter::Config cfg;
    cfg.window                 = 10;
    cfg.acceleration_threshold = 0.005;
    cfg.saturation_slope       = 0.05;
    SignalSlopeMeter m(cfg);
    for (int i = 0; i < 10; ++i) m.record(static_cast<double>(i) * 0.05);
    EXPECT_TRUE(m.is_accelerating());
}

TEST(SignalSlopeMeterTest, AccelerationCallback) {
    SignalSlopeMeter::Config cfg;
    cfg.window                 = 10;
    cfg.acceleration_threshold = 0.005;
    cfg.saturation_slope       = 0.05;
    int fires = 0;
    cfg.on_trend_acceleration = [&](double) { ++fires; };
    SignalSlopeMeter m(cfg);
    for (int i = 0; i < 20; ++i) m.record(static_cast<double>(i) * 0.05);
    EXPECT_GE(fires, 1);
}

TEST(SignalSlopeMeterTest, Reset) {
    SignalSlopeMeter m;
    for (int i = 0; i < 20; ++i) m.record(static_cast<double>(i) * 0.1);
    m.reset();
    EXPECT_EQ(m.observation_count(), 0u);
    EXPECT_EQ(m.slope(), 0.0);
    EXPECT_FALSE(m.is_accelerating());
}

TEST(SignalSlopeMeterTest, StatsJson) {
    SignalSlopeMeter m;
    m.record(0.1);
    std::string json = m.to_stats_json();
    EXPECT_NE(json.find("slope"), std::string::npos);
    EXPECT_NE(json.find("slope_score"), std::string::npos);
}

#else
TEST(SignalSlopeMeterTest, Disabled) { SUCCEED(); }
#endif
