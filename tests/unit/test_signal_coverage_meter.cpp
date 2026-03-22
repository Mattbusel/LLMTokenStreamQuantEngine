#include <gtest/gtest.h>

#ifdef LLMQUANT_COVERAGE_METER_ENABLED
#include "SignalCoverageMeter.h"
using namespace llmquant;

TEST(SignalCoverageMeterTest, InitialState) {
    SignalCoverageMeter m;
    EXPECT_EQ(m.observation_count(), 0u);
    EXPECT_GE(m.coverage(), 0.0);
    EXPECT_LE(m.coverage(), 1.0);
    EXPECT_FALSE(m.is_expanded());
}

TEST(SignalCoverageMeterTest, ObservationCountIncrement) {
    SignalCoverageMeter m;
    m.record(0.5);
    m.record(-0.3);
    EXPECT_EQ(m.observation_count(), 2u);
}

TEST(SignalCoverageMeterTest, CoverageRangeValid) {
    SignalCoverageMeter m;
    for (int i = 0; i < 50; ++i)
        m.record(i % 2 == 0 ? 0.5 : -0.3);
    EXPECT_GE(m.coverage(), 0.0);
    EXPECT_LE(m.coverage(), 1.0);
}

TEST(SignalCoverageMeterTest, FullRangeCoverageNearOne) {
    SignalCoverageMeter::Config cfg;
    cfg.window = 10;
    cfg.range_min = -1.0;
    cfg.range_max =  1.0;
    cfg.expansion_threshold = 0.9;
    SignalCoverageMeter m(cfg);
    // Inject full range values.
    m.record(-1.0);
    m.record(1.0);
    EXPECT_GT(m.coverage(), 0.9);
}

TEST(SignalCoverageMeterTest, NarrowRangeLowCoverage) {
    SignalCoverageMeter::Config cfg;
    cfg.window    = 20;
    cfg.range_min = -1.0;
    cfg.range_max =  1.0;
    cfg.expansion_threshold = 0.5;
    SignalCoverageMeter m(cfg);
    for (int i = 0; i < 20; ++i) m.record(0.01 * (i % 3));
    EXPECT_LT(m.coverage(), 0.1);
    EXPECT_FALSE(m.is_expanded());
}

TEST(SignalCoverageMeterTest, RollingMinMax) {
    SignalCoverageMeter::Config cfg;
    cfg.window = 5;
    SignalCoverageMeter m(cfg);
    m.record(-0.5);
    m.record(0.8);
    EXPECT_NEAR(m.rolling_min(), -0.5, 1e-9);
    EXPECT_NEAR(m.rolling_max(),  0.8, 1e-9);
}

TEST(SignalCoverageMeterTest, ExpansionCallback) {
    SignalCoverageMeter::Config cfg;
    cfg.window = 10;
    cfg.range_min = -1.0;
    cfg.range_max =  1.0;
    cfg.expansion_threshold = 0.5;
    int fires = 0;
    cfg.on_range_expansion = [&](double) { ++fires; };
    SignalCoverageMeter m(cfg);
    m.record(-0.8);
    m.record(0.8);
    EXPECT_GE(fires, 1);
    EXPECT_TRUE(m.is_expanded());
}

TEST(SignalCoverageMeterTest, Reset) {
    SignalCoverageMeter m;
    m.record(-0.9);
    m.record(0.9);
    m.reset();
    EXPECT_EQ(m.observation_count(), 0u);
    EXPECT_EQ(m.coverage(), 0.0);
}

TEST(SignalCoverageMeterTest, StatsJson) {
    SignalCoverageMeter m;
    m.record(0.5);
    std::string json = m.to_stats_json();
    EXPECT_NE(json.find("coverage"), std::string::npos);
    EXPECT_NE(json.find("rolling_min"), std::string::npos);
}

#else
TEST(SignalCoverageMeterTest, Disabled) { SUCCEED(); }
#endif
