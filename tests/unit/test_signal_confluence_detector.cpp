#include <gtest/gtest.h>

#ifdef LLMQUANT_CONFLUENCE_DETECTOR_ENABLED
#include "SignalConfluenceDetector.h"
using namespace llmquant;

TEST(SignalConfluenceDetectorTest, InitialState) {
    SignalConfluenceDetector d;
    EXPECT_EQ(d.confluence_score(), 0.0);
    EXPECT_EQ(d.dominant_direction(), 0);
    EXPECT_FALSE(d.is_confluent());
    EXPECT_EQ(d.observation_count(), 0u);
}

TEST(SignalConfluenceDetectorTest, ObservationCountIncrement) {
    SignalConfluenceDetector d;
    d.record(0.5);
    d.record(-0.3);
    EXPECT_EQ(d.observation_count(), 2u);
}

TEST(SignalConfluenceDetectorTest, AllBullishIsConfluent) {
    SignalConfluenceDetector::Config cfg;
    cfg.window    = 5;
    cfg.threshold = 0.6;
    SignalConfluenceDetector d(cfg);
    for (int i = 0; i < 5; ++i) d.record(0.5);
    EXPECT_TRUE(d.is_confluent());
    EXPECT_EQ(d.dominant_direction(), 1);
    EXPECT_GT(d.confluence_score(), 0.5);
}

TEST(SignalConfluenceDetectorTest, AllBearishIsConfluent) {
    SignalConfluenceDetector::Config cfg;
    cfg.window    = 5;
    cfg.threshold = 0.6;
    SignalConfluenceDetector d(cfg);
    for (int i = 0; i < 5; ++i) d.record(-0.3);
    EXPECT_TRUE(d.is_confluent());
    EXPECT_EQ(d.dominant_direction(), -1);
}

TEST(SignalConfluenceDetectorTest, MixedIsNotConfluent) {
    SignalConfluenceDetector::Config cfg;
    cfg.window    = 6;
    cfg.threshold = 0.6;
    SignalConfluenceDetector d(cfg);
    for (int i = 0; i < 3; ++i) d.record(0.5);
    for (int i = 0; i < 3; ++i) d.record(-0.5);
    EXPECT_FALSE(d.is_confluent());
}

TEST(SignalConfluenceDetectorTest, ConfluenceCallback) {
    SignalConfluenceDetector::Config cfg;
    cfg.window    = 5;
    cfg.threshold = 0.6;
    int fires = 0;
    cfg.on_confluence = [&](double, int) { ++fires; };
    SignalConfluenceDetector d(cfg);
    for (int i = 0; i < 8; ++i) d.record(0.5);
    EXPECT_GE(fires, 1);
}

TEST(SignalConfluenceDetectorTest, Reset) {
    SignalConfluenceDetector d;
    for (int i = 0; i < 10; ++i) d.record(0.5);
    d.reset();
    EXPECT_EQ(d.observation_count(), 0u);
    EXPECT_FALSE(d.is_confluent());
}

TEST(SignalConfluenceDetectorTest, StatsJson) {
    SignalConfluenceDetector d;
    d.record(0.3);
    std::string json = d.to_stats_json();
    EXPECT_NE(json.find("confluence_score"), std::string::npos);
    EXPECT_NE(json.find("dominant_dir"), std::string::npos);
}

#else
TEST(SignalConfluenceDetectorTest, Disabled) { SUCCEED(); }
#endif
