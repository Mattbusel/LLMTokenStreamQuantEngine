#include <gtest/gtest.h>

#ifdef LLMQUANT_CLIP_MONITOR_ENABLED
#include "BiasClipMonitor.h"
using namespace llmquant;

TEST(BiasClipMonitorTest, InitialState) {
    BiasClipMonitor m;
    EXPECT_EQ(m.clip_rate(), 0.0);
    EXPECT_EQ(m.clip_count(), 0u);
    EXPECT_FALSE(m.is_spiking());
    EXPECT_EQ(m.observation_count(), 0u);
}

TEST(BiasClipMonitorTest, ObservationCountIncrement) {
    BiasClipMonitor m;
    m.record(0.1);
    m.record(0.9);
    EXPECT_EQ(m.observation_count(), 2u);
}

TEST(BiasClipMonitorTest, ClipCountAccumulates) {
    BiasClipMonitor m;
    m.record(0.8);  // clipped (>0.7)
    m.record(0.5);  // not clipped
    m.record(-0.9); // clipped
    EXPECT_EQ(m.clip_count(), 2u);
}

TEST(BiasClipMonitorTest, ClipRateRangeValid) {
    BiasClipMonitor m;
    for (int i = 0; i < 50; ++i) m.record(i % 3 == 0 ? 0.9 : 0.1);
    EXPECT_GE(m.clip_rate(), 0.0);
    EXPECT_LE(m.clip_rate(), 1.0);
}

TEST(BiasClipMonitorTest, HighClipRateDetected) {
    BiasClipMonitor::Config cfg;
    cfg.clip_threshold = 0.5;
    cfg.window         = 10;
    cfg.spike_threshold = 0.5;
    BiasClipMonitor m(cfg);
    for (int i = 0; i < 10; ++i) m.record(0.9);
    EXPECT_GT(m.clip_rate(), 0.5);
    EXPECT_TRUE(m.is_spiking());
}

TEST(BiasClipMonitorTest, ClipSpikeCallback) {
    BiasClipMonitor::Config cfg;
    cfg.clip_threshold  = 0.5;
    cfg.window          = 10;
    cfg.spike_threshold = 0.5;
    int fires = 0;
    cfg.on_clip_spike = [&](double) { ++fires; };
    BiasClipMonitor m(cfg);
    for (int i = 0; i < 10; ++i) m.record(0.9);
    EXPECT_GE(fires, 1);
}

TEST(BiasClipMonitorTest, Reset) {
    BiasClipMonitor m;
    for (int i = 0; i < 20; ++i) m.record(0.9);
    m.reset();
    EXPECT_EQ(m.observation_count(), 0u);
    EXPECT_EQ(m.clip_rate(), 0.0);
}

TEST(BiasClipMonitorTest, StatsJson) {
    BiasClipMonitor m;
    m.record(0.8);
    std::string json = m.to_stats_json();
    EXPECT_NE(json.find("clip_rate"), std::string::npos);
    EXPECT_NE(json.find("clip_count"), std::string::npos);
}

#else
TEST(BiasClipMonitorTest, Disabled) { SUCCEED(); }
#endif
