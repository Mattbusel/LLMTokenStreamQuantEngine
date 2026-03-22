#include <gtest/gtest.h>

#ifdef LLMQUANT_STREAM_DIFFERENCER_ENABLED
#include "TokenStreamDifferencer.h"
#include <atomic>
#include <cmath>
#include <thread>
using namespace llmquant;

TEST(TokenStreamDifferencerTest, InitialState) {
    TokenStreamDifferencer d;
    EXPECT_NEAR(d.velocity(), 0.0, 1e-9);
    EXPECT_NEAR(d.acceleration(), 0.0, 1e-9);
    EXPECT_NEAR(d.jerk(), 0.0, 1e-9);
    EXPECT_EQ(d.observation_count(), 0u);
}

TEST(TokenStreamDifferencerTest, VelocityAfterTwoObs) {
    TokenStreamDifferencer d;
    d.record(1.0);
    d.record(3.0);
    EXPECT_NEAR(d.velocity(), 2.0, 1e-9);  // 3 - 1
}

TEST(TokenStreamDifferencerTest, AccelerationAfterThreeObs) {
    TokenStreamDifferencer d;
    d.record(0.0);
    d.record(1.0);  // d1 = 1
    d.record(3.0);  // d1 = 2, d2 = 2-1 = 1
    EXPECT_NEAR(d.acceleration(), 1.0, 1e-9);
}

TEST(TokenStreamDifferencerTest, JerkAfterFourObs) {
    TokenStreamDifferencer d;
    d.record(0.0);  // seed
    d.record(1.0);  // d1=1
    d.record(3.0);  // d1=2, d2=1
    d.record(6.0);  // d1=3, d2=1, d3=0
    EXPECT_NEAR(d.jerk(), 0.0, 1e-9);

    d.record(10.0); // d1=4, d2=1, d3=0
    EXPECT_NEAR(d.jerk(), 0.0, 1e-9);
}

TEST(TokenStreamDifferencerTest, JerkNonZeroOnNonlinear) {
    TokenStreamDifferencer d;
    // Weights: 0, 1, 4, 9 → d1: 1,3,5; d2: 2,2; d3: 0
    // Try a pattern where jerk is non-zero.
    d.record(0.0);
    d.record(1.0);  // d1=1
    d.record(4.0);  // d1=3, d2=2
    d.record(10.0); // d1=6, d2=3, d3=1
    EXPECT_NEAR(d.jerk(), 1.0, 1e-9);
}

TEST(TokenStreamDifferencerTest, JerkSpikeCallbackFires) {
    TokenStreamDifferencer::Config cfg;
    cfg.jerk_threshold = 0.5;
    std::atomic<int> fires{0};
    cfg.on_jerk_spike = [&](double, double) { ++fires; };
    TokenStreamDifferencer d(cfg);

    // Create a large jerk spike.
    d.record(0.0);
    d.record(0.0);
    d.record(0.0);
    d.record(10.0);  // d1=10, d2=10, d3=10 — jerk spike
    EXPECT_GE(fires.load(), 1);
    EXPECT_GE(d.jerk_spike_count(), 1u);
}

TEST(TokenStreamDifferencerTest, EmaSmoothed) {
    TokenStreamDifferencer::Config cfg;
    cfg.ema_alpha = 0.5;
    TokenStreamDifferencer d(cfg);
    d.record(0.0);
    d.record(2.0);  // vel=2, vel_ema starts at 0+0.5*(2-0)=1
    EXPECT_NEAR(d.velocity_ema(), 1.0, 1e-9);
}

TEST(TokenStreamDifferencerTest, ResetClearsAll) {
    TokenStreamDifferencer d;
    d.record(0.0);
    d.record(1.0);
    d.record(2.0);
    d.record(3.0);
    d.reset();
    EXPECT_EQ(d.observation_count(), 0u);
    EXPECT_NEAR(d.velocity(), 0.0, 1e-9);
    EXPECT_NEAR(d.jerk(), 0.0, 1e-9);
}

TEST(TokenStreamDifferencerTest, StatsJsonNonEmpty) {
    TokenStreamDifferencer d;
    for (int i = 0; i < 5; ++i) d.record(i);
    std::string j = d.to_stats_json();
    EXPECT_NE(j.find("velocity"),          std::string::npos);
    EXPECT_NE(j.find("acceleration"),      std::string::npos);
    EXPECT_NE(j.find("jerk"),              std::string::npos);
    EXPECT_NE(j.find("jerk_spike_count"),  std::string::npos);
}

TEST(TokenStreamDifferencerTest, ObservationCountIncrements) {
    TokenStreamDifferencer d;
    d.record(1.0);
    d.record(2.0);
    d.record(3.0);
    EXPECT_EQ(d.observation_count(), 3u);
}

TEST(TokenStreamDifferencerTest, ConcurrentSafe) {
    TokenStreamDifferencer d;
    std::atomic<bool> go{false};
    std::atomic<double> w{0.0};
    auto worker = [&] {
        while (!go.load()) {}
        for (int i = 0; i < 200; ++i) {
            double val = w.fetch_add(0.01) + 0.01;
            d.record(val);
        }
    };
    std::thread t1(worker), t2(worker);
    go.store(true);
    t1.join(); t2.join();
    EXPECT_EQ(d.observation_count(), 400u);
    EXPECT_TRUE(std::isfinite(d.velocity()));
}

#else
TEST(TokenStreamDifferencerTest, DisabledAtBuildTime) { SUCCEED(); }
#endif
