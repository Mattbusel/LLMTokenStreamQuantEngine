#include <gtest/gtest.h>

#ifdef LLMQUANT_REVERSAL_DETECTOR_ENABLED
#include "SignalReversalDetector.h"
#include <atomic>
#include <thread>
using namespace llmquant;

TEST(SignalReversalDetectorTest, InitialState) {
    SignalReversalDetector d;
    EXPECT_EQ(d.last_reversal_direction(), 0);
    EXPECT_EQ(d.bullish_reversals(), 0u);
    EXPECT_EQ(d.bearish_reversals(), 0u);
    EXPECT_EQ(d.total_records(), 0u);
}

TEST(SignalReversalDetectorTest, TotalRecordsIncrements) {
    SignalReversalDetector d;
    d.record(0.1);
    d.record(0.2);
    EXPECT_EQ(d.total_records(), 2u);
}

TEST(SignalReversalDetectorTest, BullishReversalDetected) {
    SignalReversalDetector::Config cfg;
    cfg.reversal_fraction = 0.5;
    cfg.min_thrust        = 0.05;
    std::atomic<int> bulls{0};
    cfg.on_bullish_reversal = [&](double) { ++bulls; };
    SignalReversalDetector d(cfg);

    // Bearish thrust down then bullish counter-move
    d.record(1.0);   // seed
    d.record(0.5);   // thrust down (-0.5)
    d.record(0.8);   // counter-move up (+0.3 >= 0.5 * 0.5 = 0.25)
    EXPECT_GE(bulls.load(), 1);
    EXPECT_GE(d.bullish_reversals(), 1u);
}

TEST(SignalReversalDetectorTest, BearishReversalDetected) {
    SignalReversalDetector::Config cfg;
    cfg.reversal_fraction = 0.5;
    cfg.min_thrust        = 0.05;
    std::atomic<int> bears{0};
    cfg.on_bearish_reversal = [&](double) { ++bears; };
    SignalReversalDetector d(cfg);

    // Bullish thrust up then bearish counter-move
    d.record(0.0);   // seed
    d.record(0.6);   // thrust up (+0.6)
    d.record(0.3);   // counter-move down (-0.3 >= 0.5 * 0.6 = 0.3)
    EXPECT_GE(bears.load(), 1);
    EXPECT_GE(d.bearish_reversals(), 1u);
}

TEST(SignalReversalDetectorTest, SmallMoveBelowMinThrustIgnored) {
    SignalReversalDetector::Config cfg;
    cfg.min_thrust = 0.5;
    cfg.reversal_fraction = 0.5;
    std::atomic<int> fires{0};
    cfg.on_bullish_reversal = [&](double) { ++fires; };
    cfg.on_bearish_reversal = [&](double) { ++fires; };
    SignalReversalDetector d(cfg);

    // Tiny moves — no thrust
    d.record(0.0);
    d.record(0.01);
    d.record(0.02);
    d.record(0.01);
    EXPECT_EQ(fires.load(), 0);
}

TEST(SignalReversalDetectorTest, LastReversalDirectionUpdates) {
    SignalReversalDetector::Config cfg;
    cfg.reversal_fraction = 0.5;
    cfg.min_thrust        = 0.05;
    SignalReversalDetector d(cfg);
    d.record(0.0);
    d.record(0.6);
    d.record(0.3);  // bearish reversal
    EXPECT_EQ(d.last_reversal_direction(), -1);
}

TEST(SignalReversalDetectorTest, MultipleReversalCycles) {
    SignalReversalDetector::Config cfg;
    cfg.reversal_fraction = 0.5;
    cfg.min_thrust        = 0.05;
    SignalReversalDetector d(cfg);
    // Oscillate
    for (int i = 0; i < 5; ++i) {
        d.record(0.0);
        d.record(1.0);
        d.record(0.3);
    }
    // May have multiple reversal events
    EXPECT_GE(d.total_records(), 15u);
}

TEST(SignalReversalDetectorTest, ResetClearsState) {
    SignalReversalDetector d;
    d.record(0.0);
    d.record(1.0);
    d.record(0.3);
    d.reset();
    EXPECT_EQ(d.total_records(), 0u);
    EXPECT_EQ(d.bullish_reversals(), 0u);
    EXPECT_EQ(d.bearish_reversals(), 0u);
}

TEST(SignalReversalDetectorTest, StatsJsonHasFields) {
    SignalReversalDetector d;
    d.record(0.1);
    std::string j = d.to_stats_json();
    EXPECT_NE(j.find("last_reversal_direction"), std::string::npos);
    EXPECT_NE(j.find("bullish_reversals"),       std::string::npos);
    EXPECT_NE(j.find("bearish_reversals"),        std::string::npos);
    EXPECT_NE(j.find("total_records"),            std::string::npos);
}

TEST(SignalReversalDetectorTest, ConcurrentSafe) {
    SignalReversalDetector d;
    std::atomic<bool> go{false};
    auto worker = [&](int id) {
        while (!go.load()) {}
        for (int i = 0; i < 200; ++i)
            d.record(id % 2 == 0 ? i * 0.01 : -(i * 0.01));
    };
    std::thread t1(worker, 0), t2(worker, 1);
    go.store(true);
    t1.join(); t2.join();
    EXPECT_EQ(d.total_records(), 400u);
}

#else
TEST(SignalReversalDetectorTest, DisabledAtBuildTime) { SUCCEED(); }
#endif
