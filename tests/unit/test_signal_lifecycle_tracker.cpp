#include <gtest/gtest.h>

#ifdef LLMQUANT_LIFECYCLE_TRACKER_ENABLED
#include "SignalLifecycleTracker.h"
#include <atomic>
#include <chrono>
#include <thread>
using namespace llmquant;

TEST(SignalLifecycleTrackerTest, InitialState) {
    SignalLifecycleTracker t;
    EXPECT_EQ(t.active_signal_count(), 0u);
    EXPECT_EQ(t.dead_count(), 0u);
    EXPECT_EQ(t.zombie_count(), 0u);
}

TEST(SignalLifecycleTrackerTest, RecordSignalIncreasesCount) {
    SignalLifecycleTracker t;
    t.record_signal("sig1", 0.8);
    t.record_signal("sig2", 0.5);
    EXPECT_EQ(t.active_signal_count(), 2u);
}

TEST(SignalLifecycleTrackerTest, SignalDiesWhenDecayed) {
    SignalLifecycleTracker::Config cfg;
    cfg.decay_fraction = 0.1;
    std::atomic<int> deaths{0};
    cfg.on_death = [&](const std::string&, double, double) { ++deaths; };
    SignalLifecycleTracker t(cfg);

    t.record_signal("s", 1.0);
    t.record_signal("s", 0.05);  // decayed below 10% of peak
    t.tick();
    EXPECT_GE(deaths.load(), 1);
    EXPECT_EQ(t.dead_count(), 1u);
    EXPECT_EQ(t.active_signal_count(), 0u);
}

TEST(SignalLifecycleTrackerTest, ZombieCallbackFires) {
    SignalLifecycleTracker::Config cfg;
    cfg.max_age_s  = 0.001;  // 1 ms
    cfg.decay_fraction = 0.0; // never dies naturally
    std::atomic<int> zombies{0};
    cfg.on_zombie = [&](const std::string&, double) { ++zombies; };
    SignalLifecycleTracker t(cfg);

    t.record_signal("z", 1.0);
    std::this_thread::sleep_for(std::chrono::milliseconds(5));
    t.tick();
    EXPECT_GE(zombies.load(), 1);
    EXPECT_GE(t.zombie_count(), 1u);
}

TEST(SignalLifecycleTrackerTest, MeanHalflifeComputedOnDeath) {
    SignalLifecycleTracker::Config cfg;
    cfg.decay_fraction   = 0.1;
    cfg.halflife_fraction = 0.5;
    SignalLifecycleTracker t(cfg);

    t.record_signal("s1", 1.0);
    t.record_signal("s1", 0.4);  // below halflife 50%
    t.record_signal("s1", 0.05); // below decay 10%
    t.tick();

    EXPECT_GE(t.mean_halflife_s(), 0.0);
}

TEST(SignalLifecycleTrackerTest, UpdateSignalPeakTracked) {
    SignalLifecycleTracker::Config cfg;
    cfg.decay_fraction = 0.1;
    std::atomic<double> peak_seen{0.0};
    cfg.on_death = [&](const std::string&, double, double peak) {
        peak_seen.store(peak);
    };
    SignalLifecycleTracker t(cfg);

    t.record_signal("s", 0.5);
    t.record_signal("s", 1.2);  // new peak
    t.record_signal("s", 0.05); // below 10% of 1.2
    t.tick();
    EXPECT_NEAR(peak_seen.load(), 1.2, 1e-9);
}

TEST(SignalLifecycleTrackerTest, ResetClearsAll) {
    SignalLifecycleTracker t;
    t.record_signal("a", 0.5);
    t.record_signal("b", 0.3);
    t.reset();
    EXPECT_EQ(t.active_signal_count(), 0u);
    EXPECT_EQ(t.dead_count(), 0u);
}

TEST(SignalLifecycleTrackerTest, StatsJsonNonEmpty) {
    SignalLifecycleTracker t;
    t.record_signal("x", 0.9);
    std::string j = t.to_stats_json();
    EXPECT_NE(j.find("active_signal_count"), std::string::npos);
    EXPECT_NE(j.find("dead_count"),          std::string::npos);
    EXPECT_NE(j.find("zombie_count"),        std::string::npos);
    EXPECT_NE(j.find("mean_halflife_s"),     std::string::npos);
}

TEST(SignalLifecycleTrackerTest, ConcurrentSafe) {
    SignalLifecycleTracker t;
    std::atomic<bool> go{false};
    auto w_record = [&] {
        while (!go.load()) {}
        for (int i = 0; i < 100; ++i) {
            t.record_signal("s" + std::to_string(i % 5), 0.5);
        }
    };
    auto w_tick = [&] {
        while (!go.load()) {}
        for (int i = 0; i < 50; ++i) { t.tick(); }
    };
    std::thread t1(w_record), t2(w_tick);
    go.store(true);
    t1.join(); t2.join();
    // No crash = success.
}

#else
TEST(SignalLifecycleTrackerTest, DisabledAtBuildTime) { SUCCEED(); }
#endif
