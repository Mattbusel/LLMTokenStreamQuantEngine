#include <gtest/gtest.h>
#include <thread>
#include <vector>

#ifdef LLMQUANT_NARRATIVE_DRIFT_ENABLED
#include "NarrativeDriftDetector.h"
using namespace llmquant;

TEST(NarrativeDriftDetector, ConstructDefault) {
    NarrativeDriftDetector ndd;
    EXPECT_DOUBLE_EQ(ndd.upward_stat(),   0.0);
    EXPECT_DOUBLE_EQ(ndd.downward_stat(), 0.0);
    EXPECT_DOUBLE_EQ(ndd.running_mean(),  0.0);
    EXPECT_EQ(ndd.upward_alarms(),   0u);
    EXPECT_EQ(ndd.downward_alarms(), 0u);
    EXPECT_EQ(ndd.total_records(),   0u);
}

TEST(NarrativeDriftDetector, TotalRecordsIncrement) {
    NarrativeDriftDetector ndd;
    ndd.record(0.1);
    ndd.record(-0.2);
    ndd.record(0.3);
    EXPECT_EQ(ndd.total_records(), 3u);
}

TEST(NarrativeDriftDetector, RunningMeanConverges) {
    NarrativeDriftDetector ndd;
    for (int i = 0; i < 100; ++i) ndd.record(0.5);
    EXPECT_NEAR(ndd.running_mean(), 0.5, 0.01);
}

TEST(NarrativeDriftDetector, UpwardDriftDetected) {
    NarrativeDriftDetector::Config cfg;
    cfg.delta       = 0.0;
    cfg.lambda      = 0.1;
    cfg.min_samples = 5;
    int fired = 0;
    cfg.on_upward_drift = [&](double) { ++fired; };
    NarrativeDriftDetector ndd(cfg);
    // Feed low values then high values — upward drift
    for (int i = 0; i < 10; ++i) ndd.record(0.0);
    for (int i = 0; i < 20; ++i) ndd.record(1.0);
    EXPECT_GT(fired, 0);
    EXPECT_GT(ndd.upward_alarms(), 0u);
}

TEST(NarrativeDriftDetector, DownwardDriftDetected) {
    NarrativeDriftDetector::Config cfg;
    cfg.delta       = 0.0;
    cfg.lambda      = 0.1;
    cfg.min_samples = 5;
    int fired = 0;
    cfg.on_downward_drift = [&](double) { ++fired; };
    NarrativeDriftDetector ndd(cfg);
    for (int i = 0; i < 10; ++i) ndd.record(1.0);
    for (int i = 0; i < 20; ++i) ndd.record(-1.0);
    EXPECT_GT(fired, 0);
    EXPECT_GT(ndd.downward_alarms(), 0u);
}

TEST(NarrativeDriftDetector, NoDriftOnConstantSignal) {
    NarrativeDriftDetector::Config cfg;
    cfg.delta  = 0.1;
    cfg.lambda = 1.0;
    cfg.min_samples = 5;
    NarrativeDriftDetector ndd(cfg);
    for (int i = 0; i < 100; ++i) ndd.record(0.5);
    // Constant signal shouldn't produce many alarms
    EXPECT_LE(ndd.upward_alarms() + ndd.downward_alarms(), 5u);
}

TEST(NarrativeDriftDetector, CallbackReceivesMagnitude) {
    NarrativeDriftDetector::Config cfg;
    cfg.delta       = 0.0;
    cfg.lambda      = 0.05;
    cfg.min_samples = 3;
    double last_mag = -1.0;
    cfg.on_upward_drift = [&](double m) { last_mag = m; };
    NarrativeDriftDetector ndd(cfg);
    for (int i = 0; i < 5; ++i)  ndd.record(0.0);
    for (int i = 0; i < 20; ++i) ndd.record(1.0);
    EXPECT_GT(last_mag, 0.0);
}

TEST(NarrativeDriftDetector, StatisticsResetAfterAlarm) {
    NarrativeDriftDetector::Config cfg;
    cfg.delta       = 0.0;
    cfg.lambda      = 0.1;
    cfg.min_samples = 3;
    NarrativeDriftDetector ndd(cfg);
    for (int i = 0; i < 5; ++i)  ndd.record(0.0);
    for (int i = 0; i < 20; ++i) ndd.record(1.0);
    // After alarm U_ resets to 0
    EXPECT_DOUBLE_EQ(ndd.upward_stat(), 0.0);
}

TEST(NarrativeDriftDetector, Reset) {
    NarrativeDriftDetector ndd;
    for (int i = 0; i < 30; ++i) ndd.record(0.5);
    ndd.reset();
    EXPECT_DOUBLE_EQ(ndd.upward_stat(),   0.0);
    EXPECT_DOUBLE_EQ(ndd.downward_stat(), 0.0);
    EXPECT_DOUBLE_EQ(ndd.running_mean(),  0.0);
    EXPECT_EQ(ndd.upward_alarms(),   0u);
    EXPECT_EQ(ndd.downward_alarms(), 0u);
    EXPECT_EQ(ndd.total_records(),   0u);
}

TEST(NarrativeDriftDetector, UpdateConfigResetsState) {
    NarrativeDriftDetector ndd;
    for (int i = 0; i < 20; ++i) ndd.record(0.5);
    NarrativeDriftDetector::Config cfg;
    cfg.lambda = 1.0;
    ndd.update_config(cfg);
    EXPECT_EQ(ndd.total_records(), 0u);
    EXPECT_DOUBLE_EQ(ndd.running_mean(), 0.0);
}

TEST(NarrativeDriftDetector, ToStatsJsonFields) {
    NarrativeDriftDetector ndd;
    for (int i = 0; i < 10; ++i) ndd.record(0.1 * i);
    auto js = ndd.to_stats_json();
    EXPECT_NE(js.find("upward_stat"),     std::string::npos);
    EXPECT_NE(js.find("downward_stat"),   std::string::npos);
    EXPECT_NE(js.find("running_mean"),    std::string::npos);
    EXPECT_NE(js.find("upward_alarms"),   std::string::npos);
    EXPECT_NE(js.find("downward_alarms"), std::string::npos);
    EXPECT_NE(js.find("total_records"),   std::string::npos);
}

TEST(NarrativeDriftDetector, ConcurrentRecord) {
    NarrativeDriftDetector ndd;
    std::vector<std::thread> threads;
    for (int t = 0; t < 4; ++t)
        threads.emplace_back([&] {
            for (int i = 0; i < 50; ++i) ndd.record(0.1 * (i % 5));
        });
    for (auto& th : threads) th.join();
    EXPECT_GE(ndd.total_records(), 100u);
}

#else
TEST(NarrativeDriftDetector, DisabledAtBuildTime) { SUCCEED(); }
#endif
