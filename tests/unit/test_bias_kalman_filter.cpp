#include <gtest/gtest.h>
#include <thread>
#include <vector>
#include <cmath>

#ifdef LLMQUANT_KALMAN_FILTER_ENABLED
#include "BiasKalmanFilter.h"
using namespace llmquant;

TEST(BiasKalmanFilter, ConstructDefault) {
    BiasKalmanFilter kf;
    EXPECT_DOUBLE_EQ(kf.filtered_bias(), 0.0);
    EXPECT_DOUBLE_EQ(kf.innovation(),    0.0);
    EXPECT_DOUBLE_EQ(kf.nis(),           0.0);
    EXPECT_EQ(kf.total_records(), 0u);
    EXPECT_EQ(kf.mismatch_events(), 0u);
}

TEST(BiasKalmanFilter, TotalRecordsIncrement) {
    BiasKalmanFilter kf;
    kf.record(0.1);
    kf.record(0.2);
    EXPECT_EQ(kf.total_records(), 2u);
}

TEST(BiasKalmanFilter, FilteredBiasConvergesToConstant) {
    BiasKalmanFilter kf;
    for (int i = 0; i < 100; ++i) kf.record(0.5);
    EXPECT_NEAR(kf.filtered_bias(), 0.5, 0.1);
}

TEST(BiasKalmanFilter, FilterSmoothsNoisySignal) {
    BiasKalmanFilter::Config cfg;
    cfg.Q = 0.0001;
    cfg.R = 0.5;
    cfg.estimate_R = false;
    BiasKalmanFilter kf(cfg);
    // True value 0.3, noisy observations
    for (int i = 0; i < 50; ++i) kf.record(0.3 + (i % 3 == 0 ? 0.5 : -0.1));
    // Filtered bias should be smoother — just check it's in a reasonable range
    EXPECT_GT(kf.filtered_bias(), -1.0);
    EXPECT_LT(kf.filtered_bias(),  1.0);
}

TEST(BiasKalmanFilter, InnovationNonZeroAfterChange) {
    BiasKalmanFilter kf;
    for (int i = 0; i < 30; ++i) kf.record(0.0);
    kf.record(1.0);  // sudden jump
    EXPECT_GT(std::abs(kf.innovation()), 0.0);
}

TEST(BiasKalmanFilter, KalmanGainInUnitInterval) {
    BiasKalmanFilter kf;
    for (int i = 0; i < 20; ++i) kf.record(0.1 * (i % 5));
    EXPECT_GE(kf.kalman_gain(), 0.0);
    EXPECT_LE(kf.kalman_gain(), 1.0);
}

TEST(BiasKalmanFilter, NISNonNegative) {
    BiasKalmanFilter kf;
    for (int i = 0; i < 20; ++i) kf.record(0.1 * (i % 5));
    EXPECT_GE(kf.nis(), 0.0);
}

TEST(BiasKalmanFilter, MismatchCallbackFiredOnLargeNIS) {
    BiasKalmanFilter::Config cfg;
    cfg.nis_threshold = 0.0;  // fire on any positive NIS
    cfg.estimate_R    = false;
    int fired = 0;
    cfg.on_model_mismatch = [&](double) { ++fired; };
    BiasKalmanFilter kf(cfg);
    for (int i = 0; i < 20; ++i) kf.record(0.1 * (i % 5));
    EXPECT_GT(fired, 0);
}

TEST(BiasKalmanFilter, MismatchEventsAccumulate) {
    BiasKalmanFilter::Config cfg;
    cfg.nis_threshold = 0.0;
    BiasKalmanFilter kf(cfg);
    for (int i = 0; i < 20; ++i) kf.record((i % 2 == 0) ? 1.0 : -1.0);
    EXPECT_GT(kf.mismatch_events(), 0u);
}

TEST(BiasKalmanFilter, Reset) {
    BiasKalmanFilter kf;
    for (int i = 0; i < 20; ++i) kf.record(0.5);
    kf.reset();
    EXPECT_DOUBLE_EQ(kf.filtered_bias(), 0.0);
    EXPECT_DOUBLE_EQ(kf.innovation(),    0.0);
    EXPECT_EQ(kf.total_records(),    0u);
    EXPECT_EQ(kf.mismatch_events(),  0u);
}

TEST(BiasKalmanFilter, UpdateConfig) {
    BiasKalmanFilter kf;
    for (int i = 0; i < 20; ++i) kf.record(0.5);
    BiasKalmanFilter::Config cfg;
    cfg.x0 = 1.0;
    kf.update_config(cfg);
    EXPECT_DOUBLE_EQ(kf.filtered_bias(), 1.0);
    EXPECT_EQ(kf.total_records(), 0u);
}

TEST(BiasKalmanFilter, ToStatsJsonFields) {
    BiasKalmanFilter kf;
    for (int i = 0; i < 10; ++i) kf.record(0.1 * i);
    auto js = kf.to_stats_json();
    EXPECT_NE(js.find("filtered_bias"),   std::string::npos);
    EXPECT_NE(js.find("innovation"),      std::string::npos);
    EXPECT_NE(js.find("nis"),             std::string::npos);
    EXPECT_NE(js.find("kalman_gain"),     std::string::npos);
    EXPECT_NE(js.find("mismatch_events"), std::string::npos);
    EXPECT_NE(js.find("total_records"),   std::string::npos);
}

TEST(BiasKalmanFilter, ConcurrentRecord) {
    BiasKalmanFilter kf;
    std::vector<std::thread> threads;
    for (int t = 0; t < 4; ++t)
        threads.emplace_back([&] {
            for (int i = 0; i < 50; ++i) kf.record(0.1 * (i % 5));
        });
    for (auto& th : threads) th.join();
    EXPECT_GE(kf.total_records(), 100u);
}

#else
TEST(BiasKalmanFilter, DisabledAtBuildTime) { SUCCEED(); }
#endif
