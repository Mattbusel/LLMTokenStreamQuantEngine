#include <gtest/gtest.h>

#ifdef LLMQUANT_HURST_ESTIMATOR_ENABLED

#include "SignalHurstEstimator.h"

#include <atomic>
#include <cmath>
#include <thread>
#include <vector>

using namespace llmquant;

namespace {

TEST(SignalHurstEstimator, DefaultState) {
    SignalHurstEstimator s;
    EXPECT_NEAR(s.hurst(), 0.5, 1e-9);
    EXPECT_FALSE(s.is_trending());
    EXPECT_FALSE(s.is_mean_reverting());
    EXPECT_EQ(s.total_records(), 0u);
    EXPECT_EQ(s.trend_events(), 0u);
    EXPECT_EQ(s.revert_events(), 0u);
}

TEST(SignalHurstEstimator, TotalRecordsIncrements) {
    SignalHurstEstimator s;
    s.record(0.1);
    s.record(0.2);
    s.record(0.3);
    EXPECT_EQ(s.total_records(), 3u);
}

TEST(SignalHurstEstimator, HurstBounded) {
    SignalHurstEstimator::Config cfg;
    cfg.window      = 32;
    cfg.min_samples = 8;
    SignalHurstEstimator s(cfg);
    for (int i = 0; i < 64; ++i)
        s.record(static_cast<double>(i % 5) * 0.2 - 0.4);
    double h = s.hurst();
    EXPECT_GE(h, 0.0);
    EXPECT_LE(h, 1.0);
}

TEST(SignalHurstEstimator, TrendingSeriesHighHurst) {
    // Monotonically increasing series → H close to 1.
    SignalHurstEstimator::Config cfg;
    cfg.window           = 32;
    cfg.min_samples      = 16;
    cfg.trending_threshold = 0.55;
    SignalHurstEstimator s(cfg);
    for (int i = 0; i < 64; ++i)
        s.record(static_cast<double>(i) * 0.01);
    EXPECT_GT(s.hurst(), 0.5);
}

TEST(SignalHurstEstimator, MeanRevertingSeriesLowHurst) {
    // Alternating ±1 series → strongly anti-persistent → H < 0.5.
    SignalHurstEstimator::Config cfg;
    cfg.window              = 32;
    cfg.min_samples         = 16;
    cfg.reverting_threshold = 0.45;
    SignalHurstEstimator s(cfg);
    for (int i = 0; i < 64; ++i)
        s.record(i % 2 == 0 ? 1.0 : -1.0);
    EXPECT_LT(s.hurst(), 0.5);
}

TEST(SignalHurstEstimator, ConstantSeriesDoesNotCrash) {
    // Constant input → zero variance → should return H=0.5 (degenerate).
    SignalHurstEstimator::Config cfg;
    cfg.window      = 16;
    cfg.min_samples = 8;
    SignalHurstEstimator s(cfg);
    for (int i = 0; i < 32; ++i) s.record(0.5);
    double h = s.hurst();
    EXPECT_GE(h, 0.0);
    EXPECT_LE(h, 1.0);
}

TEST(SignalHurstEstimator, TrendingCallbackFires) {
    SignalHurstEstimator::Config cfg;
    cfg.window             = 16;
    cfg.min_samples        = 8;
    cfg.trending_threshold = 0.55;
    std::atomic<int> fires{0};
    cfg.on_trending = [&](double) { ++fires; };
    SignalHurstEstimator s(cfg);
    for (int i = 0; i < 64; ++i)
        s.record(static_cast<double>(i) * 0.01);
    EXPECT_GE(fires.load(), 1);
    EXPECT_GT(s.trend_events(), 0u);
    EXPECT_TRUE(s.is_trending());
}

TEST(SignalHurstEstimator, RevertingCallbackFires) {
    SignalHurstEstimator::Config cfg;
    cfg.window              = 16;
    cfg.min_samples         = 8;
    cfg.reverting_threshold = 0.45;
    std::atomic<int> fires{0};
    cfg.on_mean_reverting = [&](double) { ++fires; };
    SignalHurstEstimator s(cfg);
    for (int i = 0; i < 64; ++i)
        s.record(i % 2 == 0 ? 1.0 : -1.0);
    EXPECT_GE(fires.load(), 1);
    EXPECT_GT(s.revert_events(), 0u);
    EXPECT_TRUE(s.is_mean_reverting());
}

TEST(SignalHurstEstimator, ResetClearsState) {
    SignalHurstEstimator s;
    for (int i = 0; i < 32; ++i) s.record(static_cast<double>(i) * 0.1);
    s.reset();
    EXPECT_EQ(s.total_records(), 0u);
    EXPECT_NEAR(s.hurst(), 0.5, 1e-9);
    EXPECT_FALSE(s.is_trending());
    EXPECT_FALSE(s.is_mean_reverting());
}

TEST(SignalHurstEstimator, UpdateConfigResetsState) {
    SignalHurstEstimator s;
    for (int i = 0; i < 32; ++i) s.record(static_cast<double>(i) * 0.1);
    SignalHurstEstimator::Config cfg2;
    cfg2.window = 16;
    s.update_config(cfg2);
    EXPECT_EQ(s.total_records(), 0u);
}

TEST(SignalHurstEstimator, StatsJsonContainsFields) {
    SignalHurstEstimator s;
    for (int i = 0; i < 20; ++i) s.record(static_cast<double>(i) * 0.05);
    std::string j = s.to_stats_json();
    EXPECT_NE(j.find("hurst"),         std::string::npos);
    EXPECT_NE(j.find("is_trending"),   std::string::npos);
    EXPECT_NE(j.find("is_reverting"),  std::string::npos);
    EXPECT_NE(j.find("trend_events"),  std::string::npos);
    EXPECT_NE(j.find("revert_events"), std::string::npos);
    EXPECT_NE(j.find("total_records"), std::string::npos);
}

TEST(SignalHurstEstimator, ConcurrentRecordSafe) {
    SignalHurstEstimator::Config cfg;
    cfg.window      = 64;
    cfg.min_samples = 16;
    SignalHurstEstimator s(cfg);
    std::atomic<bool> go{false};
    auto worker = [&](int id) {
        while (!go.load()) {}
        for (int i = 0; i < 250; ++i)
            s.record(id % 2 == 0 ? static_cast<double>(i) * 0.01
                                 : static_cast<double>(i % 5) * 0.2 - 0.4);
    };
    std::vector<std::thread> threads;
    for (int t = 0; t < 4; ++t) threads.emplace_back(worker, t);
    go.store(true);
    for (auto& t : threads) t.join();
    EXPECT_EQ(s.total_records(), 1000u);
    double h = s.hurst();
    EXPECT_GE(h, 0.0);
    EXPECT_LE(h, 1.0);
}

}  // namespace

#else

TEST(SignalHurstEstimator, DisabledAtBuildTime) { SUCCEED(); }

#endif  // LLMQUANT_HURST_ESTIMATOR_ENABLED
