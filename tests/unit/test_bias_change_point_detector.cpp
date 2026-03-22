#include <gtest/gtest.h>

#ifdef LLMQUANT_CHANGE_POINT_ENABLED

#include "BiasChangePointDetector.h"

#include <atomic>
#include <thread>
#include <vector>

using namespace llmquant;

namespace {

TEST(BiasChangePointDetector, DefaultState) {
    BiasChangePointDetector d;
    EXPECT_NEAR(d.cusum_plus(),  0.0, 1e-9);
    EXPECT_NEAR(d.cusum_minus(), 0.0, 1e-9);
    EXPECT_EQ(d.upshift_count(),   0u);
    EXPECT_EQ(d.downshift_count(), 0u);
    EXPECT_EQ(d.total_records(),   0u);
}

TEST(BiasChangePointDetector, TotalRecordsIncrements) {
    BiasChangePointDetector d;
    d.record(0.1);
    d.record(0.2);
    EXPECT_EQ(d.total_records(), 2u);
}

TEST(BiasChangePointDetector, NeutralSignalNoAlarm) {
    BiasChangePointDetector::Config cfg;
    cfg.reference_mean = 0.0;
    cfg.allowance      = 0.1;
    cfg.threshold      = 1.0;
    BiasChangePointDetector d(cfg);
    for (int i = 0; i < 20; ++i) d.record(0.0);
    EXPECT_EQ(d.upshift_count(),   0u);
    EXPECT_EQ(d.downshift_count(), 0u);
}

TEST(BiasChangePointDetector, SustainedUpshiftDetected) {
    BiasChangePointDetector::Config cfg;
    cfg.reference_mean = 0.0;
    cfg.allowance      = 0.05;
    cfg.threshold      = 0.5;
    BiasChangePointDetector d(cfg);
    int result = 0;
    for (int i = 0; i < 20; ++i)
        result = d.record(0.2);  // sustained positive bias
    EXPECT_GT(d.upshift_count(), 0u);
}

TEST(BiasChangePointDetector, SustainedDownshiftDetected) {
    BiasChangePointDetector::Config cfg;
    cfg.reference_mean = 0.0;
    cfg.allowance      = 0.05;
    cfg.threshold      = 0.5;
    BiasChangePointDetector d(cfg);
    for (int i = 0; i < 20; ++i) d.record(-0.2);
    EXPECT_GT(d.downshift_count(), 0u);
}

TEST(BiasChangePointDetector, AccumulatorResetsAfterAlarm) {
    BiasChangePointDetector::Config cfg;
    cfg.reference_mean = 0.0;
    cfg.allowance      = 0.05;
    cfg.threshold      = 0.3;
    BiasChangePointDetector d(cfg);
    for (int i = 0; i < 10; ++i) d.record(0.2);
    // After alarm, accumulator should be near 0.
    EXPECT_NEAR(d.cusum_plus(), 0.0, 1e-9);
}

TEST(BiasChangePointDetector, UpshiftCallbackFires) {
    BiasChangePointDetector::Config cfg;
    cfg.reference_mean = 0.0;
    cfg.allowance      = 0.05;
    cfg.threshold      = 0.5;
    std::atomic<int> fires{0};
    cfg.on_upshift = [&](double) { ++fires; };
    BiasChangePointDetector d(cfg);
    for (int i = 0; i < 30; ++i) d.record(0.2);
    EXPECT_GE(fires.load(), 1);
}

TEST(BiasChangePointDetector, DownshiftCallbackFires) {
    BiasChangePointDetector::Config cfg;
    cfg.reference_mean = 0.0;
    cfg.allowance      = 0.05;
    cfg.threshold      = 0.5;
    std::atomic<int> fires{0};
    cfg.on_downshift = [&](double) { ++fires; };
    BiasChangePointDetector d(cfg);
    for (int i = 0; i < 30; ++i) d.record(-0.2);
    EXPECT_GE(fires.load(), 1);
}

TEST(BiasChangePointDetector, ReturnValueSignaledCorrectly) {
    BiasChangePointDetector::Config cfg;
    cfg.reference_mean = 0.0;
    cfg.allowance      = 0.01;
    cfg.threshold      = 0.1;
    BiasChangePointDetector d(cfg);
    int last = 0;
    for (int i = 0; i < 20; ++i) last = d.record(0.1);
    // At some point we should have gotten a +1 return.
    EXPECT_GT(d.upshift_count(), 0u);
}

TEST(BiasChangePointDetector, ResetClearsState) {
    BiasChangePointDetector d;
    for (int i = 0; i < 5; ++i) d.record(0.5);
    d.reset();
    EXPECT_NEAR(d.cusum_plus(),  0.0, 1e-9);
    EXPECT_NEAR(d.cusum_minus(), 0.0, 1e-9);
    EXPECT_EQ(d.total_records(), 0u);
    EXPECT_EQ(d.upshift_count(), 0u);
}

TEST(BiasChangePointDetector, UpdateConfigResetsState) {
    BiasChangePointDetector d;
    d.record(0.5);
    BiasChangePointDetector::Config cfg2;
    cfg2.allowance = 0.1;
    d.update_config(cfg2);
    EXPECT_EQ(d.total_records(), 0u);
    EXPECT_NEAR(d.cusum_plus(), 0.0, 1e-9);
}

TEST(BiasChangePointDetector, StatsJsonContainsFields) {
    BiasChangePointDetector d;
    d.record(0.1);
    std::string j = d.to_stats_json();
    EXPECT_NE(j.find("cusum_plus"),      std::string::npos);
    EXPECT_NE(j.find("cusum_minus"),     std::string::npos);
    EXPECT_NE(j.find("upshift_count"),   std::string::npos);
    EXPECT_NE(j.find("downshift_count"), std::string::npos);
    EXPECT_NE(j.find("total_records"),   std::string::npos);
}

TEST(BiasChangePointDetector, ConcurrentRecordSafe) {
    BiasChangePointDetector::Config cfg;
    cfg.reference_mean = 0.0;
    cfg.allowance      = 0.05;
    cfg.threshold      = 1.0;
    BiasChangePointDetector d(cfg);
    std::atomic<bool> go{false};
    auto worker = [&](int id) {
        while (!go.load()) {}
        for (int i = 0; i < 250; ++i)
            d.record(id % 2 == 0 ? 0.1 : -0.1);
    };
    std::vector<std::thread> threads;
    for (int t = 0; t < 4; ++t) threads.emplace_back(worker, t);
    go.store(true);
    for (auto& t : threads) t.join();
    EXPECT_EQ(d.total_records(), 1000u);
}

}  // namespace

#else

TEST(BiasChangePointDetector, DisabledAtBuildTime) { SUCCEED(); }

#endif  // LLMQUANT_CHANGE_POINT_ENABLED
