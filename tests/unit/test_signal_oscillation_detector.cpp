#include <gtest/gtest.h>

#ifdef LLMQUANT_OSCILLATION_DETECTOR_ENABLED

#include "SignalOscillationDetector.h"

#include <atomic>
#include <thread>
#include <vector>

using namespace llmquant;

namespace {

TEST(SignalOscillationDetector, DefaultState) {
    SignalOscillationDetector d;
    EXPECT_NEAR(d.zcr(), 0.0, 1e-9);
    EXPECT_FALSE(d.is_oscillating());
    EXPECT_EQ(d.oscillation_events(), 0u);
    EXPECT_EQ(d.total_records(), 0u);
}

TEST(SignalOscillationDetector, TotalRecordsIncrements) {
    SignalOscillationDetector d;
    d.record(0.1);
    d.record(0.2);
    EXPECT_EQ(d.total_records(), 2u);
}

TEST(SignalOscillationDetector, ZCRBounded) {
    SignalOscillationDetector::Config cfg;
    cfg.window      = 16;
    cfg.min_samples = 4;
    SignalOscillationDetector d(cfg);
    for (int i = 0; i < 32; ++i)
        d.record(static_cast<double>(i % 3) * 0.3 - 0.2);
    double z = d.zcr();
    EXPECT_GE(z, 0.0);
    EXPECT_LE(z, 1.0);
}

TEST(SignalOscillationDetector, AlternatingSignalHighZCR) {
    SignalOscillationDetector::Config cfg;
    cfg.window                = 16;
    cfg.min_samples           = 4;
    cfg.oscillation_threshold = 0.5;
    SignalOscillationDetector d(cfg);
    for (int i = 0; i < 32; ++i)
        d.record(i % 2 == 0 ? 1.0 : -1.0);
    EXPECT_GT(d.zcr(), 0.5);
    EXPECT_TRUE(d.is_oscillating());
}

TEST(SignalOscillationDetector, MonotonicSignalLowZCR) {
    SignalOscillationDetector::Config cfg;
    cfg.window      = 16;
    cfg.min_samples = 4;
    SignalOscillationDetector d(cfg);
    for (int i = 0; i < 32; ++i)
        d.record(static_cast<double>(i) * 0.01);
    EXPECT_NEAR(d.zcr(), 0.0, 0.1);
    EXPECT_FALSE(d.is_oscillating());
}

TEST(SignalOscillationDetector, OscillatingCallbackFires) {
    SignalOscillationDetector::Config cfg;
    cfg.window                = 8;
    cfg.min_samples           = 4;
    cfg.oscillation_threshold = 0.5;
    std::atomic<int> fires{0};
    cfg.on_oscillating = [&](double) { ++fires; };
    SignalOscillationDetector d(cfg);
    for (int i = 0; i < 32; ++i)
        d.record(i % 2 == 0 ? 1.0 : -1.0);
    EXPECT_GE(fires.load(), 1);
    EXPECT_GT(d.oscillation_events(), 0u);
}

TEST(SignalOscillationDetector, StabilizedCallbackFires) {
    SignalOscillationDetector::Config cfg;
    cfg.window                = 8;
    cfg.min_samples           = 4;
    cfg.oscillation_threshold = 0.5;
    cfg.clear_hysteresis      = 0.5;
    std::atomic<int> stabilized{0};
    cfg.on_stabilized = [&](double) { ++stabilized; };
    SignalOscillationDetector d(cfg);
    for (int i = 0; i < 16; ++i) d.record(i % 2 == 0 ? 1.0 : -1.0);
    for (int i = 0; i < 32; ++i) d.record(static_cast<double>(i) * 0.01);
    EXPECT_GE(stabilized.load(), 1);
}

TEST(SignalOscillationDetector, ConstantSeriesNoOscillation) {
    SignalOscillationDetector::Config cfg;
    cfg.window      = 16;
    cfg.min_samples = 4;
    SignalOscillationDetector d(cfg);
    for (int i = 0; i < 32; ++i) d.record(0.5);
    EXPECT_NEAR(d.zcr(), 0.0, 1e-6);
    EXPECT_FALSE(d.is_oscillating());
}

TEST(SignalOscillationDetector, ResetClearsState) {
    SignalOscillationDetector d;
    for (int i = 0; i < 20; ++i) d.record(i % 2 == 0 ? 1.0 : -1.0);
    d.reset();
    EXPECT_EQ(d.total_records(), 0u);
    EXPECT_NEAR(d.zcr(), 0.0, 1e-9);
    EXPECT_FALSE(d.is_oscillating());
}

TEST(SignalOscillationDetector, UpdateConfigResetsState) {
    SignalOscillationDetector d;
    d.record(0.5);
    SignalOscillationDetector::Config cfg2;
    cfg2.window = 8;
    d.update_config(cfg2);
    EXPECT_EQ(d.total_records(), 0u);
}

TEST(SignalOscillationDetector, StatsJsonContainsFields) {
    SignalOscillationDetector d;
    for (int i = 0; i < 10; ++i) d.record(i % 2 == 0 ? 0.5 : -0.5);
    std::string j = d.to_stats_json();
    EXPECT_NE(j.find("zcr"),                 std::string::npos);
    EXPECT_NE(j.find("is_oscillating"),      std::string::npos);
    EXPECT_NE(j.find("oscillation_events"),  std::string::npos);
    EXPECT_NE(j.find("total_records"),       std::string::npos);
}

TEST(SignalOscillationDetector, ConcurrentRecordSafe) {
    SignalOscillationDetector::Config cfg;
    cfg.window      = 16;
    cfg.min_samples = 4;
    SignalOscillationDetector d(cfg);
    std::atomic<bool> go{false};
    auto worker = [&](int id) {
        while (!go.load()) {}
        for (int i = 0; i < 250; ++i)
            d.record(id % 2 == 0 ? 1.0 : -1.0);
    };
    std::vector<std::thread> threads;
    for (int t = 0; t < 4; ++t) threads.emplace_back(worker, t);
    go.store(true);
    for (auto& t : threads) t.join();
    EXPECT_EQ(d.total_records(), 1000u);
    double z = d.zcr();
    EXPECT_GE(z, 0.0);
    EXPECT_LE(z, 1.0);
}

}  // namespace

#else

TEST(SignalOscillationDetector, DisabledAtBuildTime) { SUCCEED(); }

#endif  // LLMQUANT_OSCILLATION_DETECTOR_ENABLED
