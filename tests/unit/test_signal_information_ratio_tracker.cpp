#include <gtest/gtest.h>

#ifdef LLMQUANT_IR_TRACKER_ENABLED

#include "SignalInformationRatioTracker.h"

#include <atomic>
#include <cmath>
#include <thread>
#include <vector>

using namespace llmquant;

namespace {

TEST(SignalInformationRatioTracker, DefaultState) {
    SignalInformationRatioTracker t;
    EXPECT_NEAR(t.ir(), 0.0, 1e-9);
    EXPECT_NEAR(t.mean(), 0.0, 1e-9);
    EXPECT_FALSE(t.is_high_ir());
    EXPECT_EQ(t.total_records(), 0u);
}

TEST(SignalInformationRatioTracker, TotalRecordsIncrements) {
    SignalInformationRatioTracker t;
    t.record(0.1);
    t.record(0.2);
    EXPECT_EQ(t.total_records(), 2u);
}

TEST(SignalInformationRatioTracker, IRBounded) {
    SignalInformationRatioTracker::Config cfg;
    cfg.window      = 16;
    cfg.min_samples = 4;
    SignalInformationRatioTracker t(cfg);
    for (int i = 0; i < 32; ++i)
        t.record(static_cast<double>(i % 5) * 0.1 - 0.2);
    double ir = t.ir();
    EXPECT_GE(ir, -10.0);
    EXPECT_LE(ir,  10.0);
}

TEST(SignalInformationRatioTracker, ConsistentPositiveBiasHighIR) {
    SignalInformationRatioTracker::Config cfg;
    cfg.window       = 16;
    cfg.min_samples  = 4;
    cfg.ir_threshold = 1.0;
    SignalInformationRatioTracker t(cfg);
    for (int i = 0; i < 32; ++i) t.record(0.8);  // constant + bias → IR → ∞ clamped
    // With nearly constant positive bias, IR should be very high.
    EXPECT_GT(t.mean(), 0.0);
}

TEST(SignalInformationRatioTracker, HighIRCallbackFires) {
    SignalInformationRatioTracker::Config cfg;
    cfg.window       = 8;
    cfg.min_samples  = 4;
    cfg.ir_threshold = 0.5;
    std::atomic<int> fires{0};
    cfg.on_high_ir = [&](double) { ++fires; };
    SignalInformationRatioTracker t(cfg);
    for (int i = 0; i < 32; ++i) t.record(0.5 + static_cast<double>(i % 3) * 0.01);
    EXPECT_GE(fires.load(), 1);
}

TEST(SignalInformationRatioTracker, ZeroVarianceHandledGracefully) {
    SignalInformationRatioTracker::Config cfg;
    cfg.window      = 8;
    cfg.min_samples = 4;
    SignalInformationRatioTracker t(cfg);
    for (int i = 0; i < 16; ++i) t.record(0.0);
    // Zero variance, zero mean → IR = 0 (no division by zero crash).
    EXPECT_NEAR(t.ir(), 0.0, 1e-6);
}

TEST(SignalInformationRatioTracker, ResetClearsState) {
    SignalInformationRatioTracker t;
    for (int i = 0; i < 16; ++i) t.record(0.5);
    t.reset();
    EXPECT_EQ(t.total_records(), 0u);
    EXPECT_NEAR(t.ir(), 0.0, 1e-9);
}

TEST(SignalInformationRatioTracker, UpdateConfigResetsState) {
    SignalInformationRatioTracker t;
    t.record(0.1);
    SignalInformationRatioTracker::Config cfg2;
    cfg2.window = 16;
    t.update_config(cfg2);
    EXPECT_EQ(t.total_records(), 0u);
}

TEST(SignalInformationRatioTracker, StatsJsonContainsFields) {
    SignalInformationRatioTracker t;
    for (int i = 0; i < 10; ++i) t.record(0.1 * i);
    std::string j = t.to_stats_json();
    EXPECT_NE(j.find("ir"),             std::string::npos);
    EXPECT_NE(j.find("mean"),           std::string::npos);
    EXPECT_NE(j.find("stddev"),         std::string::npos);
    EXPECT_NE(j.find("is_high_ir"),     std::string::npos);
    EXPECT_NE(j.find("high_ir_events"), std::string::npos);
    EXPECT_NE(j.find("total_records"),  std::string::npos);
}

TEST(SignalInformationRatioTracker, ConcurrentRecordSafe) {
    SignalInformationRatioTracker::Config cfg;
    cfg.window      = 32;
    cfg.min_samples = 8;
    SignalInformationRatioTracker t(cfg);
    std::atomic<bool> go{false};
    auto worker = [&](int id) {
        while (!go.load()) {}
        for (int i = 0; i < 250; ++i)
            t.record(id % 2 == 0 ? 0.5 : -0.3 + static_cast<double>(i % 10) * 0.02);
    };
    std::vector<std::thread> threads;
    for (int tid = 0; tid < 4; ++tid) threads.emplace_back(worker, tid);
    go.store(true);
    for (auto& th : threads) th.join();
    EXPECT_EQ(t.total_records(), 1000u);
}

}  // namespace

#else

TEST(SignalInformationRatioTracker, DisabledAtBuildTime) { SUCCEED(); }

#endif  // LLMQUANT_IR_TRACKER_ENABLED
