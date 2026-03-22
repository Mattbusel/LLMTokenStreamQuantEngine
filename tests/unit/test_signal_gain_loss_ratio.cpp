#include <gtest/gtest.h>

#ifdef LLMQUANT_GAIN_LOSS_RATIO_ENABLED

#include "SignalGainLossRatio.h"

#include <atomic>
#include <thread>
#include <vector>

using namespace llmquant;

namespace {

TEST(SignalGainLossRatio, DefaultState) {
    SignalGainLossRatio r;
    EXPECT_NEAR(r.ratio(), 1.0, 1e-9);
    EXPECT_NEAR(r.gain_sum(), 0.0, 1e-9);
    EXPECT_NEAR(r.loss_sum(), 0.0, 1e-9);
    EXPECT_EQ(r.total_records(), 0u);
}

TEST(SignalGainLossRatio, TotalRecordsIncrements) {
    SignalGainLossRatio r;
    r.record(0.1);
    r.record(0.2);
    EXPECT_EQ(r.total_records(), 2u);
}

TEST(SignalGainLossRatio, AllGainsHighRatio) {
    SignalGainLossRatio::Config cfg;
    cfg.window      = 16;
    cfg.min_samples = 4;
    SignalGainLossRatio r(cfg);
    for (int i = 0; i < 20; ++i) r.record(static_cast<double>(i) * 0.1);
    EXPECT_GT(r.ratio(), 1.0);
    EXPECT_GT(r.gain_sum(), 0.0);
}

TEST(SignalGainLossRatio, AllLossesLowRatio) {
    SignalGainLossRatio::Config cfg;
    cfg.window      = 16;
    cfg.min_samples = 4;
    SignalGainLossRatio r(cfg);
    for (int i = 0; i < 20; ++i) r.record(-static_cast<double>(i) * 0.1);
    EXPECT_LT(r.ratio(), 1.0);
    EXPECT_GT(r.loss_sum(), 0.0);
}

TEST(SignalGainLossRatio, HighGainCallbackFires) {
    SignalGainLossRatio::Config cfg;
    cfg.window              = 8;
    cfg.min_samples         = 4;
    cfg.high_gain_threshold = 1.2;
    std::atomic<int> fires{0};
    cfg.on_high_gain = [&](double) { ++fires; };
    SignalGainLossRatio r(cfg);
    for (int i = 0; i < 30; ++i) r.record(static_cast<double>(i) * 0.05);
    EXPECT_GE(fires.load(), 1);
}

TEST(SignalGainLossRatio, HighLossCallbackFires) {
    SignalGainLossRatio::Config cfg;
    cfg.window              = 8;
    cfg.min_samples         = 4;
    cfg.high_loss_threshold = 0.8;
    std::atomic<int> fires{0};
    cfg.on_high_loss = [&](double) { ++fires; };
    SignalGainLossRatio r(cfg);
    for (int i = 0; i < 30; ++i) r.record(-static_cast<double>(i) * 0.05);
    EXPECT_GE(fires.load(), 1);
}

TEST(SignalGainLossRatio, ConstantBiasBalancedRatio) {
    SignalGainLossRatio::Config cfg;
    cfg.window      = 16;
    cfg.min_samples = 4;
    SignalGainLossRatio r(cfg);
    for (int i = 0; i < 32; ++i) r.record(0.5);  // zero delta every step
    // No gains, no losses after first tick.
    EXPECT_NEAR(r.gain_sum(), 0.0, 1e-6);
}

TEST(SignalGainLossRatio, RatioCappedAtMaxRatio) {
    SignalGainLossRatio::Config cfg;
    cfg.window      = 16;
    cfg.min_samples = 4;
    cfg.max_ratio   = 5.0;
    SignalGainLossRatio r(cfg);
    for (int i = 0; i < 32; ++i) r.record(static_cast<double>(i) * 1.0);
    EXPECT_LE(r.ratio(), cfg.max_ratio + 1e-9);
}

TEST(SignalGainLossRatio, ResetClearsState) {
    SignalGainLossRatio r;
    r.record(0.1);
    r.record(0.2);
    r.reset();
    EXPECT_EQ(r.total_records(), 0u);
    EXPECT_NEAR(r.gain_sum(), 0.0, 1e-9);
}

TEST(SignalGainLossRatio, UpdateConfigResetsState) {
    SignalGainLossRatio r;
    r.record(0.5);
    SignalGainLossRatio::Config cfg2;
    cfg2.window = 16;
    r.update_config(cfg2);
    EXPECT_EQ(r.total_records(), 0u);
}

TEST(SignalGainLossRatio, StatsJsonContainsFields) {
    SignalGainLossRatio r;
    r.record(0.1);
    r.record(-0.05);
    std::string j = r.to_stats_json();
    EXPECT_NE(j.find("ratio"),            std::string::npos);
    EXPECT_NE(j.find("gain_sum"),         std::string::npos);
    EXPECT_NE(j.find("loss_sum"),         std::string::npos);
    EXPECT_NE(j.find("high_gain_events"), std::string::npos);
    EXPECT_NE(j.find("high_loss_events"), std::string::npos);
    EXPECT_NE(j.find("total_records"),    std::string::npos);
}

TEST(SignalGainLossRatio, ConcurrentRecordSafe) {
    SignalGainLossRatio::Config cfg;
    cfg.window      = 32;
    cfg.min_samples = 8;
    SignalGainLossRatio r(cfg);
    std::atomic<bool> go{false};
    auto worker = [&](int id) {
        while (!go.load()) {}
        for (int i = 0; i < 250; ++i)
            r.record(id % 2 == 0 ? static_cast<double>(i) * 0.01 : -static_cast<double>(i) * 0.01);
    };
    std::vector<std::thread> threads;
    for (int t = 0; t < 4; ++t) threads.emplace_back(worker, t);
    go.store(true);
    for (auto& t : threads) t.join();
    EXPECT_EQ(r.total_records(), 1000u);
    EXPECT_GE(r.ratio(), 0.0);
}

}  // namespace

#else

TEST(SignalGainLossRatio, DisabledAtBuildTime) { SUCCEED(); }

#endif  // LLMQUANT_GAIN_LOSS_RATIO_ENABLED
