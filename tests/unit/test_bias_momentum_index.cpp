#include <gtest/gtest.h>

#ifdef LLMQUANT_MOMENTUM_INDEX_ENABLED

#include "BiasMomentumIndex.h"

#include <atomic>
#include <cmath>
#include <thread>
#include <vector>

using namespace llmquant;

namespace {

TEST(BiasMomentumIndex, DefaultState) {
    BiasMomentumIndex m;
    EXPECT_NEAR(m.macd_line(),   0.0, 1e-9);
    EXPECT_NEAR(m.signal_line(), 0.0, 1e-9);
    EXPECT_NEAR(m.histogram(),   0.0, 1e-9);
    EXPECT_EQ(m.total_records(), 0u);
}

TEST(BiasMomentumIndex, TotalRecordsIncrements) {
    BiasMomentumIndex m;
    m.record(0.5);
    m.record(0.6);
    EXPECT_EQ(m.total_records(), 2u);
}

TEST(BiasMomentumIndex, ReturnValueIsHistogram) {
    BiasMomentumIndex m;
    double h = m.record(0.5);
    EXPECT_DOUBLE_EQ(h, m.histogram());
}

TEST(BiasMomentumIndex, UpwardTrendBullish) {
    BiasMomentumIndex m;
    for (int i = 0; i < 50; ++i)
        m.record(static_cast<double>(i) * 0.02);
    // Fast EMA should exceed slow EMA in uptrend → bullish.
    EXPECT_GT(m.fast_ema(), m.slow_ema());
}

TEST(BiasMomentumIndex, DownwardTrendBearish) {
    BiasMomentumIndex m;
    for (int i = 0; i < 50; ++i)
        m.record(1.0 - static_cast<double>(i) * 0.02);
    EXPECT_LT(m.fast_ema(), m.slow_ema());
}

TEST(BiasMomentumIndex, BullishCrossoverCallback) {
    BiasMomentumIndex::Config cfg;
    std::atomic<int> fires{0};
    cfg.on_bullish_crossover = [&](double) { ++fires; };
    BiasMomentumIndex m(cfg);
    // Seed downtrend then reverse.
    for (int i = 0; i < 30; ++i) m.record(-0.5 + static_cast<double>(i % 5) * 0.1);
    for (int i = 0; i < 30; ++i) m.record(0.5 + static_cast<double>(i) * 0.01);
    EXPECT_GE(fires.load(), 1);
}

TEST(BiasMomentumIndex, BearishCrossoverCallback) {
    BiasMomentumIndex::Config cfg;
    std::atomic<int> fires{0};
    cfg.on_bearish_crossover = [&](double) { ++fires; };
    BiasMomentumIndex m(cfg);
    for (int i = 0; i < 30; ++i) m.record(0.5);
    for (int i = 0; i < 30; ++i) m.record(-static_cast<double>(i) * 0.03);
    EXPECT_GE(fires.load(), 1);
}

TEST(BiasMomentumIndex, ResetClearsState) {
    BiasMomentumIndex m;
    for (int i = 0; i < 20; ++i) m.record(0.5);
    m.reset();
    EXPECT_EQ(m.total_records(), 0u);
    EXPECT_NEAR(m.macd_line(), 0.0, 1e-9);
    EXPECT_NEAR(m.histogram(), 0.0, 1e-9);
}

TEST(BiasMomentumIndex, UpdateConfigResetsState) {
    BiasMomentumIndex m;
    m.record(0.5);
    BiasMomentumIndex::Config cfg2;
    cfg2.fast_alpha = 0.3;
    m.update_config(cfg2);
    EXPECT_EQ(m.total_records(), 0u);
}

TEST(BiasMomentumIndex, StatsJsonContainsFields) {
    BiasMomentumIndex m;
    for (int i = 0; i < 10; ++i) m.record(static_cast<double>(i) * 0.1);
    std::string j = m.to_stats_json();
    EXPECT_NE(j.find("fast_ema"),           std::string::npos);
    EXPECT_NE(j.find("slow_ema"),           std::string::npos);
    EXPECT_NE(j.find("macd_line"),          std::string::npos);
    EXPECT_NE(j.find("signal_line"),        std::string::npos);
    EXPECT_NE(j.find("histogram"),          std::string::npos);
    EXPECT_NE(j.find("is_bullish"),         std::string::npos);
    EXPECT_NE(j.find("bullish_crossovers"), std::string::npos);
    EXPECT_NE(j.find("total_records"),      std::string::npos);
}

TEST(BiasMomentumIndex, ConcurrentRecordSafe) {
    BiasMomentumIndex m;
    std::atomic<bool> go{false};
    auto worker = [&](int id) {
        while (!go.load()) {}
        for (int i = 0; i < 250; ++i)
            m.record(id % 2 == 0 ? static_cast<double>(i) * 0.01 : -static_cast<double>(i) * 0.01);
    };
    std::vector<std::thread> threads;
    for (int t = 0; t < 4; ++t) threads.emplace_back(worker, t);
    go.store(true);
    for (auto& t : threads) t.join();
    EXPECT_EQ(m.total_records(), 1000u);
}

}  // namespace

#else

TEST(BiasMomentumIndex, DisabledAtBuildTime) { SUCCEED(); }

#endif  // LLMQUANT_MOMENTUM_INDEX_ENABLED
