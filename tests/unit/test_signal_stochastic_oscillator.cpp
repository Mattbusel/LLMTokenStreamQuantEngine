#include <gtest/gtest.h>

#ifdef LLMQUANT_STOCHASTIC_OSC_ENABLED

#include "SignalStochasticOscillator.h"

#include <atomic>
#include <cmath>
#include <thread>
#include <vector>

using namespace llmquant;

namespace {

TEST(SignalStochasticOscillator, DefaultState) {
    SignalStochasticOscillator s;
    EXPECT_TRUE(std::isnan(s.k_pct()));
    EXPECT_TRUE(std::isnan(s.d_pct()));
    EXPECT_FALSE(s.is_overbought());
    EXPECT_FALSE(s.is_oversold());
    EXPECT_EQ(s.total_records(), 0u);
}

TEST(SignalStochasticOscillator, TotalRecordsIncrements) {
    SignalStochasticOscillator s;
    s.record(0.5);
    s.record(0.3);
    EXPECT_EQ(s.total_records(), 2u);
}

TEST(SignalStochasticOscillator, NaNBeforeMinSamples) {
    SignalStochasticOscillator::Config cfg;
    cfg.k_period    = 14;
    cfg.min_samples = 14;
    SignalStochasticOscillator s(cfg);
    for (int i = 0; i < 13; ++i) s.record(static_cast<double>(i));
    EXPECT_TRUE(std::isnan(s.k_pct()));
}

TEST(SignalStochasticOscillator, KPctAtMaxWhenAtHigh) {
    SignalStochasticOscillator::Config cfg;
    cfg.k_period    = 5;
    cfg.d_period    = 3;
    cfg.min_samples = 5;
    SignalStochasticOscillator s(cfg);
    for (int i = 1; i <= 5; ++i) s.record(static_cast<double>(i));
    EXPECT_NEAR(s.k_pct(), 100.0, 1e-6);
}

TEST(SignalStochasticOscillator, KPctAtMinWhenAtLow) {
    SignalStochasticOscillator::Config cfg;
    cfg.k_period    = 5;
    cfg.min_samples = 5;
    SignalStochasticOscillator s(cfg);
    for (int i = 5; i >= 1; --i) s.record(static_cast<double>(i));
    EXPECT_NEAR(s.k_pct(), 0.0, 1e-6);
}

TEST(SignalStochasticOscillator, OverboughtCallbackFires) {
    SignalStochasticOscillator::Config cfg;
    cfg.k_period             = 5;
    cfg.d_period             = 3;
    cfg.min_samples          = 5;
    cfg.overbought_threshold = 80.0;
    std::atomic<int> fires{0};
    cfg.on_overbought = [&](double) { ++fires; };
    SignalStochasticOscillator s(cfg);
    for (int i = 1; i <= 10; ++i) s.record(static_cast<double>(i));
    EXPECT_GE(fires.load(), 1);
    EXPECT_GE(s.overbought_events(), 1u);
}

TEST(SignalStochasticOscillator, OversoldCallbackFires) {
    SignalStochasticOscillator::Config cfg;
    cfg.k_period           = 5;
    cfg.d_period           = 3;
    cfg.min_samples        = 5;
    cfg.oversold_threshold = 20.0;
    std::atomic<int> fires{0};
    cfg.on_oversold = [&](double) { ++fires; };
    SignalStochasticOscillator s(cfg);
    for (int i = 10; i >= 1; --i) s.record(static_cast<double>(i));
    EXPECT_GE(fires.load(), 1);
    EXPECT_GE(s.oversold_events(), 1u);
}

TEST(SignalStochasticOscillator, DPctValidAfterDPeriod) {
    SignalStochasticOscillator::Config cfg;
    cfg.k_period    = 5;
    cfg.d_period    = 3;
    cfg.min_samples = 5;
    SignalStochasticOscillator s(cfg);
    for (int i = 1; i <= 10; ++i) s.record(static_cast<double>(i));
    EXPECT_FALSE(std::isnan(s.d_pct()));
    EXPECT_GE(s.d_pct(), 0.0);
    EXPECT_LE(s.d_pct(), 100.0);
}

TEST(SignalStochasticOscillator, KDBullishCrossCallbackFires) {
    SignalStochasticOscillator::Config cfg;
    cfg.k_period    = 5;
    cfg.d_period    = 3;
    cfg.min_samples = 5;
    std::atomic<int> bull_fires{0};
    cfg.on_kd_bullish_cross = [&](double, double) { ++bull_fires; };
    SignalStochasticOscillator s(cfg);
    double vals[] = {5,4,3,2,1, 2,3,4,5,6, 5,4,3,2,1, 2,3,4,5,6};
    for (double v : vals) s.record(v);
    EXPECT_GE(bull_fires.load(), 1);
}

TEST(SignalStochasticOscillator, ResetClearsState) {
    SignalStochasticOscillator s;
    for (int i = 1; i <= 20; ++i) s.record(static_cast<double>(i));
    s.reset();
    EXPECT_EQ(s.total_records(), 0u);
    EXPECT_TRUE(std::isnan(s.k_pct()));
    EXPECT_FALSE(s.is_overbought());
    EXPECT_EQ(s.overbought_events(), 0u);
}

TEST(SignalStochasticOscillator, UpdateConfigResetsState) {
    SignalStochasticOscillator s;
    for (int i = 1; i <= 20; ++i) s.record(static_cast<double>(i));
    SignalStochasticOscillator::Config cfg2;
    cfg2.k_period = 10;
    s.update_config(cfg2);
    EXPECT_EQ(s.total_records(), 0u);
    EXPECT_TRUE(std::isnan(s.k_pct()));
}

TEST(SignalStochasticOscillator, ToStatsJsonContainsKeys) {
    SignalStochasticOscillator::Config cfg;
    cfg.k_period    = 5;
    cfg.min_samples = 5;
    SignalStochasticOscillator s(cfg);
    for (int i = 1; i <= 10; ++i) s.record(static_cast<double>(i));
    std::string j = s.to_stats_json();
    EXPECT_NE(j.find("k_pct"),             std::string::npos);
    EXPECT_NE(j.find("d_pct"),             std::string::npos);
    EXPECT_NE(j.find("is_overbought"),     std::string::npos);
    EXPECT_NE(j.find("is_oversold"),       std::string::npos);
    EXPECT_NE(j.find("total_records"),     std::string::npos);
    EXPECT_NE(j.find("overbought_events"), std::string::npos);
}

TEST(SignalStochasticOscillator, ConcurrentRecordSafe) {
    SignalStochasticOscillator s;
    std::vector<std::thread> threads;
    for (int t = 0; t < 4; ++t) {
        threads.emplace_back([&, t] {
            for (int i = 0; i < 250; ++i)
                s.record(static_cast<double>((i + t * 10) % 100));
        });
    }
    for (auto& th : threads) th.join();
    EXPECT_EQ(s.total_records(), 1000u);
}

} // namespace

#else

TEST(SignalStochasticOscillator, DisabledAtBuildTime) { SUCCEED(); }

#endif  // LLMQUANT_STOCHASTIC_OSC_ENABLED
