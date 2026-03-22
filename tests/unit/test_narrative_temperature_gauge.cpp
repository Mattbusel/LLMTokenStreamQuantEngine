#include <gtest/gtest.h>

#ifdef LLMQUANT_NARRATIVE_TEMPERATURE_ENABLED

#include "NarrativeTemperatureGauge.h"

#include <atomic>
#include <cmath>
#include <thread>
#include <vector>

using namespace llmquant;

namespace {

TEST(NarrativeTemperatureGauge, DefaultState) {
    NarrativeTemperatureGauge g;
    EXPECT_NEAR(g.temperature(), 0.0, 1e-9);
    EXPECT_FALSE(g.is_hot());
    EXPECT_EQ(g.heat_events(), 0u);
    EXPECT_EQ(g.total_records(), 0u);
}

TEST(NarrativeTemperatureGauge, TotalRecordsIncrements) {
    NarrativeTemperatureGauge g;
    g.record(0.1);
    g.record(-0.2);
    g.record(0.3);
    EXPECT_EQ(g.total_records(), 3u);
}

TEST(NarrativeTemperatureGauge, TemperatureBounded) {
    NarrativeTemperatureGauge::Config cfg;
    cfg.window     = 16;
    cfg.normalizer = 0.1;
    NarrativeTemperatureGauge g(cfg);
    for (int i = 0; i < 50; ++i)
        g.record((i % 2 == 0) ? 2.0 : -1.0);
    double t = g.temperature();
    EXPECT_GE(t, 0.0);
    EXPECT_LE(t, 1.0 + 1e-9);
}

TEST(NarrativeTemperatureGauge, StrongDirectionalHighVolatilityIsHot) {
    NarrativeTemperatureGauge::Config cfg;
    cfg.window        = 16;
    cfg.ema_alpha     = 0.3;
    cfg.normalizer    = 0.5;
    cfg.hot_threshold = 0.5;
    NarrativeTemperatureGauge g(cfg);
    // Strong positive bias (ema→+5) with large swings (σ~1).
    for (int i = 0; i < 32; ++i)
        g.record((i % 3 == 0) ? 5.0 : 4.0);
    EXPECT_GT(g.bias_ema(), 0.0);
    EXPECT_GT(g.bias_sigma(), 0.0);
    // Temperature = |ema|×sigma/normalizer — should be high.
    EXPECT_GT(g.temperature(), 0.0);
}

TEST(NarrativeTemperatureGauge, WeakDirectionLowVolatilityIsCool) {
    NarrativeTemperatureGauge::Config cfg;
    cfg.window        = 16;
    cfg.ema_alpha     = 0.1;
    cfg.normalizer    = 0.5;
    cfg.hot_threshold = 0.7;
    NarrativeTemperatureGauge g(cfg);
    // Constant near-zero bias → near-zero ema, near-zero sigma.
    for (int i = 0; i < 32; ++i) g.record(0.001);
    EXPECT_LT(g.temperature(), 0.5);
    EXPECT_FALSE(g.is_hot());
}

TEST(NarrativeTemperatureGauge, HotCallbackFires) {
    NarrativeTemperatureGauge::Config cfg;
    cfg.window        = 8;
    cfg.ema_alpha     = 0.5;
    cfg.normalizer    = 0.01;  // tiny normalizer → temperature saturates quickly
    cfg.hot_threshold = 0.5;
    std::atomic<int> fires{0};
    cfg.on_hot = [&](double) { ++fires; };
    NarrativeTemperatureGauge g(cfg);
    for (int i = 0; i < 20; ++i)
        g.record((i % 2 == 0) ? 1.0 : 0.8);
    EXPECT_GE(fires.load(), 1);
    EXPECT_GT(g.heat_events(), 0u);
}

TEST(NarrativeTemperatureGauge, CoolCallbackFires) {
    NarrativeTemperatureGauge::Config cfg;
    cfg.window        = 8;
    cfg.ema_alpha     = 0.5;
    cfg.normalizer    = 0.01;
    cfg.hot_threshold = 0.5;
    cfg.cool_hysteresis = 0.9;
    std::atomic<int> cools{0};
    cfg.on_cool = [&](double) { ++cools; };
    NarrativeTemperatureGauge g(cfg);
    // Heat up.
    for (int i = 0; i < 16; ++i)
        g.record((i % 2 == 0) ? 1.0 : 0.8);
    // Cool down with near-constant low bias.
    for (int i = 0; i < 50; ++i) g.record(0.0);
    EXPECT_GE(cools.load(), 1);
}

TEST(NarrativeTemperatureGauge, ResetClearsState) {
    NarrativeTemperatureGauge g;
    for (int i = 0; i < 20; ++i) g.record(i % 2 == 0 ? 2.0 : -2.0);
    g.reset();
    EXPECT_EQ(g.total_records(), 0u);
    EXPECT_EQ(g.heat_events(), 0u);
    EXPECT_FALSE(g.is_hot());
    EXPECT_NEAR(g.temperature(), 0.0, 1e-9);
}

TEST(NarrativeTemperatureGauge, UpdateConfigResetsState) {
    NarrativeTemperatureGauge g;
    for (int i = 0; i < 20; ++i) g.record(i % 2 == 0 ? 2.0 : -2.0);
    NarrativeTemperatureGauge::Config cfg2;
    cfg2.window = 8;
    g.update_config(cfg2);
    EXPECT_EQ(g.total_records(), 0u);
    EXPECT_FALSE(g.is_hot());
}

TEST(NarrativeTemperatureGauge, StatsJsonContainsFields) {
    NarrativeTemperatureGauge g;
    for (int i = 0; i < 20; ++i) g.record(i % 2 == 0 ? 1.0 : -1.0);
    std::string j = g.to_stats_json();
    EXPECT_NE(j.find("temperature"),   std::string::npos);
    EXPECT_NE(j.find("is_hot"),        std::string::npos);
    EXPECT_NE(j.find("bias_ema"),      std::string::npos);
    EXPECT_NE(j.find("bias_sigma"),    std::string::npos);
    EXPECT_NE(j.find("heat_events"),   std::string::npos);
    EXPECT_NE(j.find("total"),         std::string::npos);
}

TEST(NarrativeTemperatureGauge, ConcurrentRecordSafe) {
    NarrativeTemperatureGauge::Config cfg;
    cfg.window = 32;
    NarrativeTemperatureGauge g(cfg);
    std::atomic<bool> go{false};
    auto worker = [&](int id) {
        while (!go.load()) {}
        for (int i = 0; i < 250; ++i)
            g.record((id % 2 == 0) ? static_cast<double>(i % 5) * 0.2
                                   : -static_cast<double>(i % 5) * 0.2);
    };
    std::vector<std::thread> threads;
    for (int t = 0; t < 4; ++t) threads.emplace_back(worker, t);
    go.store(true);
    for (auto& t : threads) t.join();
    EXPECT_EQ(g.total_records(), 1000u);
    double temp = g.temperature();
    EXPECT_GE(temp, 0.0);
    EXPECT_LE(temp, 1.0 + 1e-9);
}

}  // namespace

#else

TEST(NarrativeTemperatureGauge, DisabledAtBuildTime) { SUCCEED(); }

#endif  // LLMQUANT_NARRATIVE_TEMPERATURE_ENABLED
