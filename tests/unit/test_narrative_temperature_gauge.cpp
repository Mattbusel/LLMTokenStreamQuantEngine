#include <gtest/gtest.h>

#ifdef LLMQUANT_NARRATIVE_TEMPERATURE_ENABLED

#include "NarrativeTemperatureGauge.h"

#include <atomic>
#include <cmath>
#include <thread>
#include <vector>

using namespace llmquant;

namespace {

TEST(NarrativeTemperatureGauge, DefaultConstructs) {
    NarrativeTemperatureGauge g;
    EXPECT_DOUBLE_EQ(g.temperature(), 0.0);
    EXPECT_FALSE(g.is_hot());
    EXPECT_EQ(g.heat_events(), 0u);
    EXPECT_EQ(g.total_records(), 0u);
}

TEST(NarrativeTemperatureGauge, SingleRecordDoesNotTrip) {
    NarrativeTemperatureGauge g;
    g.record(0.5);
    EXPECT_EQ(g.total_records(), 1u);
    // single point: sigma = 0, temperature = 0
    EXPECT_DOUBLE_EQ(g.temperature(), 0.0);
}

TEST(NarrativeTemperatureGauge, HighVolatilityRaisesTemperature) {
    NarrativeTemperatureGauge::Config cfg;
    cfg.hot_threshold = 0.5;
    cfg.normalizer    = 0.1;
    NarrativeTemperatureGauge g(cfg);
    // Feed alternating large swings to create high sigma
    for (int i = 0; i < 40; ++i)
        g.record(i % 2 == 0 ? 1.0 : -1.0);
    EXPECT_GT(g.temperature(), 0.0);
    EXPECT_GT(g.bias_sigma(), 0.0);
}

TEST(NarrativeTemperatureGauge, HotCallbackFires) {
    NarrativeTemperatureGauge::Config cfg;
    cfg.hot_threshold = 0.1;
    cfg.normalizer    = 0.01;
    std::atomic<int> fires{0};
    cfg.on_hot = [&](double) { ++fires; };
    NarrativeTemperatureGauge g(cfg);
    for (int i = 0; i < 30; ++i)
        g.record(i % 2 == 0 ? 1.0 : -1.0);
    EXPECT_GT(fires.load(), 0);
    EXPECT_GT(g.heat_events(), 0u);
}

TEST(NarrativeTemperatureGauge, StableSignalLowTemperature) {
    NarrativeTemperatureGauge g;
    for (int i = 0; i < 50; ++i)
        g.record(0.5);  // constant: sigma = 0
    EXPECT_DOUBLE_EQ(g.temperature(), 0.0);
    EXPECT_FALSE(g.is_hot());
}

TEST(NarrativeTemperatureGauge, ResetClearsState) {
    NarrativeTemperatureGauge::Config cfg;
    cfg.normalizer = 0.01;
    NarrativeTemperatureGauge g(cfg);
    for (int i = 0; i < 20; ++i) g.record(i % 2 == 0 ? 1.0 : -1.0);
    g.reset();
    EXPECT_DOUBLE_EQ(g.temperature(), 0.0);
    EXPECT_EQ(g.total_records(), 0u);
    EXPECT_EQ(g.heat_events(), 0u);
    EXPECT_FALSE(g.is_hot());
}

TEST(NarrativeTemperatureGauge, ToStatsJsonContainsKeys) {
    NarrativeTemperatureGauge g;
    g.record(0.3);
    std::string j = g.to_stats_json();
    EXPECT_NE(j.find("temperature"),  std::string::npos);
    EXPECT_NE(j.find("is_hot"),       std::string::npos);
    EXPECT_NE(j.find("bias_sigma"),   std::string::npos);
    EXPECT_NE(j.find("heat_events"),  std::string::npos);
}

TEST(NarrativeTemperatureGauge, UpdateConfigResetsState) {
    NarrativeTemperatureGauge g;
    for (int i = 0; i < 10; ++i) g.record(static_cast<double>(i));
    NarrativeTemperatureGauge::Config cfg;
    cfg.window = 16;
    g.update_config(cfg);
    EXPECT_EQ(g.total_records(), 0u);
}

TEST(NarrativeTemperatureGauge, ConcurrentRecordSafe) {
    NarrativeTemperatureGauge g;
    std::vector<std::thread> threads;
    for (int t = 0; t < 4; ++t) {
        threads.emplace_back([&, t] {
            for (int i = 0; i < 100; ++i)
                g.record(i % 2 == 0 ? 0.5 : -0.5);
        });
    }
    for (auto& th : threads) th.join();
    EXPECT_EQ(g.total_records(), 400u);
    EXPECT_GE(g.temperature(), 0.0);
    EXPECT_LE(g.temperature(), 1.0);
}

} // namespace

#else

TEST(NarrativeTemperatureGauge, DisabledAtBuildTime) {
    SUCCEED();
}

#endif // LLMQUANT_NARRATIVE_TEMPERATURE_ENABLED
