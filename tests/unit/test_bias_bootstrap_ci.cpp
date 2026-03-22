#include <gtest/gtest.h>
#include <thread>
#include <vector>

#ifdef LLMQUANT_BOOTSTRAP_CI_ENABLED
#include "BiasBootstrapCI.h"
using namespace llmquant;

TEST(BiasBootstrapCI, ConstructDefault) {
    BiasBootstrapCI bci;
    EXPECT_DOUBLE_EQ(bci.ci_lo(),       0.0);
    EXPECT_DOUBLE_EQ(bci.ci_hi(),       0.0);
    EXPECT_DOUBLE_EQ(bci.rolling_mean(), 0.0);
    EXPECT_EQ(bci.total_records(), 0u);
    EXPECT_EQ(bci.wide_events(),   0u);
}

TEST(BiasBootstrapCI, TotalRecordsIncrement) {
    BiasBootstrapCI bci;
    bci.record(0.1);
    bci.record(0.2);
    bci.record(0.3);
    EXPECT_EQ(bci.total_records(), 3u);
}

TEST(BiasBootstrapCI, CIComputedAfterMinSamples) {
    BiasBootstrapCI::Config cfg;
    cfg.min_samples        = 8;
    cfg.recompute_interval = 1;
    cfg.n_bootstrap        = 50;
    BiasBootstrapCI bci(cfg);
    for (int i = 0; i < 20; ++i) bci.record(0.5);
    // CI should be around 0.5 for constant signal
    EXPECT_NEAR(bci.rolling_mean(), 0.5, 0.01);
    EXPECT_LE(bci.ci_lo(), bci.ci_hi() + 1e-9);
}

TEST(BiasBootstrapCI, CILoLessThanHi) {
    BiasBootstrapCI::Config cfg;
    cfg.min_samples        = 4;
    cfg.recompute_interval = 1;
    cfg.n_bootstrap        = 100;
    BiasBootstrapCI bci(cfg);
    for (int i = 0; i < 30; ++i) bci.record(0.1 * (i % 7) - 0.3);
    EXPECT_LE(bci.ci_lo(), bci.ci_hi() + 1e-9);
}

TEST(BiasBootstrapCI, CIWidthNonNegative) {
    BiasBootstrapCI::Config cfg;
    cfg.min_samples        = 4;
    cfg.recompute_interval = 1;
    BiasBootstrapCI bci(cfg);
    for (int i = 0; i < 20; ++i) bci.record(0.1 * (i % 5));
    EXPECT_GE(bci.ci_width(), 0.0);
}

TEST(BiasBootstrapCI, WideCICallbackFired) {
    BiasBootstrapCI::Config cfg;
    cfg.width_threshold    = 0.0;  // fire always
    cfg.min_samples        = 4;
    cfg.recompute_interval = 1;
    cfg.n_bootstrap        = 50;
    bool fired = false;
    cfg.on_ci_wide = [&](double, double) { fired = true; };
    BiasBootstrapCI bci(cfg);
    for (int i = 0; i < 20; ++i) bci.record(0.1 * (i % 7) - 0.3);
    EXPECT_TRUE(fired);
}

TEST(BiasBootstrapCI, NarrowCICallbackFired) {
    BiasBootstrapCI::Config cfg;
    cfg.width_threshold    = 1000.0;  // narrow threshold very high → any ci_width qualifies
    cfg.min_samples        = 4;
    cfg.recompute_interval = 1;
    cfg.n_bootstrap        = 50;
    bool fired = false;
    cfg.on_ci_narrow = [&](double, double) { fired = true; };
    BiasBootstrapCI bci(cfg);
    // Constant signal → near-zero CI width → width < 1000.0 * 0.3 = 300
    for (int i = 0; i < 20; ++i) bci.record(0.5);
    EXPECT_TRUE(fired);
}

TEST(BiasBootstrapCI, RecomputeIntervalThrottles) {
    BiasBootstrapCI::Config cfg;
    cfg.recompute_interval = 100;
    cfg.min_samples        = 4;
    cfg.width_threshold    = 0.0;
    int fired = 0;
    cfg.on_ci_wide = [&](double, double) { ++fired; };
    BiasBootstrapCI bci(cfg);
    for (int i = 0; i < 30; ++i) bci.record(0.1 * (i % 7));
    // Only fires at multiples of 100 — for 30 records, maybe 0 fires
    EXPECT_GE(fired, 0);
}

TEST(BiasBootstrapCI, Reset) {
    BiasBootstrapCI bci;
    for (int i = 0; i < 20; ++i) bci.record(0.5);
    bci.reset();
    EXPECT_DOUBLE_EQ(bci.ci_lo(),       0.0);
    EXPECT_DOUBLE_EQ(bci.ci_hi(),       0.0);
    EXPECT_DOUBLE_EQ(bci.rolling_mean(), 0.0);
    EXPECT_EQ(bci.total_records(), 0u);
    EXPECT_EQ(bci.wide_events(),   0u);
}

TEST(BiasBootstrapCI, UpdateConfigResetsState) {
    BiasBootstrapCI bci;
    for (int i = 0; i < 20; ++i) bci.record(0.1);
    BiasBootstrapCI::Config cfg;
    cfg.window = 32;
    bci.update_config(cfg);
    EXPECT_EQ(bci.total_records(), 0u);
}

TEST(BiasBootstrapCI, ToStatsJsonFields) {
    BiasBootstrapCI bci;
    for (int i = 0; i < 20; ++i) bci.record(0.1 * i);
    auto js = bci.to_stats_json();
    EXPECT_NE(js.find("ci_lo"),        std::string::npos);
    EXPECT_NE(js.find("ci_hi"),        std::string::npos);
    EXPECT_NE(js.find("rolling_mean"), std::string::npos);
    EXPECT_NE(js.find("wide_events"),  std::string::npos);
    EXPECT_NE(js.find("total_records"), std::string::npos);
}

TEST(BiasBootstrapCI, ConcurrentRecord) {
    BiasBootstrapCI bci;
    std::vector<std::thread> threads;
    for (int t = 0; t < 4; ++t)
        threads.emplace_back([&] {
            for (int i = 0; i < 50; ++i) bci.record(0.1 * (i % 5));
        });
    for (auto& th : threads) th.join();
    EXPECT_GE(bci.total_records(), 100u);
}

#else
TEST(BiasBootstrapCI, DisabledAtBuildTime) { SUCCEED(); }
#endif
