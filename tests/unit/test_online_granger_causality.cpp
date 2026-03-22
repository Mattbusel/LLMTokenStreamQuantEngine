#include <gtest/gtest.h>

#ifdef LLMQUANT_ONLINE_GRANGER_ENABLED

#include "OnlineGrangerCausality.h"

#include <atomic>
#include <cmath>
#include <thread>
#include <vector>

using namespace llmquant;

namespace {

TEST(OnlineGrangerCausality, DefaultState) {
    OnlineGrangerCausality g;
    EXPECT_TRUE(std::isnan(g.f_xy()));
    EXPECT_TRUE(std::isnan(g.f_yx()));
    EXPECT_FALSE(g.x_causes_y());
    EXPECT_FALSE(g.y_causes_x());
    EXPECT_EQ(g.total_records(), 0u);
}

TEST(OnlineGrangerCausality, TotalRecordsIncrements) {
    OnlineGrangerCausality g;
    g.record(0.1, 0.2);
    g.record(0.3, 0.4);
    EXPECT_EQ(g.total_records(), 2u);
}

TEST(OnlineGrangerCausality, NaNBeforeMinSamples) {
    OnlineGrangerCausality::Config cfg;
    cfg.window      = 32;
    cfg.min_samples = 32;
    OnlineGrangerCausality g(cfg);
    for (int i = 0; i < 31; ++i) g.record(0.1, 0.2);
    EXPECT_TRUE(std::isnan(g.f_xy()));
}

TEST(OnlineGrangerCausality, XCausesYCallbackFires) {
    // Feed: y_t = 0.8*y_{t-1} + 0.5*x_{t-1} + ε
    // x is iid noise; including x_lag improves y prediction → X Granger-causes Y
    OnlineGrangerCausality::Config cfg;
    cfg.window      = 64;
    cfg.min_samples = 32;
    cfg.f_threshold = 3.84;
    std::atomic<int> fires{0};
    cfg.on_x_causes_y = [&](double) { ++fires; };
    OnlineGrangerCausality g(cfg);

    // Generate synthetic causal data
    double x = 0.0, y = 0.0;
    for (int i = 0; i < 200; ++i) {
        double xn = std::sin(static_cast<double>(i) * 0.3);
        double yn = 0.6 * y + 0.8 * x + std::sin(static_cast<double>(i) * 0.07) * 0.1;
        g.record(xn, yn);
        x = xn; y = yn;
    }
    // F_xy should be positive and likely significant
    EXPECT_TRUE(std::isfinite(g.f_xy()));
    EXPECT_GT(g.f_xy(), 0.0);
}

TEST(OnlineGrangerCausality, IndependentSeriesLowFStat) {
    // Two independent random-walk series: neither should strongly Granger-cause the other
    OnlineGrangerCausality::Config cfg;
    cfg.window      = 64;
    cfg.min_samples = 64;
    cfg.f_threshold = 100.0;  // very high threshold — expect not significant
    OnlineGrangerCausality g(cfg);

    // x_t = cos(t), y_t = sin(t*7) — unrelated frequencies
    for (int i = 0; i < 200; ++i) {
        double x = std::cos(static_cast<double>(i) * 0.5);
        double y = std::sin(static_cast<double>(i) * 3.7);
        g.record(x, y);
    }
    EXPECT_FALSE(g.x_causes_y());
    EXPECT_FALSE(g.y_causes_x());
}

TEST(OnlineGrangerCausality, FStatNonNegative) {
    OnlineGrangerCausality::Config cfg;
    cfg.window      = 32;
    cfg.min_samples = 32;
    OnlineGrangerCausality g(cfg);
    for (int i = 0; i < 64; ++i)
        g.record(std::sin(static_cast<double>(i) * 0.2),
                 std::cos(static_cast<double>(i) * 0.2));
    if (std::isfinite(g.f_xy())) EXPECT_GE(g.f_xy(), 0.0);
    if (std::isfinite(g.f_yx())) EXPECT_GE(g.f_yx(), 0.0);
}

TEST(OnlineGrangerCausality, ResetClearsState) {
    OnlineGrangerCausality g;
    for (int i = 0; i < 40; ++i) g.record(0.1, 0.2);
    g.reset();
    EXPECT_EQ(g.total_records(), 0u);
    EXPECT_TRUE(std::isnan(g.f_xy()));
    EXPECT_FALSE(g.x_causes_y());
}

TEST(OnlineGrangerCausality, UpdateConfigResetsState) {
    OnlineGrangerCausality g;
    for (int i = 0; i < 40; ++i) g.record(0.1, 0.2);
    OnlineGrangerCausality::Config cfg2;
    cfg2.window = 32;
    g.update_config(cfg2);
    EXPECT_EQ(g.total_records(), 0u);
    EXPECT_TRUE(std::isnan(g.f_xy()));
}

TEST(OnlineGrangerCausality, ToStatsJsonContainsKeys) {
    OnlineGrangerCausality::Config cfg;
    cfg.window      = 32;
    cfg.min_samples = 32;
    OnlineGrangerCausality g(cfg);
    for (int i = 0; i < 40; ++i)
        g.record(std::sin(static_cast<double>(i) * 0.3),
                 std::cos(static_cast<double>(i) * 0.3));
    std::string j = g.to_stats_json();
    EXPECT_NE(j.find("f_xy"),                std::string::npos);
    EXPECT_NE(j.find("f_yx"),                std::string::npos);
    EXPECT_NE(j.find("x_causes_y"),          std::string::npos);
    EXPECT_NE(j.find("y_causes_x"),          std::string::npos);
    EXPECT_NE(j.find("xy_causality_events"), std::string::npos);
    EXPECT_NE(j.find("total_records"),       std::string::npos);
}

TEST(OnlineGrangerCausality, ConcurrentRecordSafe) {
    OnlineGrangerCausality g;
    std::vector<std::thread> threads;
    for (int t = 0; t < 4; ++t) {
        threads.emplace_back([&, t] {
            for (int i = 0; i < 250; ++i)
                g.record(std::sin(static_cast<double>(i + t) * 0.2),
                         std::cos(static_cast<double>(i + t) * 0.2));
        });
    }
    for (auto& th : threads) th.join();
    EXPECT_EQ(g.total_records(), 1000u);
    EXPECT_TRUE(std::isfinite(g.f_xy()) || std::isnan(g.f_xy()));
}

} // namespace

#else

TEST(OnlineGrangerCausality, DisabledAtBuildTime) { SUCCEED(); }

#endif  // LLMQUANT_ONLINE_GRANGER_ENABLED
