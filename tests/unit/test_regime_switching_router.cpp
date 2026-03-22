#include <gtest/gtest.h>

#ifdef LLMQUANT_REGIME_ROUTER_ENABLED
#include "RegimeSwitchingSignalRouter.h"
#include <atomic>
#include <thread>
using namespace llmquant;

TEST(RegimeSwitchingRouterTest, InitialRegimeIsRanging) {
    RegimeSwitchingSignalRouter r;
    EXPECT_EQ(r.current_regime(), RegimeSwitchingSignalRouter::Regime::Ranging);
    EXPECT_EQ(r.regime_name(), "Ranging");
}

TEST(RegimeSwitchingRouterTest, HighVolEntersCrash) {
    RegimeSwitchingSignalRouter::Config cfg;
    cfg.crash_vol_threshold = 0.02;
    cfg.vol_alpha = 0.9;
    RegimeSwitchingSignalRouter r(cfg);
    // Feed large returns to spike volatility.
    for (int i = 0; i < 20; ++i)
        r.record_return((i % 2 == 0) ? 0.1 : -0.1);
    EXPECT_EQ(r.current_regime(), RegimeSwitchingSignalRouter::Regime::Crash);
}

TEST(RegimeSwitchingRouterTest, HighMomentumEntersTrending) {
    RegimeSwitchingSignalRouter::Config cfg;
    cfg.trend_threshold    = 0.02;
    cfg.crash_vol_threshold = 10.0;  // disable crash
    cfg.momentum_alpha     = 0.9;
    RegimeSwitchingSignalRouter r(cfg);
    // Sustained positive returns → positive momentum.
    for (int i = 0; i < 20; ++i) r.record_return(0.05);
    EXPECT_EQ(r.current_regime(), RegimeSwitchingSignalRouter::Regime::Trending);
}

TEST(RegimeSwitchingRouterTest, RegimeChangeCallbackFires) {
    RegimeSwitchingSignalRouter::Config cfg;
    cfg.crash_vol_threshold = 0.02;
    cfg.vol_alpha = 0.9;
    std::atomic<int> fires{0};
    cfg.on_regime_change = [&](auto, auto, double) { ++fires; };
    RegimeSwitchingSignalRouter r(cfg);
    for (int i = 0; i < 20; ++i)
        r.record_return((i % 2 == 0) ? 0.1 : -0.1);
    EXPECT_GE(fires.load(), 1);
    EXPECT_GE(r.regime_change_count(), 1u);
}

TEST(RegimeSwitchingRouterTest, RouteBiasAppliesMultiplier) {
    RegimeSwitchingSignalRouter::Config cfg;
    cfg.crash_vol_threshold = 0.02;
    cfg.vol_alpha = 0.9;
    cfg.crash_cfg.bias_multiplier = 0.0;  // silence signals in crash
    RegimeSwitchingSignalRouter r(cfg);
    for (int i = 0; i < 20; ++i)
        r.record_return((i % 2 == 0) ? 0.1 : -0.1);
    EXPECT_NEAR(r.route_bias(1.0), 0.0, 1e-9);
}

TEST(RegimeSwitchingRouterTest, SmoothedVolIncreases) {
    RegimeSwitchingSignalRouter r;
    double v0 = r.smoothed_volatility();
    r.record_return(0.0);
    r.record_return(0.5);
    EXPECT_GT(r.smoothed_volatility(), v0);
}

TEST(RegimeSwitchingRouterTest, ResetClearsState) {
    RegimeSwitchingSignalRouter::Config cfg;
    cfg.crash_vol_threshold = 0.02;
    cfg.vol_alpha = 0.9;
    RegimeSwitchingSignalRouter r(cfg);
    for (int i = 0; i < 20; ++i)
        r.record_return((i % 2 == 0) ? 0.1 : -0.1);
    r.reset();
    EXPECT_EQ(r.current_regime(), RegimeSwitchingSignalRouter::Regime::Ranging);
    EXPECT_NEAR(r.smoothed_volatility(), 0.0, 1e-9);
}

TEST(RegimeSwitchingRouterTest, StatsJsonNonEmpty) {
    RegimeSwitchingSignalRouter r;
    r.record_return(0.01);
    std::string j = r.to_stats_json();
    EXPECT_NE(j.find("regime"),          std::string::npos);
    EXPECT_NE(j.find("vol_ema"),         std::string::npos);
    EXPECT_NE(j.find("bias_multiplier"), std::string::npos);
}

TEST(RegimeSwitchingRouterTest, ConcurrentSafe) {
    RegimeSwitchingSignalRouter r;
    std::atomic<bool> go{false};
    auto worker = [&] {
        while (!go.load()) {}
        for (int i = 0; i < 200; ++i)
            r.record_return(i % 5 == 0 ? 0.1 : 0.001);
    };
    std::thread t1(worker), t2(worker);
    go.store(true);
    t1.join(); t2.join();
    // No crash = success.
}

#else
TEST(RegimeSwitchingRouterTest, DisabledAtBuildTime) { SUCCEED(); }
#endif
