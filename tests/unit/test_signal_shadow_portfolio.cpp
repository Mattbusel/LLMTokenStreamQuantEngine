#include <gtest/gtest.h>

#ifdef LLMQUANT_SHADOW_PORTFOLIO_ENABLED
#include "SignalShadowPortfolio.h"
#include <atomic>
#include <thread>
using namespace llmquant;

TEST(SignalShadowPortfolioTest, InitialState) {
    SignalShadowPortfolio p;
    EXPECT_NEAR(p.shadow_pnl(), 0.0, 1e-9);
    EXPECT_NEAR(p.live_pnl(), 0.0, 1e-9);
    EXPECT_NEAR(p.constraint_drag(), 0.0, 1e-9);
    EXPECT_EQ(p.price_updates(), 0u);
}

TEST(SignalShadowPortfolioTest, ShadowPnlAccumulates) {
    SignalShadowPortfolio::Config cfg;
    cfg.unit_size = 100.0;
    SignalShadowPortfolio p(cfg);
    p.record_signal(1.0, 1.0);  // position = 100
    p.record_price(100.0);       // seed
    p.record_price(101.0);       // price up 1 → pnl += 100
    EXPECT_NEAR(p.shadow_pnl(), 100.0, 1e-6);
}

TEST(SignalShadowPortfolioTest, LivePnlAccumulates) {
    SignalShadowPortfolio::Config cfg;
    cfg.unit_size = 100.0;
    SignalShadowPortfolio p(cfg);
    p.record_live_signal(0.5);  // live pos = 50
    p.record_price(100.0);
    p.record_price(102.0);      // up 2 → live pnl += 100
    EXPECT_NEAR(p.live_pnl(), 100.0, 1e-6);
}

TEST(SignalShadowPortfolioTest, ConstraintDragPositiveWhenShadowBetter) {
    SignalShadowPortfolio::Config cfg;
    cfg.unit_size = 100.0;
    cfg.drag_alert_threshold = 1e9; // suppress alerts
    SignalShadowPortfolio p(cfg);
    p.record_signal(1.0, 1.0);     // shadow pos = 100
    p.record_live_signal(0.0);     // live pos = 0 (blocked by risk gate)
    p.record_price(100.0);
    p.record_price(101.0);
    EXPECT_GT(p.constraint_drag(), 0.0);  // shadow made money, live did not
    EXPECT_NEAR(p.constraint_drag(), 100.0, 1e-6);
}

TEST(SignalShadowPortfolioTest, DragAlertFires) {
    SignalShadowPortfolio::Config cfg;
    cfg.unit_size = 100.0;
    cfg.drag_alert_threshold = 50.0;
    std::atomic<int> fires{0};
    cfg.on_drag_alert = [&](double, double, double) { ++fires; };
    SignalShadowPortfolio p(cfg);
    p.record_signal(1.0, 1.0);
    p.record_live_signal(0.0);
    p.record_price(100.0);
    for (int i = 1; i <= 10; ++i) p.record_price(100.0 + i);
    EXPECT_GE(fires.load(), 1);
}

TEST(SignalShadowPortfolioTest, PositionClampedToMax) {
    SignalShadowPortfolio::Config cfg;
    cfg.unit_size    = 1000.0;
    cfg.max_position = 500.0;
    SignalShadowPortfolio p(cfg);
    p.record_signal(1.0, 1.0);  // would be 1000, clamped to 500
    EXPECT_NEAR(p.shadow_position(), 500.0, 1e-9);
}

TEST(SignalShadowPortfolioTest, NegativePositionPnl) {
    SignalShadowPortfolio::Config cfg;
    cfg.unit_size = 100.0;
    SignalShadowPortfolio p(cfg);
    p.record_signal(-1.0, 1.0);  // short 100
    p.record_price(100.0);
    p.record_price(99.0);         // price down 1 → short gains 100
    EXPECT_NEAR(p.shadow_pnl(), 100.0, 1e-6);
}

TEST(SignalShadowPortfolioTest, ResetClearsAll) {
    SignalShadowPortfolio p;
    p.record_signal(1.0, 1.0);
    p.record_price(100.0);
    p.record_price(105.0);
    p.reset();
    EXPECT_NEAR(p.shadow_pnl(), 0.0, 1e-9);
    EXPECT_NEAR(p.live_pnl(), 0.0, 1e-9);
    EXPECT_EQ(p.price_updates(), 0u);
}

TEST(SignalShadowPortfolioTest, StatsJsonNonEmpty) {
    SignalShadowPortfolio p;
    p.record_price(100.0);
    std::string j = p.to_stats_json();
    EXPECT_NE(j.find("shadow_pnl"), std::string::npos);
    EXPECT_NE(j.find("constraint_drag"), std::string::npos);
    EXPECT_NE(j.find("drag_alert_count"), std::string::npos);
}

TEST(SignalShadowPortfolioTest, ConcurrentSafe) {
    SignalShadowPortfolio p;
    p.record_price(100.0);
    std::atomic<bool> go{false};
    auto w_sig   = [&] { while(!go.load()){} for(int i=0;i<200;++i) p.record_signal(0.5, 1.0); };
    auto w_live  = [&] { while(!go.load()){} for(int i=0;i<200;++i) p.record_live_signal(0.3); };
    auto w_price = [&] {
        while(!go.load()){}
        double px = 100.0;
        for(int i=0;i<200;++i) { px += 0.01; p.record_price(px); }
    };
    std::thread t1(w_sig), t2(w_live), t3(w_price);
    go.store(true);
    t1.join(); t2.join(); t3.join();
    // No crash = success.
}

#else
TEST(SignalShadowPortfolioTest, DisabledAtBuildTime) { SUCCEED(); }
#endif
