#include <gtest/gtest.h>

#ifdef LLMQUANT_ADAPTIVE_SIZER_ENABLED
#include "AdaptivePositionSizer.h"
#include <atomic>
#include <thread>
using namespace llmquant;

TEST(AdaptivePositionSizerTest, InitialStateFullSize) {
    AdaptivePositionSizer s;
    EXPECT_NEAR(s.multiplier(), 1.0, 1e-6);
    EXPECT_EQ(s.change_events(), 0u);
}

TEST(AdaptivePositionSizerTest, LowCoherenceReducesMultiplier) {
    AdaptivePositionSizer::Config cfg;
    cfg.coherence_weight   = 0.5;
    cfg.confidence_weight  = 0.0;
    cfg.vol_weight         = 0.0;
    cfg.regime_weight      = 0.0;
    AdaptivePositionSizer s(cfg);
    s.set_coherence(0.0);  // worst coherence
    EXPECT_LT(s.multiplier(), 1.0);
}

TEST(AdaptivePositionSizerTest, HighVolReducesMultiplier) {
    AdaptivePositionSizer::Config cfg;
    cfg.coherence_weight  = 0.0;
    cfg.confidence_weight = 0.0;
    cfg.vol_weight        = 0.5;
    cfg.regime_weight     = 0.0;
    cfg.rv_max            = 1.0;
    AdaptivePositionSizer s(cfg);
    s.set_realized_vol(1.0);  // maximum vol
    EXPECT_LT(s.multiplier(), 1.0);
}

TEST(AdaptivePositionSizerTest, CrashRegimeReducesMultiplier) {
    AdaptivePositionSizer::Config cfg;
    cfg.coherence_weight          = 0.0;
    cfg.confidence_weight         = 0.0;
    cfg.vol_weight                = 0.0;
    cfg.regime_weight             = 0.5;
    cfg.regime_factor_crash       = 0.1;
    AdaptivePositionSizer s(cfg);
    s.set_regime(AdaptivePositionSizer::Regime::Crash);
    EXPECT_LT(s.multiplier(), 1.0);
}

TEST(AdaptivePositionSizerTest, MinMultiplierFloor) {
    AdaptivePositionSizer::Config cfg;
    cfg.coherence_weight  = 1.0;
    cfg.confidence_weight = 0.0;
    cfg.vol_weight        = 0.0;
    cfg.regime_weight     = 0.0;
    cfg.min_multiplier    = 0.1;
    AdaptivePositionSizer s(cfg);
    s.set_coherence(0.0);
    EXPECT_GE(s.multiplier(), 0.1);
}

TEST(AdaptivePositionSizerTest, ChangeCallbackFires) {
    AdaptivePositionSizer::Config cfg;
    cfg.coherence_weight    = 0.5;
    cfg.confidence_weight   = 0.0;
    cfg.vol_weight          = 0.0;
    cfg.regime_weight       = 0.0;
    cfg.change_threshold    = 0.01;  // very sensitive
    std::atomic<int> fires{0};
    cfg.on_size_change = [&](double, double) { ++fires; };
    AdaptivePositionSizer s(cfg);
    s.set_coherence(1.0);  // initial set
    s.set_coherence(0.0);  // large change → fire
    EXPECT_GE(fires.load(), 1);
    EXPECT_GE(s.change_events(), 1u);
}

TEST(AdaptivePositionSizerTest, SetConfidence) {
    AdaptivePositionSizer::Config cfg;
    cfg.confidence_weight = 0.5;
    cfg.coherence_weight  = 0.0;
    cfg.vol_weight        = 0.0;
    cfg.regime_weight     = 0.0;
    AdaptivePositionSizer s(cfg);
    s.set_confidence(0.0);
    EXPECT_LT(s.multiplier(), 1.0);
}

TEST(AdaptivePositionSizerTest, TrendingRegimeFullSize) {
    AdaptivePositionSizer::Config cfg;
    cfg.regime_weight             = 0.5;
    cfg.coherence_weight          = 0.0;
    cfg.confidence_weight         = 0.0;
    cfg.vol_weight                = 0.0;
    cfg.regime_factor_trending    = 1.0;
    AdaptivePositionSizer s(cfg);
    s.set_regime(AdaptivePositionSizer::Regime::Trending);
    EXPECT_NEAR(s.multiplier(), 1.0, 1e-6);
}

TEST(AdaptivePositionSizerTest, ResetRestoresDefaults) {
    AdaptivePositionSizer s;
    s.set_coherence(0.0);
    s.set_realized_vol(0.8);
    s.reset();
    EXPECT_NEAR(s.multiplier(), 1.0, 1e-6);
}

TEST(AdaptivePositionSizerTest, StatsJsonHasFields) {
    AdaptivePositionSizer s;
    std::string j = s.to_stats_json();
    EXPECT_NE(j.find("multiplier"),    std::string::npos);
    EXPECT_NE(j.find("change_events"), std::string::npos);
    EXPECT_NE(j.find("coherence_in"),  std::string::npos);
    EXPECT_NE(j.find("rv_in"),         std::string::npos);
}

TEST(AdaptivePositionSizerTest, ConcurrentSafe) {
    AdaptivePositionSizer s;
    std::atomic<bool> go{false};
    auto worker0 = [&] {
        while (!go.load()) {}
        for (int i = 0; i < 200; ++i)
            s.set_coherence(static_cast<double>(i % 10) / 10.0);
    };
    auto worker1 = [&] {
        while (!go.load()) {}
        for (int i = 0; i < 200; ++i)
            s.set_realized_vol(static_cast<double>(i % 5) * 0.1);
    };
    std::thread t1(worker0), t2(worker1);
    go.store(true);
    t1.join(); t2.join();
    double m = s.multiplier();
    EXPECT_GE(m, s.to_stats_json().find("multiplier") != std::string::npos ? 0.0 : 0.0);
    EXPECT_LE(m, 1.0 + 1e-6);
}

#else
TEST(AdaptivePositionSizerTest, DisabledAtBuildTime) { SUCCEED(); }
#endif
