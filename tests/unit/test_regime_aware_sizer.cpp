#include <gtest/gtest.h>

#ifdef LLMQUANT_REGIME_SIZER_ENABLED

#include "RegimeAwareSizer.h"

#include <atomic>
#include <cmath>
#include <thread>
#include <vector>

using namespace llmquant;

namespace {

// ============================================================
// Default state
// ============================================================

TEST(RegimeAwareSizer, DefaultStateNeutral) {
    RegimeAwareSizer sizer;
    // No vol fed → vol_factor stays 1.0; H = 0.5 → regime_factor = regime_neutral
    EXPECT_DOUBLE_EQ(sizer.regime_factor(), 1.0);
    EXPECT_DOUBLE_EQ(sizer.vol_factor(), 1.0);
    EXPECT_DOUBLE_EQ(sizer.size_multiplier(), 1.0);
    EXPECT_DOUBLE_EQ(sizer.current_hurst(), 0.5);
}

// ============================================================
// Trending regime (H > 0.5) increases multiplier
// ============================================================

TEST(RegimeAwareSizer, TrendingHurstIncreasesMultiplier) {
    RegimeAwareSizer::Config cfg;
    cfg.hurst_max      = 0.8;
    cfg.regime_neutral = 1.0;
    cfg.regime_max     = 1.5;
    cfg.regime_min     = 0.4;
    RegimeAwareSizer sizer(cfg);

    sizer.update_hurst(0.8);
    EXPECT_GT(sizer.size_multiplier(), 1.0);
    EXPECT_NEAR(sizer.regime_factor(), 1.5, 1e-9);
}

// ============================================================
// Mean-reverting regime (H < 0.5) decreases multiplier
// ============================================================

TEST(RegimeAwareSizer, MeanRevertingHurstDecreasesMultiplier) {
    RegimeAwareSizer::Config cfg;
    cfg.regime_neutral = 1.0;
    cfg.regime_min     = 0.4;
    RegimeAwareSizer sizer(cfg);

    sizer.update_hurst(0.0);
    EXPECT_LT(sizer.size_multiplier(), 1.0);
    EXPECT_NEAR(sizer.regime_factor(), cfg.regime_min, 1e-9);
}

// ============================================================
// High volatility reduces multiplier
// ============================================================

TEST(RegimeAwareSizer, HighVolReducesMultiplier) {
    RegimeAwareSizer::Config cfg;
    cfg.target_vol = 0.20;
    cfg.vol_floor  = 0.1;
    cfg.vol_cap    = 3.0;
    RegimeAwareSizer sizer(cfg);

    sizer.update_hurst(0.5);   // neutral regime
    sizer.update_vol(0.40);    // 2× target → vol_factor = 0.5

    EXPECT_NEAR(sizer.vol_factor(), 0.5, 1e-6);
    EXPECT_NEAR(sizer.size_multiplier(), 0.5, 1e-6);
}

// ============================================================
// Low volatility increases multiplier (capped at vol_cap)
// ============================================================

TEST(RegimeAwareSizer, LowVolIncreasesMultiplierToCap) {
    RegimeAwareSizer::Config cfg;
    cfg.target_vol = 0.20;
    cfg.vol_cap    = 2.0;
    RegimeAwareSizer sizer(cfg);

    sizer.update_hurst(0.5);
    sizer.update_vol(0.05);  // 4× below target → capped at 2.0
    EXPECT_NEAR(sizer.vol_factor(), 2.0, 1e-6);
}

// ============================================================
// Combined: trending + low vol → large multiplier
// ============================================================

TEST(RegimeAwareSizer, TrendingLowVolLargeMultiplier) {
    RegimeAwareSizer::Config cfg;
    cfg.hurst_max      = 0.8;
    cfg.regime_neutral = 1.0;
    cfg.regime_max     = 1.5;
    cfg.target_vol     = 0.20;
    cfg.vol_cap        = 3.0;
    RegimeAwareSizer sizer(cfg);

    sizer.update_hurst(0.8);
    sizer.update_vol(0.10);  // vol_factor = 2.0

    // regime_factor = 1.5, vol_factor = 2.0 → 3.0 (but vol_cap = 3.0, so fine)
    EXPECT_GT(sizer.size_multiplier(), 1.5);
}

// ============================================================
// on_size_change callback fires when change exceeds threshold
// ============================================================

TEST(RegimeAwareSizer, CallbackFiresOnSignificantChange) {
    RegimeAwareSizer::Config cfg;
    cfg.change_threshold = 0.05;
    cfg.regime_max       = 1.5;
    cfg.hurst_max        = 0.8;
    std::atomic<int> fires{0};
    cfg.on_size_change = [&](double, double) { ++fires; };
    RegimeAwareSizer sizer(cfg);

    sizer.update_hurst(0.5);  // baseline
    sizer.update_hurst(0.8);  // large jump → fires callback
    EXPECT_GT(fires.load(), 0);
    EXPECT_GT(sizer.change_events(), 0u);
}

// ============================================================
// reset() returns to neutral
// ============================================================

TEST(RegimeAwareSizer, ResetRestoresNeutral) {
    RegimeAwareSizer sizer;
    sizer.update_hurst(0.8);
    sizer.update_vol(0.10);
    sizer.reset();
    EXPECT_DOUBLE_EQ(sizer.current_hurst(), 0.5);
    EXPECT_DOUBLE_EQ(sizer.size_multiplier(), 1.0);
    EXPECT_EQ(sizer.update_count(), 0u);
}

// ============================================================
// update_config resets state
// ============================================================

TEST(RegimeAwareSizer, UpdateConfigResetsState) {
    RegimeAwareSizer sizer;
    sizer.update_hurst(0.9);
    RegimeAwareSizer::Config cfg2;
    sizer.update_config(cfg2);
    EXPECT_DOUBLE_EQ(sizer.size_multiplier(), 1.0);
    EXPECT_EQ(sizer.update_count(), 0u);
}

// ============================================================
// update_count increments on each call
// ============================================================

TEST(RegimeAwareSizer, UpdateCountIncrements) {
    RegimeAwareSizer sizer;
    sizer.update_hurst(0.5);
    sizer.update_vol(0.20);
    sizer.update_hurst(0.7);
    EXPECT_EQ(sizer.update_count(), 3u);
}

// ============================================================
// to_stats_json contains required keys
// ============================================================

TEST(RegimeAwareSizer, ToStatsJsonContainsKeys) {
    RegimeAwareSizer sizer;
    sizer.update_hurst(0.6);
    sizer.update_vol(0.20);
    std::string j = sizer.to_stats_json();
    EXPECT_NE(j.find("size_multiplier"),  std::string::npos);
    EXPECT_NE(j.find("regime_factor"),    std::string::npos);
    EXPECT_NE(j.find("vol_factor"),       std::string::npos);
    EXPECT_NE(j.find("hurst"),            std::string::npos);
    EXPECT_NE(j.find("update_count"),     std::string::npos);
}

// ============================================================
// Concurrent updates are thread-safe
// ============================================================

TEST(RegimeAwareSizer, ConcurrentUpdatesSafe) {
    RegimeAwareSizer::Config cfg;
    cfg.change_threshold = 0.01;
    RegimeAwareSizer sizer(cfg);

    std::vector<std::thread> threads;
    for (int t = 0; t < 4; ++t) {
        threads.emplace_back([&, t] {
            for (int i = 0; i < 100; ++i) {
                sizer.update_hurst(0.3 + (i % 6) * 0.1);
                sizer.update_vol(0.10 + (i % 4) * 0.05);
            }
        });
    }
    for (auto& th : threads) th.join();

    double m = sizer.size_multiplier();
    EXPECT_GT(m, 0.0);  // should never go negative
}

} // namespace

#else  // LLMQUANT_REGIME_SIZER_ENABLED not defined

TEST(RegimeAwareSizer, DisabledAtBuildTime) {
    SUCCEED();
}

#endif  // LLMQUANT_REGIME_SIZER_ENABLED
