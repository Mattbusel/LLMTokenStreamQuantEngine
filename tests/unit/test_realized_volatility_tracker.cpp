#include <gtest/gtest.h>

#ifdef LLMQUANT_REALIZED_VOL_ENABLED

#include "RealizedVolatilityTracker.h"
#include <atomic>
#include <cmath>
#include <thread>
#include <vector>

using namespace llmquant;

namespace {

TEST(RealizedVolatilityTracker, DefaultState) {
    RealizedVolatilityTracker rvt;
    EXPECT_DOUBLE_EQ(rvt.realized_vol(), 0.0);
    EXPECT_FALSE(rvt.is_high_volatility());
    EXPECT_EQ(rvt.total_records(), 0u);
    EXPECT_EQ(rvt.alert_events(), 0u);
}

// First record seeds without computing a return; RV stays 0.
TEST(RealizedVolatilityTracker, FirstRecordSeeds) {
    RealizedVolatilityTracker rvt;
    rvt.record(0.5);
    EXPECT_EQ(rvt.total_records(), 1u);
    EXPECT_DOUBLE_EQ(rvt.realized_vol(), 0.0);
}

// Constant bias → all returns = 0 → RV = 0.
TEST(RealizedVolatilityTracker, ConstantBiasZeroRV) {
    RealizedVolatilityTracker::Config cfg;
    cfg.window = 10;
    RealizedVolatilityTracker rvt(cfg);
    for (int i = 0; i < 15; ++i) rvt.record(0.5);
    EXPECT_DOUBLE_EQ(rvt.realized_vol(), 0.0);
}

// Known return → known RV.
// All returns = 0.1.  RV = sqrt(mean(0.01, 0.01, ...)) = 0.1.
TEST(RealizedVolatilityTracker, KnownReturnsGiveCorrectRV) {
    RealizedVolatilityTracker::Config cfg;
    cfg.window          = 8;
    cfg.alert_threshold = 10.0;  // suppress alert
    RealizedVolatilityTracker rvt(cfg);
    for (int i = 0; i < 10; ++i)
        rvt.record(static_cast<double>(i) * 0.1);
    // All consecutive differences = 0.1 → RV = 0.1.
    EXPECT_NEAR(rvt.realized_vol(), 0.1, 1e-9);
}

// RV is non-negative.
TEST(RealizedVolatilityTracker, RVNonNegative) {
    RealizedVolatilityTracker::Config cfg;
    cfg.window = 16;
    RealizedVolatilityTracker rvt(cfg);
    for (int i = 0; i < 20; ++i)
        rvt.record(static_cast<double>(i % 5) * 0.2 - 0.4);
    EXPECT_GE(rvt.realized_vol(), 0.0);
}

// High-volatility callback fires when RV exceeds threshold.
TEST(RealizedVolatilityTracker, HighVolCallbackFires) {
    RealizedVolatilityTracker::Config cfg;
    cfg.window          = 4;
    cfg.alert_threshold = 0.05;
    std::atomic<int> fires{0};
    cfg.on_high_volatility = [&](double) { ++fires; };
    RealizedVolatilityTracker rvt(cfg);

    // Large alternating returns → high RV.
    for (int i = 0; i < 10; ++i)
        rvt.record((i % 2 == 0) ? 1.0 : -1.0);

    EXPECT_GT(fires.load(), 0);
    EXPECT_TRUE(rvt.is_high_volatility());
    EXPECT_GT(rvt.alert_events(), 0u);
}

// Low-volatility (clear) callback fires when RV retreats.
TEST(RealizedVolatilityTracker, LowVolCallbackFires) {
    RealizedVolatilityTracker::Config cfg;
    cfg.window           = 4;
    cfg.alert_threshold  = 0.05;
    cfg.clear_hysteresis = 0.5;  // clear at RV < 0.025
    std::atomic<int> cleared{0};
    cfg.on_low_volatility = [&](double) { ++cleared; };
    RealizedVolatilityTracker rvt(cfg);

    // Cause high vol.
    for (int i = 0; i < 8; ++i)
        rvt.record((i % 2 == 0) ? 1.0 : -1.0);
    ASSERT_TRUE(rvt.is_high_volatility());

    // Return to constant bias → returns = 0 → RV → 0.
    for (int i = 0; i < 8; ++i) rvt.record(0.0);

    EXPECT_GT(cleared.load(), 0);
}

// Reset clears all state.
TEST(RealizedVolatilityTracker, ResetClearsState) {
    RealizedVolatilityTracker::Config cfg;
    cfg.window          = 4;
    cfg.alert_threshold = 0.01;
    RealizedVolatilityTracker rvt(cfg);
    for (int i = 0; i < 10; ++i)
        rvt.record((i % 2 == 0) ? 1.0 : -1.0);
    rvt.reset();
    EXPECT_DOUBLE_EQ(rvt.realized_vol(), 0.0);
    EXPECT_EQ(rvt.total_records(), 0u);
    EXPECT_FALSE(rvt.is_high_volatility());
}

// update_config resets state.
TEST(RealizedVolatilityTracker, UpdateConfigResetsState) {
    RealizedVolatilityTracker rvt;
    for (int i = 0; i < 5; ++i) rvt.record(static_cast<double>(i));
    RealizedVolatilityTracker::Config cfg2;
    cfg2.window = 4;
    rvt.update_config(cfg2);
    EXPECT_EQ(rvt.total_records(), 0u);
    EXPECT_DOUBLE_EQ(rvt.realized_vol(), 0.0);
}

// to_stats_json contains expected fields.
TEST(RealizedVolatilityTracker, StatsJsonContainsFields) {
    RealizedVolatilityTracker rvt;
    rvt.record(0.1); rvt.record(0.2);
    std::string j = rvt.to_stats_json();
    EXPECT_NE(j.find("realized_vol"),     std::string::npos);
    EXPECT_NE(j.find("is_high_volatility"), std::string::npos);
    EXPECT_NE(j.find("total_records"),    std::string::npos);
    EXPECT_NE(j.find("alert_events"),     std::string::npos);
}

// Concurrent record() is thread-safe.
TEST(RealizedVolatilityTracker, ConcurrentRecordSafe) {
    RealizedVolatilityTracker::Config cfg;
    cfg.window = 32;
    RealizedVolatilityTracker rvt(cfg);

    std::atomic<bool> go{false};
    auto worker = [&] {
        while (!go.load()) {}
        for (int i = 0; i < 250; ++i)
            rvt.record(static_cast<double>(i % 7) * 0.1 - 0.3);
    };
    std::vector<std::thread> threads;
    for (int t = 0; t < 4; ++t) threads.emplace_back(worker);
    go.store(true);
    for (auto& t : threads) t.join();

    EXPECT_EQ(rvt.total_records(), 1000u);
    EXPECT_GE(rvt.realized_vol(), 0.0);
}

}  // namespace

#else

TEST(RealizedVolatilityTracker, DisabledAtBuildTime) { SUCCEED(); }

#endif  // LLMQUANT_REALIZED_VOL_ENABLED
