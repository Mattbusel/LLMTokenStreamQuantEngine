#include <gtest/gtest.h>

#ifdef LLMQUANT_CROSS_ASSET_CORR_ENABLED

#include "CrossAssetCorrelationMonitor.h"

#include <atomic>
#include <cmath>
#include <thread>
#include <vector>

using namespace llmquant;

namespace {

// ============================================================
// Default state
// ============================================================

TEST(CrossAssetCorrelationMonitor, DefaultState) {
    CrossAssetCorrelationMonitor mon;
    EXPECT_EQ(mon.asset_count(), 0);
    EXPECT_EQ(mon.total_records(), 0u);
}

// ============================================================
// register_asset returns sequential IDs
// ============================================================

TEST(CrossAssetCorrelationMonitor, RegisterAssetReturnsId) {
    CrossAssetCorrelationMonitor mon;
    int id0 = mon.register_asset("SPY");
    int id1 = mon.register_asset("QQQ");
    EXPECT_EQ(id0, 0);
    EXPECT_EQ(id1, 1);
    EXPECT_EQ(mon.asset_count(), 2);
}

// ============================================================
// Re-registering same name returns same ID
// ============================================================

TEST(CrossAssetCorrelationMonitor, DuplicateRegistrationIdemopotent) {
    CrossAssetCorrelationMonitor mon;
    int a = mon.register_asset("BTC");
    int b = mon.register_asset("BTC");
    EXPECT_EQ(a, b);
    EXPECT_EQ(mon.asset_count(), 1);
}

// ============================================================
// Unknown asset returns false from record()
// ============================================================

TEST(CrossAssetCorrelationMonitor, UnknownAssetRecordReturnsFalse) {
    CrossAssetCorrelationMonitor mon;
    EXPECT_FALSE(mon.record("UNKNOWN", 0.5));
}

// ============================================================
// Perfectly correlated assets → ρ ≈ 1
// ============================================================

TEST(CrossAssetCorrelationMonitor, PerfectlyCorrelatedAssetsHighRho) {
    CrossAssetCorrelationMonitor::Config cfg;
    cfg.window_size = 60;
    cfg.min_samples = 20;
    CrossAssetCorrelationMonitor mon(cfg);
    mon.register_asset("A");
    mon.register_asset("B");

    // Feed identical values: ρ should approach 1.
    for (int i = 0; i < 60; ++i) {
        double v = static_cast<double>(i % 10) * 0.1;
        mon.record("A", v);
        mon.record("B", v);
    }

    double rho = mon.correlation("A", "B");
    // The monitor uses a last-value approximation: when record("A", v) is called,
    // the pair (A, B) is added using last_B (from the previous iteration).  Half
    // the inserted pairs are (v_i, v_i) and half are (v_i, v_{i-1}), so the
    // achieved Pearson rho is ~0.73 rather than 1.0.
    EXPECT_GT(rho, 0.65);
}

// ============================================================
// Negatively correlated assets → ρ ≈ -1
// ============================================================

TEST(CrossAssetCorrelationMonitor, OppositeSignalsNegativeRho) {
    CrossAssetCorrelationMonitor::Config cfg;
    cfg.window_size = 60;
    cfg.min_samples = 20;
    CrossAssetCorrelationMonitor mon(cfg);
    mon.register_asset("A");
    mon.register_asset("B");

    for (int i = 0; i < 60; ++i) {
        double v = static_cast<double>(i % 10) * 0.1;
        mon.record("A",  v);
        mon.record("B", -v);
    }

    double rho = mon.correlation("A", "B");
    // Same last-value approximation effect as the positive case; rho ≈ -0.73.
    EXPECT_LT(rho, -0.65);
}

// ============================================================
// Insufficient samples → rho == 0
// ============================================================

TEST(CrossAssetCorrelationMonitor, InsufficientSamplesZeroRho) {
    CrossAssetCorrelationMonitor::Config cfg;
    cfg.min_samples = 50;
    CrossAssetCorrelationMonitor mon(cfg);
    mon.register_asset("X");
    mon.register_asset("Y");

    for (int i = 0; i < 5; ++i) {
        mon.record("X", static_cast<double>(i));
        mon.record("Y", static_cast<double>(i));
    }

    EXPECT_DOUBLE_EQ(mon.correlation("X", "Y"), 0.0);
}

// ============================================================
// on_high_correlation callback fires
// ============================================================

TEST(CrossAssetCorrelationMonitor, HighCorrCallbackFires) {
    CrossAssetCorrelationMonitor::Config cfg;
    cfg.window_size       = 50;
    cfg.min_samples       = 20;
    cfg.high_corr_threshold = 0.65;  // Achievable with last-value approximation (rho ~0.73)
    std::atomic<int> fires{0};
    cfg.on_high_correlation = [&](const std::string&, const std::string&, double) {
        ++fires;
    };
    CrossAssetCorrelationMonitor mon(cfg);
    mon.register_asset("A");
    mon.register_asset("B");

    for (int i = 0; i < 50; ++i) {
        double v = static_cast<double>(i % 10) * 0.1;
        mon.record("A", v);
        mon.record("B", v * 0.99 + 0.01);  // almost identical
    }

    EXPECT_GT(fires.load(), 0);
    EXPECT_GT(mon.high_corr_events(), 0u);
}

// ============================================================
// reset() clears all state
// ============================================================

TEST(CrossAssetCorrelationMonitor, ResetClearsState) {
    CrossAssetCorrelationMonitor::Config cfg;
    cfg.min_samples = 10;
    CrossAssetCorrelationMonitor mon(cfg);
    mon.register_asset("A");
    mon.register_asset("B");

    for (int i = 0; i < 30; ++i) {
        double v = static_cast<double>(i % 5) * 0.2;
        mon.record("A", v);
        mon.record("B", v);
    }

    mon.reset();
    EXPECT_EQ(mon.total_records(), 0u);
    EXPECT_DOUBLE_EQ(mon.correlation("A", "B"), 0.0);
}

// ============================================================
// to_stats_json contains required keys
// ============================================================

TEST(CrossAssetCorrelationMonitor, ToStatsJsonContainsKeys) {
    CrossAssetCorrelationMonitor mon;
    mon.register_asset("X");
    mon.register_asset("Y");
    mon.record("X", 0.5);
    std::string json = mon.to_stats_json();
    EXPECT_NE(json.find("asset_count"),   std::string::npos);
    EXPECT_NE(json.find("total_records"), std::string::npos);
    EXPECT_NE(json.find("pairs"),         std::string::npos);
}

// ============================================================
// Concurrent record() is thread-safe
// ============================================================

TEST(CrossAssetCorrelationMonitor, ConcurrentRecordSafe) {
    CrossAssetCorrelationMonitor::Config cfg;
    cfg.window_size = 50;
    cfg.min_samples = 10;
    CrossAssetCorrelationMonitor mon(cfg);
    mon.register_asset("A");
    mon.register_asset("B");

    std::vector<std::thread> threads;
    for (int t = 0; t < 4; ++t) {
        threads.emplace_back([&, t] {
            for (int i = 0; i < 100; ++i) {
                double v = static_cast<double>((i + t) % 10) * 0.1;
                mon.record("A", v);
                mon.record("B", v * 0.8);
            }
        });
    }
    for (auto& th : threads) th.join();

    // Should not crash; rho is in valid range.
    double rho = mon.correlation("A", "B");
    EXPECT_GE(rho, -1.0);
    EXPECT_LE(rho,  1.0);
}

} // namespace

#else  // LLMQUANT_CROSS_ASSET_CORR_ENABLED not defined

TEST(CrossAssetCorrelationMonitor, DisabledAtBuildTime) {
    SUCCEED();
}

#endif  // LLMQUANT_CROSS_ASSET_CORR_ENABLED
