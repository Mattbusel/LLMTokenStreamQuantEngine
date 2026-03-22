#include <gtest/gtest.h>

#ifdef LLMQUANT_PORTFOLIO_HEAT_ENABLED

#include "PortfolioHeatMonitor.h"
#include <atomic>
#include <cmath>

using namespace llmquant;

namespace {

// ============================================================
// Construction
// ============================================================

TEST(PortfolioHeatMonitor, DefaultConstructsCool) {
    PortfolioHeatMonitor heat;
    EXPECT_EQ(heat.heat_level(), PortfolioHeatMonitor::Level::Cool);
    EXPECT_NEAR(heat.total_heat(), 0.0, 1e-12);
    EXPECT_EQ(heat.instrument_count(), 0u);
}

// ============================================================
// level_name
// ============================================================

TEST(PortfolioHeatMonitor, LevelNamesAreCorrect) {
    using L = PortfolioHeatMonitor::Level;
    EXPECT_STREQ(PortfolioHeatMonitor::level_name(L::Cool),     "Cool");
    EXPECT_STREQ(PortfolioHeatMonitor::level_name(L::Warm),     "Warm");
    EXPECT_STREQ(PortfolioHeatMonitor::level_name(L::Hot),      "Hot");
    EXPECT_STREQ(PortfolioHeatMonitor::level_name(L::Critical), "Critical");
}

// ============================================================
// Single instrument heat
// ============================================================

TEST(PortfolioHeatMonitor, SingleInstrumentHeatCorrect) {
    PortfolioHeatMonitor::Config cfg;
    cfg.heat_scale                = 1.0;
    cfg.double_correlated_heat    = false;
    cfg.warn_heat                 = 100.0;  // high thresholds to avoid callbacks
    cfg.critical_heat             = 200.0;
    PortfolioHeatMonitor heat(cfg);

    heat.update_instrument("AAPL", 0.5, 4.0);  // heat = 0.5 * 4.0 * 1.0 = 2.0
    EXPECT_NEAR(heat.total_heat(), 2.0, 1e-9);
}

TEST(PortfolioHeatMonitor, MultiInstrumentHeatSumsCorrectly) {
    PortfolioHeatMonitor::Config cfg;
    cfg.heat_scale             = 1.0;
    cfg.double_correlated_heat = false;
    cfg.warn_heat              = 100.0;
    cfg.critical_heat          = 200.0;
    PortfolioHeatMonitor heat(cfg);

    heat.update_instrument("AAPL", 0.5,  4.0);  // 2.0
    heat.update_instrument("MSFT", 0.25, 8.0);  // 2.0
    EXPECT_NEAR(heat.total_heat(), 4.0, 1e-9);
}

// ============================================================
// Correlated-burst penalty
// ============================================================

TEST(PortfolioHeatMonitor, CorrelatedBurstAddsHeat) {
    PortfolioHeatMonitor::Config cfg;
    cfg.heat_scale                  = 1.0;
    cfg.double_correlated_heat      = true;
    cfg.correlation_gate_threshold  = 0.3;
    cfg.warn_heat                   = 100.0;
    cfg.critical_heat               = 200.0;
    PortfolioHeatMonitor heat(cfg);

    // Both bullish above gate threshold → correlation penalty applies.
    heat.update_instrument("AAPL", 0.5, 4.0);  // heat_i = 2.0
    heat.update_instrument("MSFT", 0.5, 4.0);  // heat_j = 2.0
    // penalty = sqrt(2.0 * 2.0) = 2.0
    // total = 2.0 + 2.0 + 2.0 = 6.0
    EXPECT_NEAR(heat.total_heat(), 6.0, 1e-9);
}

TEST(PortfolioHeatMonitor, OppositeSignsNoCorrelationPenalty) {
    PortfolioHeatMonitor::Config cfg;
    cfg.heat_scale                 = 1.0;
    cfg.double_correlated_heat     = true;
    cfg.correlation_gate_threshold = 0.3;
    cfg.warn_heat                  = 100.0;
    cfg.critical_heat              = 200.0;
    PortfolioHeatMonitor heat(cfg);

    heat.update_instrument("AAPL",  0.5, 4.0);  // bullish
    heat.update_instrument("MSFT", -0.5, 4.0);  // bearish — opposite sign, no penalty
    EXPECT_NEAR(heat.total_heat(), 4.0, 1e-9);
}

// ============================================================
// Level transitions and callbacks
// ============================================================

TEST(PortfolioHeatMonitor, WarnCallbackFiredAtHotThreshold) {
    PortfolioHeatMonitor::Config cfg;
    cfg.heat_scale             = 1.0;
    cfg.double_correlated_heat = false;
    cfg.warn_heat              = 5.0;
    cfg.critical_heat          = 10.0;
    PortfolioHeatMonitor heat(cfg);

    std::atomic<int> fired{0};
    heat.set_warn_callback([&](double) { fired.fetch_add(1); });

    heat.update_instrument("X", 0.6, 9.0);   // heat = 5.4 → Hot
    EXPECT_EQ(fired.load(), 1);
    EXPECT_EQ(heat.heat_level(), PortfolioHeatMonitor::Level::Hot);
}

TEST(PortfolioHeatMonitor, CriticalCallbackFiredAboveCritical) {
    PortfolioHeatMonitor::Config cfg;
    cfg.heat_scale             = 1.0;
    cfg.double_correlated_heat = false;
    cfg.warn_heat              = 5.0;
    cfg.critical_heat          = 10.0;
    PortfolioHeatMonitor heat(cfg);

    std::atomic<int> fired{0};
    heat.set_critical_callback([&](double) { fired.fetch_add(1); });

    heat.update_instrument("X", 1.0, 11.0);  // heat = 11 → Critical
    EXPECT_EQ(fired.load(), 1);
    EXPECT_EQ(heat.heat_level(), PortfolioHeatMonitor::Level::Critical);
}

TEST(PortfolioHeatMonitor, RecoveryCallbackFiredOnCoolDown) {
    PortfolioHeatMonitor::Config cfg;
    cfg.heat_scale             = 1.0;
    cfg.double_correlated_heat = false;
    cfg.warn_heat              = 5.0;
    cfg.critical_heat          = 10.0;
    PortfolioHeatMonitor heat(cfg);

    std::atomic<int> recovered{0};
    heat.set_recovery_callback([&](double) { recovered.fetch_add(1); });

    heat.update_instrument("X", 1.0, 11.0);  // → Critical
    heat.update_instrument("X", 0.0,  0.0);  // → Cool → recovery fires
    EXPECT_EQ(recovered.load(), 1);
    EXPECT_EQ(heat.heat_level(), PortfolioHeatMonitor::Level::Cool);
}

TEST(PortfolioHeatMonitor, WarnFiresOnlyOnceOnTransition) {
    PortfolioHeatMonitor::Config cfg;
    cfg.heat_scale             = 1.0;
    cfg.double_correlated_heat = false;
    cfg.warn_heat              = 5.0;
    cfg.critical_heat          = 10.0;
    PortfolioHeatMonitor heat(cfg);

    std::atomic<int> fired{0};
    heat.set_warn_callback([&](double) { fired.fetch_add(1); });

    heat.update_instrument("X", 0.6, 9.0);   // → Hot (fires)
    heat.update_instrument("X", 0.7, 9.0);   // stays Hot (no re-fire)
    EXPECT_EQ(fired.load(), 1);
}

// ============================================================
// remove_instrument
// ============================================================

TEST(PortfolioHeatMonitor, RemoveInstrumentReducesHeat) {
    PortfolioHeatMonitor::Config cfg;
    cfg.heat_scale             = 1.0;
    cfg.double_correlated_heat = false;
    cfg.warn_heat              = 100.0;
    cfg.critical_heat          = 200.0;
    PortfolioHeatMonitor heat(cfg);

    heat.update_instrument("AAPL", 0.5, 4.0);
    heat.update_instrument("MSFT", 0.5, 4.0);
    EXPECT_EQ(heat.instrument_count(), 2u);

    heat.remove_instrument("AAPL");
    EXPECT_EQ(heat.instrument_count(), 1u);
    EXPECT_NEAR(heat.total_heat(), 2.0, 1e-9);
}

// ============================================================
// snapshot
// ============================================================

TEST(PortfolioHeatMonitor, SnapshotContainsAllInstruments) {
    PortfolioHeatMonitor::Config cfg;
    cfg.heat_scale             = 1.0;
    cfg.double_correlated_heat = false;
    cfg.warn_heat              = 100.0;
    cfg.critical_heat          = 200.0;
    PortfolioHeatMonitor heat(cfg);

    heat.update_instrument("AAPL", 0.5, 2.0);
    heat.update_instrument("MSFT", 0.3, 1.0);
    auto snap = heat.snapshot();
    EXPECT_EQ(snap.size(), 2u);
}

// ============================================================
// reset
// ============================================================

TEST(PortfolioHeatMonitor, ResetClearsInstruments) {
    PortfolioHeatMonitor heat;
    heat.update_instrument("AAPL", 1.0, 10.0);
    heat.reset();
    EXPECT_EQ(heat.instrument_count(), 0u);
    EXPECT_NEAR(heat.total_heat(), 0.0, 1e-12);
    EXPECT_EQ(heat.heat_level(), PortfolioHeatMonitor::Level::Cool);
}

// ============================================================
// update_config
// ============================================================

TEST(PortfolioHeatMonitor, UpdateConfigChangesThresholds) {
    PortfolioHeatMonitor heat;
    PortfolioHeatMonitor::Config cfg;
    cfg.heat_scale             = 2.0;   // doubled
    cfg.double_correlated_heat = false;
    cfg.warn_heat              = 100.0;
    cfg.critical_heat          = 200.0;
    heat.update_config(cfg);

    heat.update_instrument("X", 1.0, 5.0);  // heat = 1*5*2 = 10
    EXPECT_NEAR(heat.total_heat(), 10.0, 1e-9);
}

// ============================================================
// to_stats_json
// ============================================================

TEST(PortfolioHeatMonitor, ToStatsJsonContainsKeys) {
    PortfolioHeatMonitor heat;
    heat.update_instrument("AAPL", 0.5, 4.0);
    std::string json = heat.to_stats_json();
    EXPECT_NE(json.find("level"),         std::string::npos);
    EXPECT_NE(json.find("total_heat"),    std::string::npos);
    EXPECT_NE(json.find("instruments"),   std::string::npos);
    EXPECT_NE(json.find("warn_heat"),     std::string::npos);
    EXPECT_NE(json.find("critical_heat"), std::string::npos);
}

} // namespace

#else  // LLMQUANT_PORTFOLIO_HEAT_ENABLED not defined

TEST(PortfolioHeatMonitor, DisabledAtBuildTime) {
    SUCCEED();
}

#endif  // LLMQUANT_PORTFOLIO_HEAT_ENABLED
