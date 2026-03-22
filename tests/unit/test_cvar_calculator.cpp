#include <gtest/gtest.h>

#ifdef LLMQUANT_CVAR_ENABLED

#include "CVaRCalculator.h"

#include <cmath>

using namespace llmquant;

namespace {

// ============================================================
// Construction
// ============================================================

TEST(CVaRCalculator, DefaultConstructsZero) {
    CVaRCalculator calc;
    EXPECT_NEAR(calc.cvar(), 0.0, 1e-12);
    EXPECT_NEAR(calc.var(),  0.0, 1e-12);
    EXPECT_EQ(calc.sample_count(), 0);
    EXPECT_FALSE(calc.is_in_breach());
}

// ============================================================
// Returns 0 before min_samples
// ============================================================

TEST(CVaRCalculator, ZeroBeforeMinSamples) {
    CVaRCalculator::Config cfg;
    cfg.min_samples = 10;
    CVaRCalculator calc(cfg);

    for (int i = 0; i < 5; ++i) calc.record(-0.01 * i);
    EXPECT_NEAR(calc.cvar(), 0.0, 1e-12);
}

// ============================================================
// VaR is the (1-alpha) quantile
// ============================================================

TEST(CVaRCalculator, VaRIsCorrectQuantile) {
    CVaRCalculator::Config cfg;
    cfg.alpha       = 0.90;  // 90% → 10th percentile
    cfg.min_samples = 5;
    cfg.window_size = 100;
    CVaRCalculator calc(cfg);

    // Feed 100 uniform values: -0.99, -0.98, ..., -0.00, +0.01
    for (int i = 0; i < 100; ++i)
        calc.record(-0.99 + i * 0.01);

    // 10th percentile of [-0.99, .., 0] is ≈ -0.90
    EXPECT_LT(calc.var(), -0.85);
    EXPECT_GT(calc.var(), -0.95);
}

// ============================================================
// CVaR is strictly worse than VaR
// ============================================================

TEST(CVaRCalculator, CVaRWorseThanVaR) {
    CVaRCalculator::Config cfg;
    cfg.alpha       = 0.95;
    cfg.min_samples = 10;
    CVaRCalculator calc(cfg);

    for (int i = 0; i < 50; ++i) calc.record(-0.01 * i);
    EXPECT_LT(calc.cvar(), calc.var());
}

// ============================================================
// All positive P&L → CVaR is still defined
// ============================================================

TEST(CVaRCalculator, AllPositivePnL) {
    CVaRCalculator::Config cfg;
    cfg.alpha       = 0.95;
    cfg.min_samples = 10;
    CVaRCalculator calc(cfg);

    for (int i = 0; i < 30; ++i) calc.record(0.01 * (i + 1));
    // Even all-profit portfolios have a CVaR (the smallest gains).
    EXPECT_GT(calc.cvar(), 0.0);
    EXPECT_GT(calc.var(),  0.0);
}

// ============================================================
// Rolling window cap enforced
// ============================================================

TEST(CVaRCalculator, WindowCapEnforced) {
    CVaRCalculator::Config cfg;
    cfg.window_size = 10;
    cfg.min_samples = 2;
    CVaRCalculator calc(cfg);

    for (int i = 0; i < 30; ++i) calc.record(-0.01);
    EXPECT_EQ(calc.sample_count(), 10);
}

// ============================================================
// total_records counts all calls
// ============================================================

TEST(CVaRCalculator, TotalRecordsCounted) {
    CVaRCalculator calc;
    for (int i = 0; i < 12; ++i) calc.record(0.0);
    EXPECT_EQ(calc.total_records(), 12u);
}

// ============================================================
// Breach detection
// ============================================================

TEST(CVaRCalculator, BreachDetectedWhenCVaRBelowThreshold) {
    CVaRCalculator::Config cfg;
    cfg.alpha             = 0.95;
    cfg.min_samples       = 10;
    cfg.breach_threshold  = -0.005;  // breach if CVaR < -0.5%
    CVaRCalculator calc(cfg);

    // Feed large losses.
    for (int i = 0; i < 30; ++i) calc.record(-0.1);
    EXPECT_TRUE(calc.is_in_breach());
    EXPECT_GE(calc.breach_events(), 1u);
}

// ============================================================
// Breach callback fires
// ============================================================

TEST(CVaRCalculator, BreachCallbackFires) {
    CVaRCalculator::Config cfg;
    cfg.alpha            = 0.95;
    cfg.min_samples      = 5;
    cfg.breach_threshold = -0.001;
    int fires = 0;
    cfg.on_breach = [&](double, double, double) { ++fires; };
    CVaRCalculator calc(cfg);

    for (int i = 0; i < 20; ++i) calc.record(-0.05);
    EXPECT_GE(fires, 1);
}

// ============================================================
// reset
// ============================================================

TEST(CVaRCalculator, ResetClearsAll) {
    CVaRCalculator::Config cfg;
    cfg.min_samples = 5;
    CVaRCalculator calc(cfg);

    for (int i = 0; i < 20; ++i) calc.record(-0.01);
    calc.reset();
    EXPECT_EQ(calc.sample_count(), 0);
    EXPECT_NEAR(calc.cvar(), 0.0, 1e-12);
    EXPECT_NEAR(calc.var(),  0.0, 1e-12);
}

// ============================================================
// update_config resets state
// ============================================================

TEST(CVaRCalculator, UpdateConfigResetsState) {
    CVaRCalculator calc;
    for (int i = 0; i < 30; ++i) calc.record(-0.01);
    CVaRCalculator::Config cfg2;
    cfg2.alpha       = 0.99;
    cfg2.window_size = 50;
    calc.update_config(cfg2);
    EXPECT_EQ(calc.sample_count(), 0);
}

// ============================================================
// to_stats_json contains required keys
// ============================================================

TEST(CVaRCalculator, ToStatsJsonContainsKeys) {
    CVaRCalculator::Config cfg;
    cfg.min_samples = 5;
    CVaRCalculator calc(cfg);
    for (int i = 0; i < 20; ++i) calc.record(-0.01 * i);
    std::string json = calc.to_stats_json();
    EXPECT_NE(json.find("cvar"),           std::string::npos);
    EXPECT_NE(json.find("var"),            std::string::npos);
    EXPECT_NE(json.find("alpha"),          std::string::npos);
    EXPECT_NE(json.find("sample_count"),   std::string::npos);
    EXPECT_NE(json.find("total_records"),  std::string::npos);
    EXPECT_NE(json.find("breach_events"),  std::string::npos);
    EXPECT_NE(json.find("is_in_breach"),   std::string::npos);
}

} // namespace

#else  // LLMQUANT_CVAR_ENABLED not defined

TEST(CVaRCalculator, DisabledAtBuildTime) {
    SUCCEED();
}

#endif  // LLMQUANT_CVAR_ENABLED
