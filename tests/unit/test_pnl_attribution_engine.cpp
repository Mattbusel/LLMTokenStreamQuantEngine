#include <gtest/gtest.h>

#ifdef LLMQUANT_PNL_ATTRIBUTION_ENABLED

#include "PnLAttributionEngine.h"
#include <cmath>

using namespace llmquant;

namespace {

static TradeSignal make_signal(double bias, double conf = 0.8,
                                double vol = 0.01, double sq = 0.9) {
    TradeSignal s{};
    s.delta_bias_shift      = bias;
    s.confidence            = conf;
    s.volatility_adjustment = vol;
    s.signal_quality        = sq;
    return s;
}

// ============================================================
// Construction
// ============================================================

TEST(PnLAttributionEngine, DefaultConstructsEmpty) {
    PnLAttributionEngine attr;
    EXPECT_EQ(attr.total_trades(), 0u);
    EXPECT_NEAR(attr.total_pnl(), 0.0, 1e-12);
    EXPECT_TRUE(attr.all_drivers().empty());
}

// ============================================================
// Basic driver classification
// ============================================================

TEST(PnLAttributionEngine, BullishBiasClassified) {
    PnLAttributionEngine attr;
    attr.record_outcome(make_signal(0.5), 100.0);

    auto drivers = attr.all_drivers();
    bool found_bullish = false;
    for (const auto& d : drivers)
        if (d.label == "bias:bullish") { found_bullish = true; break; }
    EXPECT_TRUE(found_bullish);
}

TEST(PnLAttributionEngine, BearishBiasClassified) {
    PnLAttributionEngine attr;
    attr.record_outcome(make_signal(-0.5), -50.0);

    auto drivers = attr.all_drivers();
    bool found_bearish = false;
    for (const auto& d : drivers)
        if (d.label == "bias:bearish") { found_bearish = true; break; }
    EXPECT_TRUE(found_bearish);
}

TEST(PnLAttributionEngine, NeutralBiasClassified) {
    PnLAttributionEngine attr;
    attr.record_outcome(make_signal(0.01), 10.0);  // within neutral_band(0.05)

    auto drivers = attr.all_drivers();
    bool found_neutral = false;
    for (const auto& d : drivers)
        if (d.label == "bias:neutral") { found_neutral = true; break; }
    EXPECT_TRUE(found_neutral);
}

// ============================================================
// Confidence band classification
// ============================================================

TEST(PnLAttributionEngine, HighConfidenceClassified) {
    PnLAttributionEngine attr;
    attr.record_outcome(make_signal(0.5, /*conf=*/0.9), 50.0);

    auto drivers = attr.all_drivers();
    bool found = false;
    for (const auto& d : drivers)
        if (d.label == "conf:high") { found = true; break; }
    EXPECT_TRUE(found);
}

TEST(PnLAttributionEngine, LowConfidenceClassified) {
    PnLAttributionEngine attr;
    attr.record_outcome(make_signal(0.5, /*conf=*/0.2), 10.0);

    auto drivers = attr.all_drivers();
    bool found = false;
    for (const auto& d : drivers)
        if (d.label == "conf:low") { found = true; break; }
    EXPECT_TRUE(found);
}

// ============================================================
// Win rate accumulation
// ============================================================

TEST(PnLAttributionEngine, WinRateAccumulatesCorrectly) {
    PnLAttributionEngine attr;
    // 3 bullish trades: 2 wins, 1 loss
    attr.record_outcome(make_signal(0.5), 100.0);
    attr.record_outcome(make_signal(0.5), 200.0);
    attr.record_outcome(make_signal(0.5), -50.0);

    auto drivers = attr.all_drivers();
    for (const auto& d : drivers) {
        if (d.label == "bias:bullish") {
            EXPECT_EQ(d.trade_count, 3u);
            EXPECT_EQ(d.win_count,   2u);
            EXPECT_NEAR(d.win_rate(), 2.0/3.0, 1e-9);
            EXPECT_NEAR(d.total_pnl, 250.0, 1e-9);
            return;
        }
    }
    FAIL() << "bias:bullish driver not found";
}

// ============================================================
// top_drivers / bottom_drivers
// ============================================================

TEST(PnLAttributionEngine, TopDriversReturnsSorted) {
    PnLAttributionEngine attr;
    attr.record_outcome(make_signal(0.5, 0.9, 0.01, 0.9), 500.0);  // high conf, good sq
    attr.record_outcome(make_signal(0.5, 0.3, 0.01, 0.3), -200.0); // low conf, poor sq

    auto top = attr.top_drivers(1);
    ASSERT_EQ(top.size(), 1u);
    EXPECT_GT(top[0].total_pnl, 0.0);
}

TEST(PnLAttributionEngine, BottomDriversReturnWorstFirst) {
    PnLAttributionEngine attr;
    // Create a bearish + low-quality combo for negative P&L
    attr.record_outcome(make_signal(-0.5, 0.3, 0.05, 0.3), -300.0);
    attr.record_outcome(make_signal( 0.5, 0.9, 0.01, 0.9),  600.0);

    auto bottom = attr.bottom_drivers(1);
    ASSERT_GE(bottom.size(), 1u);
    EXPECT_LT(bottom[0].total_pnl, 0.0);
}

TEST(PnLAttributionEngine, TopDriversClampedToAvailable) {
    PnLAttributionEngine attr;
    attr.record_outcome(make_signal(0.5), 100.0);  // generates 4 drivers
    auto top = attr.top_drivers(100);
    EXPECT_EQ(top.size(), 4u);  // only 4 dimension labels
}

// ============================================================
// Total P&L
// ============================================================

TEST(PnLAttributionEngine, TotalPnLAccumulates) {
    PnLAttributionEngine attr;
    attr.record_outcome(make_signal(0.5),  200.0);
    attr.record_outcome(make_signal(-0.5), -75.0);
    EXPECT_NEAR(attr.total_pnl(), 125.0, 1e-9);
    EXPECT_EQ(attr.total_trades(), 2u);
}

// ============================================================
// reset
// ============================================================

TEST(PnLAttributionEngine, ResetClearsEverything) {
    PnLAttributionEngine attr;
    attr.record_outcome(make_signal(0.5), 100.0);
    attr.reset();
    EXPECT_EQ(attr.total_trades(), 0u);
    EXPECT_NEAR(attr.total_pnl(), 0.0, 1e-12);
    EXPECT_TRUE(attr.all_drivers().empty());
}

// ============================================================
// update_config
// ============================================================

TEST(PnLAttributionEngine, UpdateConfigResetsAndApplies) {
    PnLAttributionEngine attr;
    attr.record_outcome(make_signal(0.5), 100.0);

    PnLAttributionEngine::Config cfg;
    cfg.neutral_band = 0.9;  // everything becomes neutral
    attr.update_config(cfg);

    EXPECT_EQ(attr.total_trades(), 0u);  // cleared
    attr.record_outcome(make_signal(0.5), 50.0);

    auto drivers = attr.all_drivers();
    bool found_neutral = false;
    for (const auto& d : drivers)
        if (d.label == "bias:neutral") { found_neutral = true; break; }
    EXPECT_TRUE(found_neutral);
}

// ============================================================
// on_outcome callback
// ============================================================

TEST(PnLAttributionEngine, OnOutcomeCallbackFires) {
    int fired = 0;
    PnLAttributionEngine::Config cfg;
    cfg.on_outcome = [&](const PnLAttributionEngine::DriverStats&) { ++fired; };
    PnLAttributionEngine attr(cfg);
    attr.record_outcome(make_signal(0.5), 100.0);
    attr.record_outcome(make_signal(-0.5), -20.0);
    EXPECT_EQ(fired, 2);
}

// ============================================================
// avg_pnl helper
// ============================================================

TEST(PnLAttributionEngine, AvgPnLComputedPerDriver) {
    PnLAttributionEngine attr;
    attr.record_outcome(make_signal(0.5), 100.0);
    attr.record_outcome(make_signal(0.5), 200.0);
    for (const auto& d : attr.all_drivers()) {
        if (d.label == "bias:bullish") {
            EXPECT_NEAR(d.avg_pnl(), 150.0, 1e-9);
            return;
        }
    }
    FAIL() << "bias:bullish not found";
}

// ============================================================
// to_stats_json
// ============================================================

TEST(PnLAttributionEngine, ToStatsJsonContainsKeys) {
    PnLAttributionEngine attr;
    attr.record_outcome(make_signal(0.5), 100.0);
    std::string json = attr.to_stats_json();
    EXPECT_NE(json.find("total_trades"),  std::string::npos);
    EXPECT_NE(json.find("total_pnl"),     std::string::npos);
    EXPECT_NE(json.find("driver_count"),  std::string::npos);
}

} // namespace

#else  // LLMQUANT_PNL_ATTRIBUTION_ENABLED not defined

TEST(PnLAttributionEngine, DisabledAtBuildTime) {
    SUCCEED();
}

#endif  // LLMQUANT_PNL_ATTRIBUTION_ENABLED
