#include <gtest/gtest.h>

#ifdef LLMQUANT_REGIME_DETECTOR_ENABLED

#include "RegimeDetector.h"

#include <atomic>
#include <thread>
#include <vector>

using namespace llmquant;

namespace {

// ============================================================
// Construction
// ============================================================

TEST(RegimeDetector, DefaultConstructsAsNeutral) {
    RegimeDetector rd;
    EXPECT_EQ(rd.current_regime(), RegimeDetector::Regime::Neutral);
    EXPECT_EQ(rd.total_updates(),     0u);
    EXPECT_EQ(rd.total_transitions(), 0u);
}

// ============================================================
// regime_name
// ============================================================

TEST(RegimeDetector, RegimeNamesAreCorrect) {
    EXPECT_STREQ(RegimeDetector::regime_name(RegimeDetector::Regime::Neutral),  "Neutral");
    EXPECT_STREQ(RegimeDetector::regime_name(RegimeDetector::Regime::Bull),     "Bull");
    EXPECT_STREQ(RegimeDetector::regime_name(RegimeDetector::Regime::Bear),     "Bear");
    EXPECT_STREQ(RegimeDetector::regime_name(RegimeDetector::Regime::Volatile), "Volatile");
    EXPECT_STREQ(RegimeDetector::regime_name(RegimeDetector::Regime::RiskOff),  "RiskOff");
}

// ============================================================
// Bull detection
// ============================================================

TEST(RegimeDetector, ConvergesToBullOnPositiveBias) {
    RegimeDetector::Config cfg;
    cfg.momentum_alpha     = 0.5;
    cfg.momentum_threshold = 0.1;
    cfg.volatility_threshold = 1.0;   // never triggered
    cfg.risk_off_threshold   = 1.0;   // never triggered
    RegimeDetector rd(cfg);

    for (int i = 0; i < 30; ++i)
        rd.update(0.8, 0.01, false);

    EXPECT_EQ(rd.current_regime(), RegimeDetector::Regime::Bull);
}

// ============================================================
// Bear detection
// ============================================================

TEST(RegimeDetector, ConvergesToBearOnNegativeBias) {
    RegimeDetector::Config cfg;
    cfg.momentum_alpha     = 0.5;
    cfg.momentum_threshold = 0.1;
    cfg.volatility_threshold = 1.0;
    cfg.risk_off_threshold   = 1.0;
    RegimeDetector rd(cfg);

    for (int i = 0; i < 30; ++i)
        rd.update(-0.8, 0.01, false);

    EXPECT_EQ(rd.current_regime(), RegimeDetector::Regime::Bear);
}

// ============================================================
// Volatile detection (priority over Bull)
// ============================================================

TEST(RegimeDetector, VolatileTakesPriorityOverBull) {
    RegimeDetector::Config cfg;
    cfg.momentum_alpha       = 0.5;
    cfg.volatility_alpha     = 0.5;
    cfg.momentum_threshold   = 0.1;
    cfg.volatility_threshold = 0.2;  // easy to hit
    cfg.risk_off_threshold   = 1.0;
    RegimeDetector rd(cfg);

    for (int i = 0; i < 20; ++i)
        rd.update(0.8, 0.8, false);  // high bias AND high vol

    EXPECT_EQ(rd.current_regime(), RegimeDetector::Regime::Volatile);
}

// ============================================================
// RiskOff detection (highest priority)
// ============================================================

TEST(RegimeDetector, RiskOffTakesPriorityOverVolatile) {
    RegimeDetector::Config cfg;
    cfg.block_rate_alpha   = 0.5;
    cfg.risk_off_threshold = 0.5;
    cfg.volatility_threshold = 0.1;  // would be Volatile otherwise
    cfg.momentum_threshold   = 0.01; // would be Bull otherwise
    RegimeDetector rd(cfg);

    for (int i = 0; i < 20; ++i)
        rd.update(0.8, 0.8, true);  // all blocked

    EXPECT_EQ(rd.current_regime(), RegimeDetector::Regime::RiskOff);
}

// ============================================================
// Regime transitions and callback
// ============================================================

TEST(RegimeDetector, CallbackFiredOnRegimeChange) {
    RegimeDetector::Config cfg;
    cfg.momentum_alpha     = 0.9;
    cfg.momentum_threshold = 0.1;
    cfg.volatility_threshold = 1.0;
    cfg.risk_off_threshold   = 1.0;
    RegimeDetector rd(cfg);

    std::atomic<int> cb_count{0};
    RegimeDetector::Regime last_new = RegimeDetector::Regime::Neutral;
    rd.set_regime_change_callback([&](RegimeDetector::Regime n, RegimeDetector::Regime) {
        ++cb_count;
        last_new = n;
    });

    // Drive to Bull.
    for (int i = 0; i < 30; ++i) rd.update(1.0, 0.0, false);
    EXPECT_GT(cb_count.load(), 0);
    EXPECT_EQ(last_new, RegimeDetector::Regime::Bull);
}

TEST(RegimeDetector, TransitionCountIncrementsOnChange) {
    RegimeDetector::Config cfg;
    cfg.momentum_alpha     = 0.9;
    cfg.momentum_threshold = 0.1;
    cfg.volatility_threshold = 1.0;
    cfg.risk_off_threshold   = 1.0;
    RegimeDetector rd(cfg);

    uint64_t before = rd.total_transitions();
    for (int i = 0; i < 30; ++i) rd.update(1.0, 0.0, false);
    EXPECT_GT(rd.total_transitions(), before);
}

TEST(RegimeDetector, NoSpuriousTransitionWhenRegimeStable) {
    RegimeDetector::Config cfg;
    cfg.momentum_alpha     = 0.3;
    cfg.momentum_threshold = 0.5;  // high threshold — stays Neutral
    cfg.volatility_threshold = 1.0;
    cfg.risk_off_threshold   = 1.0;
    RegimeDetector rd(cfg);

    for (int i = 0; i < 50; ++i) rd.update(0.1, 0.0, false);
    EXPECT_EQ(rd.total_transitions(), 0u);
    EXPECT_EQ(rd.current_regime(), RegimeDetector::Regime::Neutral);
}

// ============================================================
// EMA accessors
// ============================================================

TEST(RegimeDetector, MomentumEmaConvergesPositive) {
    RegimeDetector::Config cfg;
    cfg.momentum_alpha = 0.5;
    RegimeDetector rd(cfg);
    for (int i = 0; i < 40; ++i) rd.update(1.0, 0.0, false);
    EXPECT_GT(rd.get_momentum(), 0.9);
}

TEST(RegimeDetector, BlockRateEmaConvergesHigh) {
    RegimeDetector::Config cfg;
    cfg.block_rate_alpha = 0.5;
    RegimeDetector rd(cfg);
    for (int i = 0; i < 40; ++i) rd.update(0.0, 0.0, true);
    EXPECT_GT(rd.get_block_rate_ema(), 0.9);
}

// ============================================================
// reset
// ============================================================

TEST(RegimeDetector, ResetRestoresNeutral) {
    RegimeDetector::Config cfg;
    cfg.momentum_alpha     = 0.9;
    cfg.momentum_threshold = 0.1;
    cfg.volatility_threshold = 1.0;
    cfg.risk_off_threshold   = 1.0;
    RegimeDetector rd(cfg);
    for (int i = 0; i < 30; ++i) rd.update(1.0, 0.0, false);
    EXPECT_EQ(rd.current_regime(), RegimeDetector::Regime::Bull);

    rd.reset();
    EXPECT_EQ(rd.current_regime(), RegimeDetector::Regime::Neutral);
    EXPECT_NEAR(rd.get_momentum(),       0.0, 1e-9);
    EXPECT_NEAR(rd.get_block_rate_ema(), 0.0, 1e-9);
    EXPECT_EQ(rd.total_transitions(),    0u);
    EXPECT_EQ(rd.total_updates(),        0u);
}

// ============================================================
// to_stats_json
// ============================================================

TEST(RegimeDetector, ToStatsJsonContainsKeys) {
    RegimeDetector rd;
    rd.update(0.5, 0.05, false);
    std::string json = rd.to_stats_json();
    EXPECT_NE(json.find("regime"),           std::string::npos);
    EXPECT_NE(json.find("momentum"),         std::string::npos);
    EXPECT_NE(json.find("volatility_ema"),   std::string::npos);
    EXPECT_NE(json.find("block_rate_ema"),   std::string::npos);
    EXPECT_NE(json.find("total_updates"),    std::string::npos);
    EXPECT_NE(json.find("total_transitions"),std::string::npos);
}

// ============================================================
// update_config
// ============================================================

TEST(RegimeDetector, UpdateConfigChangesThreshold) {
    RegimeDetector::Config cfg;
    cfg.momentum_alpha     = 0.9;
    cfg.momentum_threshold = 10.0;  // very high — never Bull
    cfg.volatility_threshold = 100.0;
    cfg.risk_off_threshold   = 100.0;
    RegimeDetector rd(cfg);

    for (int i = 0; i < 30; ++i) rd.update(1.0, 0.0, false);
    EXPECT_EQ(rd.current_regime(), RegimeDetector::Regime::Neutral);

    // Lower the threshold — should now classify as Bull.
    cfg.momentum_threshold = 0.1;
    rd.update_config(cfg);
    rd.update(1.0, 0.0, false);
    EXPECT_EQ(rd.current_regime(), RegimeDetector::Regime::Bull);
}

// ============================================================
// Thread safety
// ============================================================

TEST(RegimeDetector, ConcurrentUpdatesAreSafe) {
    RegimeDetector rd;
    std::atomic<bool> start{false};

    auto worker = [&](double bias, double vol, bool blocked) {
        while (!start.load()) {}
        for (int i = 0; i < 300; ++i)
            rd.update(bias, vol, blocked);
    };

    std::vector<std::thread> threads;
    threads.emplace_back(worker,  0.5, 0.05, false);
    threads.emplace_back(worker, -0.5, 0.15, true);
    threads.emplace_back(worker,  0.0, 0.01, false);
    start.store(true);
    for (auto& t : threads) t.join();

    EXPECT_EQ(rd.total_updates(), 900u);
    SUCCEED();
}

} // namespace

#else  // LLMQUANT_REGIME_DETECTOR_ENABLED not defined

TEST(RegimeDetector, DisabledAtBuildTime) {
    SUCCEED();
}

#endif  // LLMQUANT_REGIME_DETECTOR_ENABLED
