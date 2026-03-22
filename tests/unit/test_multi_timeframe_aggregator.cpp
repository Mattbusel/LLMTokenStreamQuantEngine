#include <gtest/gtest.h>

#ifdef LLMQUANT_MULTI_TIMEFRAME_ENABLED

#include "MultiTimeframeAggregator.h"

#include <cmath>
#include <string>

using namespace llmquant;

namespace {

// ============================================================
// Construction
// ============================================================

TEST(MultiTimeframeAggregator, ConstructsOk) {
    MultiTimeframeAggregator mta;
    EXPECT_EQ(mta.total_records(), 0u);
    EXPECT_EQ(mta.divergence_events(), 0u);
    EXPECT_NEAR(mta.consensus(), 0.0, 1e-12);
}

// ============================================================
// Default 4 timeframes
// ============================================================

TEST(MultiTimeframeAggregator, DefaultTimeframesArePresent) {
    MultiTimeframeAggregator mta;
    mta.record(1.0);
    // Fastest EMA (alpha=1.0) should snap to 1.0 immediately.
    EXPECT_NEAR(mta.ema("1s"), 1.0, 1e-9);
    // Slowest EMA (alpha=0.01) should be much less.
    EXPECT_LT(mta.ema("5m"), mta.ema("1s"));
}

// ============================================================
// EMA convergence
// ============================================================

TEST(MultiTimeframeAggregator, EmaConvergesWithAlphaOne) {
    MultiTimeframeAggregator::Config cfg;
    cfg.timeframes = {{"fast", 1.0, 1.0}};
    cfg.diverge_threshold = 0.0;  // disable divergence
    MultiTimeframeAggregator mta(cfg);

    mta.record(0.5);
    EXPECT_NEAR(mta.ema("fast"), 0.5, 1e-9);
    mta.record(0.8);
    EXPECT_NEAR(mta.ema("fast"), 0.8, 1e-9);
}

TEST(MultiTimeframeAggregator, EmaSmoothedWithSmallAlpha) {
    MultiTimeframeAggregator::Config cfg;
    cfg.timeframes = {{"slow", 0.1, 1.0}};
    cfg.diverge_threshold = 0.0;
    MultiTimeframeAggregator mta(cfg);

    // After one record with value=1.0: ema = 0.1 * 1.0 + 0.9 * 0.0 = 0.1
    mta.record(1.0);
    EXPECT_NEAR(mta.ema("slow"), 0.1, 1e-9);
    // After second record: ema = 0.1 * 1.0 + 0.9 * 0.1 = 0.19
    mta.record(1.0);
    EXPECT_NEAR(mta.ema("slow"), 0.19, 1e-9);
}

// ============================================================
// Consensus (weighted blend)
// ============================================================

TEST(MultiTimeframeAggregator, ConsensusIsWeightedAverage) {
    MultiTimeframeAggregator::Config cfg;
    cfg.timeframes = {
        {"A", 1.0, 1.0},  // EMA → 0.5 weight=1
        {"B", 1.0, 3.0},  // EMA → 0.5 weight=3
    };
    cfg.diverge_threshold = 0.0;
    MultiTimeframeAggregator mta(cfg);
    mta.record(0.5);
    // consensus = (1.0*0.5 + 3.0*0.5) / (1.0+3.0) = 0.5
    EXPECT_NEAR(mta.consensus(), 0.5, 1e-9);
}

TEST(MultiTimeframeAggregator, ConsensusBlendsDifferentEmas) {
    MultiTimeframeAggregator::Config cfg;
    cfg.timeframes = {
        {"fast", 1.00, 1.0},  // snaps immediately
        {"slow", 0.01, 1.0},  // very slow
    };
    cfg.diverge_threshold = 0.0;
    MultiTimeframeAggregator mta(cfg);

    mta.record(1.0);
    double fast_ema = mta.ema("fast");
    double slow_ema = mta.ema("slow");
    double expected_consensus = (fast_ema + slow_ema) / 2.0;
    EXPECT_NEAR(mta.consensus(), expected_consensus, 1e-9);
}

// ============================================================
// timeframe_spread
// ============================================================

TEST(MultiTimeframeAggregator, SpreadZeroBeforeRecords) {
    MultiTimeframeAggregator mta;
    EXPECT_NEAR(mta.timeframe_spread(), 0.0, 1e-9);
}

TEST(MultiTimeframeAggregator, SpreadNonZeroAfterRecords) {
    MultiTimeframeAggregator mta;
    for (int i = 0; i < 10; ++i) mta.record(1.0);
    double spread = mta.timeframe_spread();
    // Fast EMA ≈ 1.0, slow EMA still catching up → spread > 0
    EXPECT_GT(spread, 0.0);
}

// ============================================================
// Unknown timeframe name returns 0
// ============================================================

TEST(MultiTimeframeAggregator, UnknownTimeframeReturnsZero) {
    MultiTimeframeAggregator mta;
    mta.record(1.0);
    EXPECT_NEAR(mta.ema("nonexistent"), 0.0, 1e-9);
}

// ============================================================
// Divergence detection
// ============================================================

TEST(MultiTimeframeAggregator, DivergenceCallbackFires) {
    MultiTimeframeAggregator::Config cfg;
    cfg.timeframes = {
        {"fast", 1.00, 1.0},
        {"slow", 0.01, 1.0},
    };
    cfg.diverge_threshold = 0.05;
    int fired = 0;
    cfg.on_divergence = [&](double, double, double) { ++fired; };
    MultiTimeframeAggregator mta(cfg);

    // After first record, fast=1.0, slow=0.01 => spread ≈ 0.99 > 0.05
    mta.record(1.0);
    EXPECT_GT(fired, 0);
    EXPECT_TRUE(mta.is_diverging());
}

TEST(MultiTimeframeAggregator, NoDivergenceWhenAllSame) {
    MultiTimeframeAggregator::Config cfg;
    cfg.timeframes = {
        {"A", 1.0, 1.0},
        {"B", 1.0, 1.0},
    };
    cfg.diverge_threshold = 0.1;
    int fired = 0;
    cfg.on_divergence = [&](double, double, double) { ++fired; };
    MultiTimeframeAggregator mta(cfg);
    mta.record(0.5);
    // Both EMAs snap to 0.5 => spread = 0 < threshold
    EXPECT_EQ(fired, 0);
    EXPECT_FALSE(mta.is_diverging());
}

// ============================================================
// reset
// ============================================================

TEST(MultiTimeframeAggregator, ResetZerosEmas) {
    MultiTimeframeAggregator mta;
    for (int i = 0; i < 5; ++i) mta.record(1.0);
    EXPECT_GT(mta.ema("1s"), 0.0);
    mta.reset();
    EXPECT_NEAR(mta.ema("1s"), 0.0, 1e-12);
    EXPECT_NEAR(mta.consensus(), 0.0, 1e-12);
    EXPECT_FALSE(mta.is_diverging());
}

// ============================================================
// update_config
// ============================================================

TEST(MultiTimeframeAggregator, UpdateConfigResetsEmas) {
    MultiTimeframeAggregator mta;
    for (int i = 0; i < 5; ++i) mta.record(1.0);

    MultiTimeframeAggregator::Config cfg2;
    cfg2.timeframes = {{"x", 0.5, 1.0}};
    cfg2.diverge_threshold = 0.0;
    mta.update_config(cfg2);
    EXPECT_NEAR(mta.ema("x"), 0.0, 1e-12);  // reset on config change
    EXPECT_TRUE(std::isnan(mta.ema("1s")) || mta.ema("1s") == 0.0);  // old name gone
}

// ============================================================
// total_records counter
// ============================================================

TEST(MultiTimeframeAggregator, TotalRecordsCountsCorrectly) {
    MultiTimeframeAggregator mta;
    mta.record(0.1);
    mta.record(0.2);
    mta.record(0.3);
    EXPECT_EQ(mta.total_records(), 3u);
}

// ============================================================
// to_stats_json
// ============================================================

TEST(MultiTimeframeAggregator, StatsJsonContainsRequiredKeys) {
    MultiTimeframeAggregator mta;
    mta.record(0.5);
    std::string js = mta.to_stats_json();
    EXPECT_NE(js.find("total_records"),     std::string::npos);
    EXPECT_NE(js.find("divergence_events"), std::string::npos);
    EXPECT_NE(js.find("is_diverging"),      std::string::npos);
    EXPECT_NE(js.find("emas"),              std::string::npos);
}

TEST(MultiTimeframeAggregator, StatsJsonContainsTimeframeNames) {
    MultiTimeframeAggregator mta;
    mta.record(0.5);
    std::string js = mta.to_stats_json();
    EXPECT_NE(js.find("1s"),  std::string::npos);
    EXPECT_NE(js.find("5m"),  std::string::npos);
}

} // namespace

#else  // LLMQUANT_MULTI_TIMEFRAME_ENABLED not defined

TEST(MultiTimeframeAggregator, DisabledAtBuildTime) {
    SUCCEED();
}

#endif  // LLMQUANT_MULTI_TIMEFRAME_ENABLED
