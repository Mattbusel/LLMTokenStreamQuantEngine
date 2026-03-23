/**
 * @file  tests/unit/test_regime_sizer.cpp
 * @brief GTest unit tests for the Regime-Adaptive Position Sizer (Round 7).
 *
 * Covers RegimeClassifier, PositionSizer, and the SizingConfig defaults.
 */

#include <gtest/gtest.h>

#ifdef LLMQUANT_REGIME_ADAPTIVE_SIZER_ENABLED

#include "regime_sizer.hpp"

#include <cmath>

using namespace llmquant;

// ============================================================================
// Helper builders
// ============================================================================

static RegimeFeatures make_features(
    double avg_sentiment          = 0.0,
    double sentiment_vol          = 0.0,
    double cross_asset_correlation = 0.5,
    double alpha_decay_rate       = 0.0)
{
    RegimeFeatures f;
    f.avg_sentiment            = avg_sentiment;
    f.sentiment_vol            = sentiment_vol;
    f.cross_asset_correlation  = cross_asset_correlation;
    f.alpha_decay_rate         = alpha_decay_rate;
    return f;
}

// ============================================================================
// RegimeClassifier — classification rules
// ============================================================================

// --- Rule 1: HIGH_VOL -------------------------------------------------------

TEST(RegimeClassifier, HighSentimentVolGivesHighVol) {
    RegimeClassifier clf;
    auto f = make_features(0.5, 0.5 /* > 0.4 */, 0.8);
    EXPECT_EQ(clf.classify(f), MarketRegime::HIGH_VOL);
}

TEST(RegimeClassifier, SentimentVolExactlyAtThresholdNotHighVol) {
    RegimeClassifier clf;
    // 0.4 is NOT strictly greater than threshold 0.4
    auto f = make_features(0.5, 0.4, 0.8);
    EXPECT_NE(clf.classify(f), MarketRegime::HIGH_VOL);
}

TEST(RegimeClassifier, HighVolTakesPrecedenceOverTrending) {
    RegimeClassifier clf;
    // Both high vol and high correlation — HIGH_VOL wins
    auto f = make_features(0.9, 0.45, 0.9);
    EXPECT_EQ(clf.classify(f), MarketRegime::HIGH_VOL);
}

// --- Rule 2: TRENDING_BULL --------------------------------------------------

TEST(RegimeClassifier, HighCorrPositiveSentGivesTrendingBull) {
    RegimeClassifier clf;
    auto f = make_features(0.3 /* > 0 */, 0.1, 0.8 /* > 0.7 */);
    EXPECT_EQ(clf.classify(f), MarketRegime::TRENDING_BULL);
}

TEST(RegimeClassifier, HighCorrZeroSentGivesTrendingBear) {
    RegimeClassifier clf;
    // avg_sentiment == 0 → not > 0 → TRENDING_BEAR
    auto f = make_features(0.0, 0.1, 0.8);
    EXPECT_EQ(clf.classify(f), MarketRegime::TRENDING_BEAR);
}

// --- Rule 3: TRENDING_BEAR --------------------------------------------------

TEST(RegimeClassifier, HighCorrNegativeSentGivesTrendingBear) {
    RegimeClassifier clf;
    auto f = make_features(-0.5, 0.1, 0.9);
    EXPECT_EQ(clf.classify(f), MarketRegime::TRENDING_BEAR);
}

// --- Rule 4: MEAN_REVERTING -------------------------------------------------

TEST(RegimeClassifier, LowCorrGivesMeanReverting) {
    RegimeClassifier clf;
    auto f = make_features(0.5, 0.1, 0.2 /* < 0.3 */);
    EXPECT_EQ(clf.classify(f), MarketRegime::MEAN_REVERTING);
}

TEST(RegimeClassifier, LowCorrCorrExactlyAtThresholdNotMeanReverting) {
    RegimeClassifier clf;
    // 0.3 is NOT strictly less than threshold 0.3
    auto f = make_features(0.5, 0.1, 0.3);
    EXPECT_NE(clf.classify(f), MarketRegime::MEAN_REVERTING);
}

// --- Rule 5: LOW_VOL --------------------------------------------------------

TEST(RegimeClassifier, ModerateConditionsPositiveSentGivesLowVol) {
    RegimeClassifier clf;
    auto f = make_features(0.1, 0.1, 0.5);
    EXPECT_EQ(clf.classify(f), MarketRegime::LOW_VOL);
}

// --- Rule 6: UNKNOWN --------------------------------------------------------

TEST(RegimeClassifier, NegativeSentModerateCorrGivesUnknown) {
    RegimeClassifier clf;
    auto f = make_features(-0.3, 0.1, 0.5);
    EXPECT_EQ(clf.classify(f), MarketRegime::UNKNOWN);
}

// --- Custom thresholds ------------------------------------------------------

TEST(RegimeClassifier, CustomThresholdsApplied) {
    RegimeClassifier::Thresholds t;
    t.high_vol_sentiment_vol = 0.2;  // lower threshold
    RegimeClassifier clf(t);
    auto f = make_features(0.5, 0.25 /* > 0.2 */, 0.5);
    EXPECT_EQ(clf.classify(f), MarketRegime::HIGH_VOL);
}

// --- regime_name helper -----------------------------------------------------

TEST(RegimeName, AllRegimesHaveNames) {
    EXPECT_STREQ(regime_name(MarketRegime::TRENDING_BULL),   "TRENDING_BULL");
    EXPECT_STREQ(regime_name(MarketRegime::TRENDING_BEAR),   "TRENDING_BEAR");
    EXPECT_STREQ(regime_name(MarketRegime::MEAN_REVERTING),  "MEAN_REVERTING");
    EXPECT_STREQ(regime_name(MarketRegime::HIGH_VOL),        "HIGH_VOL");
    EXPECT_STREQ(regime_name(MarketRegime::LOW_VOL),         "LOW_VOL");
    EXPECT_STREQ(regime_name(MarketRegime::UNKNOWN),         "UNKNOWN");
}

// ============================================================================
// SizingConfig defaults
// ============================================================================

TEST(SizingConfig, DefaultBaseSizeIsPositive) {
    SizingConfig cfg;
    EXPECT_GT(cfg.base_size, 0.0);
}

TEST(SizingConfig, AllRegimesHaveDefaultMultipliers) {
    SizingConfig cfg;
    EXPECT_GT(cfg.regime_multipliers.count(MarketRegime::TRENDING_BULL), 0u);
    EXPECT_GT(cfg.regime_multipliers.count(MarketRegime::TRENDING_BEAR), 0u);
    EXPECT_GT(cfg.regime_multipliers.count(MarketRegime::MEAN_REVERTING), 0u);
    EXPECT_GT(cfg.regime_multipliers.count(MarketRegime::HIGH_VOL), 0u);
    EXPECT_GT(cfg.regime_multipliers.count(MarketRegime::LOW_VOL), 0u);
    EXPECT_GT(cfg.regime_multipliers.count(MarketRegime::UNKNOWN), 0u);
}

TEST(SizingConfig, HighVolMultiplierSmallerThanBull) {
    SizingConfig cfg;
    EXPECT_LT(cfg.regime_multipliers.at(MarketRegime::HIGH_VOL),
              cfg.regime_multipliers.at(MarketRegime::TRENDING_BULL));
}

// ============================================================================
// PositionSizer
// ============================================================================

TEST(PositionSizer, ZeroEquityReturnsZeroNotional) {
    PositionSizer sizer;
    auto ps = sizer.size(MarketRegime::LOW_VOL, 0.8, 0.0);
    EXPECT_DOUBLE_EQ(ps.notional_usd, 0.0);
    EXPECT_DOUBLE_EQ(ps.shares_or_contracts, 0.0);
}

TEST(PositionSizer, ZeroConfidenceReturnsZeroNotional) {
    PositionSizer sizer;
    auto ps = sizer.size(MarketRegime::LOW_VOL, 0.0, 100000.0);
    EXPECT_DOUBLE_EQ(ps.notional_usd, 0.0);
}

TEST(PositionSizer, FullConfidencePositiveNotional) {
    PositionSizer sizer;
    auto ps = sizer.size(MarketRegime::LOW_VOL, 1.0, 100000.0);
    EXPECT_GT(ps.notional_usd, 0.0);
}

TEST(PositionSizer, TrendingBullLargerThanHighVol) {
    PositionSizer sizer;
    auto bull = sizer.size(MarketRegime::TRENDING_BULL, 1.0, 100000.0);
    auto hvol = sizer.size(MarketRegime::HIGH_VOL,      1.0, 100000.0);
    EXPECT_GT(bull.notional_usd, hvol.notional_usd);
}

TEST(PositionSizer, RegimeStoredInResult) {
    PositionSizer sizer;
    auto ps = sizer.size(MarketRegime::MEAN_REVERTING, 0.5, 100000.0);
    EXPECT_EQ(ps.regime, MarketRegime::MEAN_REVERTING);
}

TEST(PositionSizer, ConfidenceClampedAboveOne) {
    PositionSizer sizer;
    auto unclamped = sizer.size(MarketRegime::LOW_VOL, 1.0, 100000.0);
    auto clamped   = sizer.size(MarketRegime::LOW_VOL, 2.0, 100000.0);
    EXPECT_DOUBLE_EQ(unclamped.notional_usd, clamped.notional_usd);
}

TEST(PositionSizer, ConfidenceClampedBelowZero) {
    PositionSizer sizer;
    auto ps = sizer.size(MarketRegime::LOW_VOL, -0.5, 100000.0);
    EXPECT_DOUBLE_EQ(ps.notional_usd, 0.0);
}

TEST(PositionSizer, SharesComputedFromPrice) {
    PositionSizer sizer;
    auto ps = sizer.size(MarketRegime::LOW_VOL, 1.0, 100000.0, 50.0);
    EXPECT_NEAR(ps.shares_or_contracts, ps.notional_usd / 50.0, 1e-9);
}

TEST(PositionSizer, IsValidFalseForZeroNotional) {
    PositionSizer sizer;
    auto ps = sizer.size(MarketRegime::LOW_VOL, 0.0, 100000.0);
    EXPECT_FALSE(ps.is_valid());
}

TEST(PositionSizer, IsValidTrueForPositiveNotional) {
    PositionSizer sizer;
    auto ps = sizer.size(MarketRegime::LOW_VOL, 1.0, 100000.0);
    EXPECT_TRUE(ps.is_valid());
}

TEST(PositionSizer, UnknownRegimeUsesDefaultMultiplier) {
    PositionSizer sizer;
    // UNKNOWN regime — multiplier from config
    auto ps = sizer.size(MarketRegime::UNKNOWN, 1.0, 100000.0);
    EXPECT_GE(ps.notional_usd, 0.0);
}

TEST(PositionSizer, UpdateConfigChangesOutput) {
    PositionSizer sizer;
    auto before = sizer.size(MarketRegime::LOW_VOL, 1.0, 100000.0);

    SizingConfig big;
    big.base_size = 0.20;  // 4x the default
    sizer.update_config(big);
    auto after = sizer.size(MarketRegime::LOW_VOL, 1.0, 100000.0);

    EXPECT_GT(after.notional_usd, before.notional_usd);
}

TEST(PositionSizer, ZeroPriceReturnsZeroShares) {
    PositionSizer sizer;
    auto ps = sizer.size(MarketRegime::LOW_VOL, 1.0, 100000.0, 0.0);
    EXPECT_DOUBLE_EQ(ps.shares_or_contracts, 0.0);
}

TEST(PositionSizer, ConfidenceStoredInResult) {
    PositionSizer sizer;
    auto ps = sizer.size(MarketRegime::LOW_VOL, 0.65, 100000.0);
    EXPECT_DOUBLE_EQ(ps.confidence, 0.65);
}

TEST(PositionSizer, NotionalScalesWithEquity) {
    PositionSizer sizer;
    auto ps1 = sizer.size(MarketRegime::LOW_VOL, 1.0, 50000.0);
    auto ps2 = sizer.size(MarketRegime::LOW_VOL, 1.0, 100000.0);
    EXPECT_NEAR(ps2.notional_usd / ps1.notional_usd, 2.0, 1e-9);
}

#else  // LLMQUANT_REGIME_ADAPTIVE_SIZER_ENABLED

// Placeholder when the feature flag is disabled.
TEST(RegimeSizer, FeatureFlagDisabled) {
    GTEST_SKIP() << "LLMQUANT_REGIME_ADAPTIVE_SIZER_ENABLED not defined";
}

#endif // LLMQUANT_REGIME_ADAPTIVE_SIZER_ENABLED
