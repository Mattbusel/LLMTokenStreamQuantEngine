/// @file tests/unit/test_signal_combiner.cpp
/// @brief GTest unit tests for SignalCombiner and SignalHistory — 25+ tests.

#include "signal_combiner.hpp"

#include <cmath>
#include <limits>

#include <gtest/gtest.h>

using namespace llmquant;

static constexpr double kEps = 1e-9;

// ─── Helpers ─────────────────────────────────────────────────────────────────

static SignalInput make_signal(const std::string& name, double value,
                               double confidence, int64_t ts = 0) {
    return SignalInput{name, value, confidence, ts};
}

static CombinedSignal combine_one(CombineMethod m, SignalInput s) {
    SignalCombiner sc;
    sc.add_signal(s);
    auto res = sc.combine(m);
    EXPECT_TRUE(res.has_value());
    return *res;
}

// ─── SignalInput clamping ─────────────────────────────────────────────────────

TEST(SignalCombinerClampTest, ValueClampedToMinus1Plus1) {
    SignalCombiner sc;
    sc.add_signal(make_signal("X", 5.0, 1.0));
    auto r = sc.combine(CombineMethod::WEIGHTED_AVERAGE);
    ASSERT_TRUE(r.has_value());
    EXPECT_LE(r->action, 1.0);
    EXPECT_GE(r->action, -1.0);
}

TEST(SignalCombinerClampTest, ConfidenceClampedTo01) {
    SignalCombiner sc;
    sc.add_signal(make_signal("X", 0.5, -0.5));  // negative confidence => 0
    auto r = sc.combine(CombineMethod::WEIGHTED_AVERAGE);
    ASSERT_TRUE(r.has_value());
    EXPECT_NEAR(r->action, 0.0, kEps);  // zero-confidence signal = no contribution
}

// ─── Empty combiner ───────────────────────────────────────────────────────────

TEST(SignalCombinerEmptyTest, CombineEmptyReturnsNullopt) {
    SignalCombiner sc;
    EXPECT_FALSE(sc.combine(CombineMethod::WEIGHTED_AVERAGE).has_value());
    EXPECT_FALSE(sc.combine(CombineMethod::MAJORITY).has_value());
    EXPECT_FALSE(sc.combine(CombineMethod::ENSEMBLE).has_value());
}

// ─── WeightedAverage tests ────────────────────────────────────────────────────

TEST(WeightedAverageTest, SingleSignalPassthrough) {
    auto r = combine_one(CombineMethod::WEIGHTED_AVERAGE,
                         make_signal("s", 0.6, 1.0));
    EXPECT_NEAR(r.action, 0.6, kEps);
}

TEST(WeightedAverageTest, EqualConfidenceAverages) {
    SignalCombiner sc;
    sc.add_signal(make_signal("a", 0.8, 0.5));
    sc.add_signal(make_signal("b", 0.2, 0.5));
    auto r = sc.combine(CombineMethod::WEIGHTED_AVERAGE);
    ASSERT_TRUE(r.has_value());
    EXPECT_NEAR(r->action, 0.5, kEps);
}

TEST(WeightedAverageTest, HigherConfidenceWeighsMore) {
    SignalCombiner sc;
    sc.add_signal(make_signal("strong", 0.9, 0.9));
    sc.add_signal(make_signal("weak",   0.1, 0.1));
    auto r = sc.combine(CombineMethod::WEIGHTED_AVERAGE);
    ASSERT_TRUE(r.has_value());
    // weighted = (0.9*0.9 + 0.1*0.1)/(0.9+0.1) = (0.81+0.01)/1.0 = 0.82
    EXPECT_NEAR(r->action, 0.82, 1e-9);
}

TEST(WeightedAverageTest, NeutralSignalCancels) {
    SignalCombiner sc;
    sc.add_signal(make_signal("bull", 1.0, 0.5));
    sc.add_signal(make_signal("bear", -1.0, 0.5));
    auto r = sc.combine(CombineMethod::WEIGHTED_AVERAGE);
    ASSERT_TRUE(r.has_value());
    EXPECT_NEAR(r->action, 0.0, kEps);
}

TEST(WeightedAverageTest, ContributingSignalsContainAllNames) {
    SignalCombiner sc;
    sc.add_signal(make_signal("sent", 0.5, 0.8));
    sc.add_signal(make_signal("alpha", 0.3, 0.6));
    auto r = sc.combine(CombineMethod::WEIGHTED_AVERAGE);
    ASSERT_TRUE(r.has_value());
    auto& names = r->contributing_signals;
    EXPECT_NE(std::find(names.begin(), names.end(), "sent"),  names.end());
    EXPECT_NE(std::find(names.begin(), names.end(), "alpha"), names.end());
}

TEST(WeightedAverageTest, ActionInMinus1Plus1) {
    SignalCombiner sc;
    sc.add_signal(make_signal("a", 1.0, 1.0));
    sc.add_signal(make_signal("b", 1.0, 1.0));
    auto r = sc.combine(CombineMethod::WEIGHTED_AVERAGE);
    ASSERT_TRUE(r.has_value());
    EXPECT_LE(r->action, 1.0 + 1e-9);
    EXPECT_GE(r->action, -1.0 - 1e-9);
}

// ─── Majority tests ───────────────────────────────────────────────────────────

TEST(MajorityTest, MoreBullsThanBears) {
    SignalCombiner sc;
    sc.add_signal(make_signal("a", 0.8, 0.6));
    sc.add_signal(make_signal("b", 0.5, 0.4));
    sc.add_signal(make_signal("c", -0.4, 0.1));
    auto r = sc.combine(CombineMethod::MAJORITY);
    ASSERT_TRUE(r.has_value());
    EXPECT_GT(r->action, 0.0);
}

TEST(MajorityTest, NeutralZeroValue) {
    SignalCombiner sc;
    sc.add_signal(make_signal("a", 0.0, 0.5));
    sc.add_signal(make_signal("b", 0.0, 0.5));
    auto r = sc.combine(CombineMethod::MAJORITY);
    ASSERT_TRUE(r.has_value());
    EXPECT_NEAR(r->action, 0.0, kEps);
}

TEST(MajorityTest, MajorityEquivalentStrengthBears) {
    SignalCombiner sc;
    sc.add_signal(make_signal("a", -0.8, 0.7));
    sc.add_signal(make_signal("b",  0.3, 0.3));
    auto r = sc.combine(CombineMethod::MAJORITY);
    ASSERT_TRUE(r.has_value());
    EXPECT_LT(r->action, 0.0);
}

// ─── Ensemble tests ───────────────────────────────────────────────────────────

TEST(EnsembleTest, SingleHighConfidenceConvergesQuickly) {
    // K = 0.9/(0.9+0.1) = 0.9; x = 0 + 0.9*(0.8-0) = 0.72
    auto r = combine_one(CombineMethod::ENSEMBLE,
                         make_signal("s", 0.8, 0.9));
    EXPECT_NEAR(r.action, 0.72, 1e-9);
}

TEST(EnsembleTest, LowConfidenceSmallUpdate) {
    // K = 0.1/(0.1+0.9) = 0.1; x = 0 + 0.1*(1.0-0) = 0.1
    auto r = combine_one(CombineMethod::ENSEMBLE,
                         make_signal("s", 1.0, 0.1));
    EXPECT_NEAR(r.action, 0.1, 1e-9);
}

TEST(EnsembleTest, ActionBounded) {
    SignalCombiner sc;
    for (int i = 0; i < 20; ++i) {
        sc.add_signal(make_signal("s", 1.0, 1.0));
    }
    auto r = sc.combine(CombineMethod::ENSEMBLE);
    ASSERT_TRUE(r.has_value());
    EXPECT_LE(r->action, 1.0 + 1e-9);
}

// ─── Regime classification tests ─────────────────────────────────────────────

TEST(RegimeTest, TrendingAbove40) {
    SignalCombiner sc;
    sc.add_signal(make_signal("s", 0.8, 1.0));
    auto r = sc.combine(CombineMethod::WEIGHTED_AVERAGE);
    ASSERT_TRUE(r.has_value());
    EXPECT_EQ(r->regime, SignalRegime::TRENDING);
}

TEST(RegimeTest, MeanRevertingBelow15) {
    SignalCombiner sc;
    sc.add_signal(make_signal("s", 0.10, 1.0));
    auto r = sc.combine(CombineMethod::WEIGHTED_AVERAGE);
    ASSERT_TRUE(r.has_value());
    EXPECT_EQ(r->regime, SignalRegime::MEAN_REVERTING);
}

TEST(RegimeTest, UncertainBetweenThresholds) {
    SignalCombiner sc;
    sc.add_signal(make_signal("s", 0.25, 1.0));
    auto r = sc.combine(CombineMethod::WEIGHTED_AVERAGE);
    ASSERT_TRUE(r.has_value());
    EXPECT_EQ(r->regime, SignalRegime::UNCERTAIN);
}

TEST(RegimeTest, NegativeTrendingAlsoTrending) {
    SignalCombiner sc;
    sc.add_signal(make_signal("s", -0.7, 1.0));
    auto r = sc.combine(CombineMethod::WEIGHTED_AVERAGE);
    ASSERT_TRUE(r.has_value());
    EXPECT_EQ(r->regime, SignalRegime::TRENDING);
}

// ─── reset() tests ────────────────────────────────────────────────────────────

TEST(ResetTest, AfterResetNullopt) {
    SignalCombiner sc;
    sc.add_signal(make_signal("s", 0.5, 0.8));
    sc.reset();
    EXPECT_EQ(sc.signal_count(), 0u);
    EXPECT_FALSE(sc.combine().has_value());
}

// ─── SignalHistory tests ──────────────────────────────────────────────────────

TEST(SignalHistoryTest, EmptyHistoryLatestNullopt) {
    SignalHistory h;
    EXPECT_FALSE(h.latest().has_value());
}

TEST(SignalHistoryTest, PushAndLatest) {
    SignalHistory h;
    CombinedSignal cs{0.5, 0.8, {"a"}, SignalRegime::UNCERTAIN};
    h.push(cs);
    auto l = h.latest();
    ASSERT_TRUE(l.has_value());
    EXPECT_NEAR(l->action, 0.5, kEps);
}

TEST(SignalHistoryTest, RingBufferCapacityRespected) {
    SignalHistory h(3);
    for (int i = 0; i < 10; ++i) {
        CombinedSignal cs{static_cast<double>(i) * 0.1, 0.5, {}, SignalRegime::UNCERTAIN};
        h.push(cs);
    }
    EXPECT_EQ(h.size(), 3u);
}

TEST(SignalHistoryTest, TrendDirectionPositive) {
    SignalHistory h;
    for (int i = 0; i < 5; ++i) {
        h.push(CombinedSignal{0.8, 0.9, {}, SignalRegime::TRENDING});
    }
    EXPECT_EQ(h.trend_direction(), 1);
}

TEST(SignalHistoryTest, TrendDirectionNegative) {
    SignalHistory h;
    for (int i = 0; i < 5; ++i) {
        h.push(CombinedSignal{-0.8, 0.9, {}, SignalRegime::TRENDING});
    }
    EXPECT_EQ(h.trend_direction(), -1);
}

TEST(SignalHistoryTest, TrendDirectionFlat) {
    SignalHistory h;
    for (int i = 0; i < 4; ++i) {
        h.push(CombinedSignal{0.01, 0.5, {}, SignalRegime::MEAN_REVERTING});
    }
    EXPECT_EQ(h.trend_direction(), 0);
}

TEST(SignalHistoryTest, VolatilityZeroForSingleEntry) {
    SignalHistory h;
    h.push(CombinedSignal{0.5, 0.8, {}, SignalRegime::UNCERTAIN});
    EXPECT_NEAR(h.volatility(), 0.0, kEps);
}

TEST(SignalHistoryTest, VolatilityPositiveForVariedEntries) {
    SignalHistory h;
    h.push(CombinedSignal{0.8, 0.9, {}, SignalRegime::TRENDING});
    h.push(CombinedSignal{-0.8, 0.9, {}, SignalRegime::TRENDING});
    EXPECT_GT(h.volatility(), 0.0);
}

TEST(SignalHistoryTest, ClearEmptiesBuffer) {
    SignalHistory h;
    h.push(CombinedSignal{0.5, 0.8, {}, SignalRegime::UNCERTAIN});
    h.clear();
    EXPECT_EQ(h.size(), 0u);
}

TEST(SignalHistoryTest, CapacityOneKeepsOnlyLatest) {
    SignalHistory h(1);
    h.push(CombinedSignal{0.1, 0.5, {}, SignalRegime::MEAN_REVERTING});
    h.push(CombinedSignal{0.9, 0.9, {}, SignalRegime::TRENDING});
    EXPECT_EQ(h.size(), 1u);
    EXPECT_NEAR(h.latest()->action, 0.9, kEps);
}
