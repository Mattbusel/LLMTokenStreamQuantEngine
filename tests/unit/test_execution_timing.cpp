#include <gtest/gtest.h>

#ifdef LLMQUANT_EXECUTION_TIMING_ENABLED

#include "execution_timing.hpp"

#include <cmath>
#include <numeric>
#include <vector>

using namespace llmquant;

// ============================================================
// MarketImpactModel — construction
// ============================================================

TEST(MarketImpactModel, DefaultEtaAlpha) {
    MarketImpactModel m;
    EXPECT_DOUBLE_EQ(m.eta(), 0.1);
    EXPECT_DOUBLE_EQ(m.alpha(), 0.5);
}

TEST(MarketImpactModel, CustomParameters) {
    MarketImpactModel m(0.2, 0.8);
    EXPECT_DOUBLE_EQ(m.eta(), 0.2);
    EXPECT_DOUBLE_EQ(m.alpha(), 0.8);
}

// ============================================================
// MarketImpactModel — estimate
// ============================================================

TEST(MarketImpactModel, EstimateReturnsValueForValidInputs) {
    MarketImpactModel m;
    auto result = m.estimate(10'000.0, 1'000'000.0, 0.015);
    ASSERT_TRUE(result.has_value());
    EXPECT_GT(result->total_impact_bps, 0.0);
}

TEST(MarketImpactModel, SqrtImpactScalesWithSqrtParticipation) {
    MarketImpactModel m(0.1, 0.0);   // alpha=0 → only sqrt component
    auto r1 = m.estimate(10'000.0, 1'000'000.0, 0.01);
    auto r2 = m.estimate(40'000.0, 1'000'000.0, 0.01);
    ASSERT_TRUE(r1.has_value());
    ASSERT_TRUE(r2.has_value());
    // Doubling Q → sqrt(4x) = 2x
    EXPECT_NEAR(r2->sqrt_impact_bps / r1->sqrt_impact_bps, 2.0, 0.01);
}

TEST(MarketImpactModel, LinearImpactScalesLinearlyWithParticipation) {
    MarketImpactModel m(0.0, 0.5);   // eta=0 → only linear component
    auto r1 = m.estimate(10'000.0, 1'000'000.0, 0.01);
    auto r2 = m.estimate(20'000.0, 1'000'000.0, 0.01);
    ASSERT_TRUE(r1.has_value());
    ASSERT_TRUE(r2.has_value());
    EXPECT_NEAR(r2->linear_impact_bps / r1->linear_impact_bps, 2.0, 0.01);
}

TEST(MarketImpactModel, TotalEqualsSumOfComponents) {
    MarketImpactModel m;
    auto r = m.estimate(5'000.0, 500'000.0, 0.02);
    ASSERT_TRUE(r.has_value());
    EXPECT_NEAR(r->total_impact_bps,
                r->linear_impact_bps + r->sqrt_impact_bps, 1e-9);
}

TEST(MarketImpactModel, ZeroOrderQtyReturnsNullopt) {
    MarketImpactModel m;
    EXPECT_FALSE(m.estimate(0.0, 1'000'000.0, 0.01).has_value());
}

TEST(MarketImpactModel, NegativeOrderQtyReturnsNullopt) {
    MarketImpactModel m;
    EXPECT_FALSE(m.estimate(-100.0, 1'000'000.0, 0.01).has_value());
}

TEST(MarketImpactModel, ZeroVolumeReturnsNullopt) {
    MarketImpactModel m;
    EXPECT_FALSE(m.estimate(1'000.0, 0.0, 0.01).has_value());
}

TEST(MarketImpactModel, ZeroVolatilityGivesZeroImpact) {
    MarketImpactModel m;
    auto r = m.estimate(10'000.0, 1'000'000.0, 0.0);
    ASSERT_TRUE(r.has_value());
    EXPECT_DOUBLE_EQ(r->total_impact_bps, 0.0);
}

TEST(MarketImpactModel, ImpactIncreasesWithVolatility) {
    MarketImpactModel m;
    auto lo = m.estimate(10'000.0, 1'000'000.0, 0.01);
    auto hi = m.estimate(10'000.0, 1'000'000.0, 0.05);
    ASSERT_TRUE(lo && hi);
    EXPECT_GT(hi->total_impact_bps, lo->total_impact_bps);
}

// ============================================================
// ExecutionScheduler — construction
// ============================================================

TEST(ExecutionScheduler, DefaultNSlices) {
    ExecutionScheduler s;
    EXPECT_EQ(s.n_slices(), 10);
}

TEST(ExecutionScheduler, CustomNSlices) {
    ExecutionScheduler s(5, 0, 3'600'000);
    EXPECT_EQ(s.n_slices(), 5);
}

TEST(ExecutionScheduler, ZeroNSlicesClampedToOne) {
    ExecutionScheduler s(0);
    EXPECT_EQ(s.n_slices(), 1);
}

// ============================================================
// ExecutionScheduler — twap
// ============================================================

TEST(ExecutionScheduler, TwapReturnsCorrectSliceCount) {
    ExecutionScheduler s(8, 0, 3'600'000);
    auto sched = s.twap(10'000.0, 100.0);
    ASSERT_TRUE(sched.has_value());
    EXPECT_EQ(static_cast<int>(sched->slices.size()), 8);
}

TEST(ExecutionScheduler, TwapSlicesHaveEqualQty) {
    ExecutionScheduler s(5, 0, 3'600'000);
    auto sched = s.twap(1'000.0, 50.0);
    ASSERT_TRUE(sched.has_value());
    for (auto& sl : sched->slices) {
        EXPECT_NEAR(sl.target_qty, 200.0, 1e-9);
    }
}

TEST(ExecutionScheduler, TwapTotalQtyPreserved) {
    ExecutionScheduler s(6, 0, 3'600'000);
    auto sched = s.twap(900.0, 100.0);
    ASSERT_TRUE(sched.has_value());
    double total = 0.0;
    for (auto& sl : sched->slices) total += sl.target_qty;
    EXPECT_NEAR(total, 900.0, 1e-9);
}

TEST(ExecutionScheduler, TwapSlicesMonotonicallyIncreasingTime) {
    ExecutionScheduler s(4, 1'000, 4'000);
    auto sched = s.twap(400.0, 100.0);
    ASSERT_TRUE(sched.has_value());
    for (std::size_t i = 1; i < sched->slices.size(); ++i) {
        EXPECT_LT(sched->slices[i-1].start_ms, sched->slices[i].start_ms);
    }
}

TEST(ExecutionScheduler, TwapZeroQtyReturnsNullopt) {
    ExecutionScheduler s;
    EXPECT_FALSE(s.twap(0.0, 100.0).has_value());
}

TEST(ExecutionScheduler, TwapNegativeSpotReturnsNullopt) {
    ExecutionScheduler s;
    EXPECT_FALSE(s.twap(100.0, -10.0).has_value());
}

// ============================================================
// ExecutionScheduler — vwap
// ============================================================

TEST(ExecutionScheduler, VwapWithProfileDistributesQty) {
    ExecutionScheduler s(4, 0, 4'000);
    std::vector<double> profile = {1.0, 2.0, 2.0, 1.0};
    auto sched = s.vwap(1'000.0, 100.0, profile);
    ASSERT_TRUE(sched.has_value());
    EXPECT_EQ(sched->slices.size(), 4u);
    // Higher-weight slices should have more quantity
    EXPECT_GT(sched->slices[1].target_qty, sched->slices[0].target_qty);
}

TEST(ExecutionScheduler, VwapTotalQtyPreserved) {
    ExecutionScheduler s(5, 0, 5'000);
    std::vector<double> profile = {1.0, 3.0, 2.0, 3.0, 1.0};
    auto sched = s.vwap(500.0, 50.0, profile);
    ASSERT_TRUE(sched.has_value());
    double total = 0.0;
    for (auto& sl : sched->slices) total += sl.target_qty;
    EXPECT_NEAR(total, 500.0, 1e-6);
}

TEST(ExecutionScheduler, VwapEmptyProfileFallsBackToTwap) {
    ExecutionScheduler s(4, 0, 4'000);
    auto sched = s.vwap(400.0, 100.0, {});
    ASSERT_TRUE(sched.has_value());
    // All slices equal (TWAP fallback)
    for (auto& sl : sched->slices) {
        EXPECT_NEAR(sl.target_qty, 100.0, 1e-6);
    }
}

TEST(ExecutionScheduler, VwapZeroQtyReturnsNullopt) {
    ExecutionScheduler s;
    EXPECT_FALSE(s.vwap(0.0, 100.0, {1.0, 1.0}).has_value());
}

// ============================================================
// OptimalExecution — minimize_impact
// ============================================================

TEST(OptimalExecution, Urgency1GivesTwapLabel) {
    ExecutionScheduler sched(4, 0, 4'000);
    OptimalExecution oe(std::move(sched));
    auto result = oe.minimize_impact(400.0, 1.0, {1.0, 2.0, 2.0, 1.0});
    ASSERT_TRUE(result.has_value());
    EXPECT_EQ(result->method, "TWAP");
}

TEST(OptimalExecution, Urgency0GivesVwapLabel) {
    ExecutionScheduler sched(4, 0, 4'000);
    OptimalExecution oe(std::move(sched));
    auto result = oe.minimize_impact(400.0, 0.0, {1.0, 2.0, 2.0, 1.0});
    ASSERT_TRUE(result.has_value());
    EXPECT_EQ(result->method, "VWAP");
}

TEST(OptimalExecution, MiddleUrgencyGivesBlendedLabel) {
    ExecutionScheduler sched(4, 0, 4'000);
    OptimalExecution oe(std::move(sched));
    auto result = oe.minimize_impact(400.0, 0.5, {1.0, 2.0, 2.0, 1.0});
    ASSERT_TRUE(result.has_value());
    EXPECT_EQ(result->method, "BLENDED");
}

TEST(OptimalExecution, TotalQtyPreservedAtAllUrgencies) {
    for (double u : {0.0, 0.25, 0.5, 0.75, 1.0}) {
        ExecutionScheduler sched(5, 0, 5'000);
        OptimalExecution oe(std::move(sched));
        auto result = oe.minimize_impact(1'000.0, u, {1.0, 2.0, 3.0, 2.0, 1.0});
        ASSERT_TRUE(result.has_value()) << "urgency=" << u;
        double total = 0.0;
        for (auto& sl : result->slices) total += sl.target_qty;
        EXPECT_NEAR(total, 1'000.0, 1e-6) << "urgency=" << u;
    }
}

TEST(OptimalExecution, UrgencyClampedAbove1) {
    ExecutionScheduler sched(3, 0, 3'000);
    OptimalExecution oe(std::move(sched));
    auto r = oe.minimize_impact(300.0, 1.5, {1.0, 1.0, 1.0});
    ASSERT_TRUE(r.has_value());
    EXPECT_EQ(r->method, "TWAP");
}

TEST(OptimalExecution, UrgencyClampedBelow0) {
    ExecutionScheduler sched(3, 0, 3'000);
    OptimalExecution oe(std::move(sched));
    auto r = oe.minimize_impact(300.0, -0.5, {1.0, 2.0, 1.0});
    ASSERT_TRUE(r.has_value());
    EXPECT_EQ(r->method, "VWAP");
}

TEST(OptimalExecution, ZeroQtyReturnsNullopt) {
    ExecutionScheduler sched;
    OptimalExecution oe(std::move(sched));
    EXPECT_FALSE(oe.minimize_impact(0.0, 0.5, {}).has_value());
}

TEST(OptimalExecution, NegativeSpotReturnsNullopt) {
    ExecutionScheduler sched;
    OptimalExecution oe(std::move(sched));
    EXPECT_FALSE(oe.minimize_impact(100.0, 0.5, {}, -10.0).has_value());
}

TEST(OptimalExecution, SliceTimestampsInOrder) {
    ExecutionScheduler sched(6, 0, 6'000);
    OptimalExecution oe(std::move(sched));
    auto r = oe.minimize_impact(600.0, 0.5, {1.0, 2.0, 3.0, 3.0, 2.0, 1.0});
    ASSERT_TRUE(r.has_value());
    for (std::size_t i = 1; i < r->slices.size(); ++i) {
        EXPECT_LE(r->slices[i-1].start_ms, r->slices[i].start_ms);
    }
}

#else

TEST(ExecutionTimingDisabled, FeatureFlagNotSet) {
    // Placeholder so the compilation unit is not empty when the feature is OFF.
    SUCCEED();
}

#endif // LLMQUANT_EXECUTION_TIMING_ENABLED
