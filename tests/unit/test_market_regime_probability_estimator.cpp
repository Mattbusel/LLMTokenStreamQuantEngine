#include <gtest/gtest.h>

#ifdef LLMQUANT_REGIME_PROB_ENABLED

#include "MarketRegimeProbabilityEstimator.h"

#include <atomic>
#include <cmath>
#include <thread>
#include <vector>

using namespace llmquant;

namespace {

// ============================================================
// Default construction
// ============================================================

TEST(MarketRegimeProbabilityEstimator, DefaultConstructs) {
    MarketRegimeProbabilityEstimator est;
    EXPECT_NEAR(est.prob_risk_on(),  0.5, 1e-9);
    EXPECT_NEAR(est.prob_risk_off(), 0.5, 1e-9);
    EXPECT_EQ(est.observation_count(), 0u);
    EXPECT_EQ(est.transition_count(),  0u);
}

// ============================================================
// Probabilities sum to 1.0
// ============================================================

TEST(MarketRegimeProbabilityEstimator, ProbsSumToOne) {
    MarketRegimeProbabilityEstimator est;
    for (int i = 0; i < 30; ++i)
        est.update(0.5, 0.1);
    EXPECT_NEAR(est.prob_risk_on() + est.prob_risk_off(), 1.0, 1e-9);
}

// ============================================================
// Bearish sentiment + high vol pushes toward risk-OFF
// ============================================================

TEST(MarketRegimeProbabilityEstimator, BearishObservationsPushRiskOff) {
    MarketRegimeProbabilityEstimator::Config cfg;
    cfg.emission_alpha = 0.3;
    MarketRegimeProbabilityEstimator est(cfg);

    for (int i = 0; i < 100; ++i)
        est.update(-0.8, 0.8);  // very bearish, very volatile

    EXPECT_GT(est.prob_risk_off(), 0.5);
}

// ============================================================
// Bullish sentiment + low vol pushes toward risk-ON
// ============================================================

TEST(MarketRegimeProbabilityEstimator, BullishObservationsPushRiskOn) {
    MarketRegimeProbabilityEstimator::Config cfg;
    cfg.emission_alpha = 0.3;
    MarketRegimeProbabilityEstimator est(cfg);

    for (int i = 0; i < 100; ++i)
        est.update(0.9, 0.02);  // very bullish, very calm

    EXPECT_GT(est.prob_risk_on(), 0.5);
}

// ============================================================
// observation_count increments
// ============================================================

TEST(MarketRegimeProbabilityEstimator, ObservationCountIncrements) {
    MarketRegimeProbabilityEstimator est;
    est.update(0.1, 0.2);
    est.update(0.2, 0.1);
    EXPECT_EQ(est.observation_count(), 2u);
}

// ============================================================
// Regime change callback fires on transition
// ============================================================

TEST(MarketRegimeProbabilityEstimator, CallbackFiresOnTransition) {
    MarketRegimeProbabilityEstimator::Config cfg;
    cfg.emission_alpha      = 0.5;
    cfg.transition_threshold = 0.6;
    cfg.min_observations    = 5;
    std::atomic<int> fires{0};
    cfg.on_regime_change = [&](double, bool) { ++fires; };
    MarketRegimeProbabilityEstimator est(cfg);

    // Push strongly bullish to force risk-ON.
    for (int i = 0; i < 40; ++i)
        est.update(0.9, 0.01);

    // Then push strongly bearish to force risk-OFF.
    for (int i = 0; i < 40; ++i)
        est.update(-0.9, 0.9);

    EXPECT_GT(fires.load(), 0);
}

// ============================================================
// reset() returns to uniform prior
// ============================================================

TEST(MarketRegimeProbabilityEstimator, ResetReturnsToUniform) {
    MarketRegimeProbabilityEstimator est;
    for (int i = 0; i < 50; ++i) est.update(0.9, 0.01);
    est.reset();
    EXPECT_NEAR(est.prob_risk_on(),  0.5, 1e-9);
    EXPECT_NEAR(est.prob_risk_off(), 0.5, 1e-9);
    EXPECT_EQ(est.observation_count(), 0u);
}

// ============================================================
// to_stats_json contains expected keys
// ============================================================

TEST(MarketRegimeProbabilityEstimator, ToStatsJsonContainsKeys) {
    MarketRegimeProbabilityEstimator est;
    est.update(0.3, 0.1);
    std::string json = est.to_stats_json();
    EXPECT_NE(json.find("prob_risk_on"),   std::string::npos);
    EXPECT_NE(json.find("prob_risk_off"),  std::string::npos);
    EXPECT_NE(json.find("observations"),   std::string::npos);
    EXPECT_NE(json.find("transitions"),    std::string::npos);
}

// ============================================================
// Concurrent update() is thread-safe
// ============================================================

TEST(MarketRegimeProbabilityEstimator, ConcurrentUpdateSafe) {
    MarketRegimeProbabilityEstimator::Config cfg;
    cfg.emission_alpha = 0.1;
    MarketRegimeProbabilityEstimator est(cfg);

    std::vector<std::thread> threads;
    for (int t = 0; t < 4; ++t) {
        threads.emplace_back([&, t] {
            for (int i = 0; i < 50; ++i)
                est.update(static_cast<double>(i % 2 == 0 ? 1 : -1) * 0.3,
                           0.1 + static_cast<double>(t) * 0.05);
        });
    }
    for (auto& th : threads) th.join();

    EXPECT_EQ(est.observation_count(), 200u);
    double pon = est.prob_risk_on();
    EXPECT_GE(pon, 0.0);
    EXPECT_LE(pon, 1.0);
    EXPECT_NEAR(pon + est.prob_risk_off(), 1.0, 1e-9);
}

} // namespace

#else

TEST(MarketRegimeProbabilityEstimator, DisabledAtBuildTime) { SUCCEED(); }

#endif  // LLMQUANT_REGIME_PROB_ENABLED
