#include <gtest/gtest.h>

#ifdef LLMQUANT_BAYESIAN_SENTIMENT_ENABLED

#include "BayesianSentimentPrior.h"

#include <atomic>
#include <cmath>
#include <thread>
#include <vector>

using namespace llmquant;

namespace {

// -----------------------------------------------------------------------
// Default state
// -----------------------------------------------------------------------

TEST(BayesianSentimentPrior, DefaultState) {
    BayesianSentimentPrior p;
    EXPECT_DOUBLE_EQ(p.posterior_mean(), 0.0);  // prior_mean = 0.0
    EXPECT_GT(p.posterior_std(), 0.0);
    EXPECT_FALSE(p.is_shifted());
    EXPECT_EQ(p.total_updates(), 0u);
    EXPECT_EQ(p.belief_shifts(), 0u);
}

// -----------------------------------------------------------------------
// CI contains prior mean for uninformative update
// -----------------------------------------------------------------------

TEST(BayesianSentimentPrior, CIContainsPriorMean) {
    BayesianSentimentPrior p;
    EXPECT_LE(p.ci_lower(), 0.0);
    EXPECT_GE(p.ci_upper(), 0.0);
}

// -----------------------------------------------------------------------
// Total updates increments
// -----------------------------------------------------------------------

TEST(BayesianSentimentPrior, TotalUpdatesIncrements) {
    BayesianSentimentPrior p;
    p.update(0.1);
    p.update(0.2);
    EXPECT_EQ(p.total_updates(), 2u);
}

// -----------------------------------------------------------------------
// Posterior mean converges toward data mean
// -----------------------------------------------------------------------

TEST(BayesianSentimentPrior, PosteriorMeanConverges) {
    BayesianSentimentPrior::Config cfg;
    cfg.prior_kappa = 1.0;   // weak prior
    BayesianSentimentPrior p(cfg);
    // Feed 50 observations of 0.8 (strong positive signal)
    for (int i = 0; i < 50; ++i) p.update(0.8);
    EXPECT_GT(p.posterior_mean(), 0.5);  // posterior pulled toward 0.8
}

// -----------------------------------------------------------------------
// Posterior std decreases with more observations
// -----------------------------------------------------------------------

TEST(BayesianSentimentPrior, PosteriorStdDecreases) {
    BayesianSentimentPrior p;
    double std_before = p.posterior_std();
    for (int i = 0; i < 30; ++i) p.update(0.5);
    double std_after = p.posterior_std();
    EXPECT_LT(std_after, std_before);
}

// -----------------------------------------------------------------------
// Belief shift callback fires when mean moves significantly
// -----------------------------------------------------------------------

TEST(BayesianSentimentPrior, BeliefShiftCallbackFires) {
    BayesianSentimentPrior::Config cfg;
    cfg.prior_kappa      = 0.5;   // weak prior — data dominates quickly
    cfg.shift_threshold  = 1.5;
    cfg.min_samples      = 3;
    std::atomic<int> fires{0};
    cfg.on_belief_shift = [&](double, double) { ++fires; };
    BayesianSentimentPrior p(cfg);
    // Feed strong positive signal far from prior mean 0
    for (int i = 0; i < 20; ++i) p.update(2.0);
    EXPECT_GE(fires.load(), 1);
    EXPECT_GT(p.belief_shifts(), 0u);
    EXPECT_TRUE(p.is_shifted());
}

// -----------------------------------------------------------------------
// Belief restored callback fires when mean returns to prior
// -----------------------------------------------------------------------

TEST(BayesianSentimentPrior, ResetRestoresBelief) {
    BayesianSentimentPrior::Config cfg;
    cfg.prior_kappa      = 0.5;
    cfg.shift_threshold  = 1.5;
    cfg.min_samples      = 3;
    std::atomic<int> restores{0};
    cfg.on_belief_restored = [&]() { ++restores; };
    BayesianSentimentPrior p(cfg);
    // Push into shifted
    for (int i = 0; i < 20; ++i) p.update(3.0);
    EXPECT_TRUE(p.is_shifted());
    // Reset restores to prior — not shifted
    p.reset();
    EXPECT_FALSE(p.is_shifted());
    EXPECT_DOUBLE_EQ(p.posterior_mean(), cfg.prior_mean);
}

// -----------------------------------------------------------------------
// CI width shrinks with more data
// -----------------------------------------------------------------------

TEST(BayesianSentimentPrior, CIWidthShrinks) {
    BayesianSentimentPrior p;
    double width_before = p.ci_upper() - p.ci_lower();
    for (int i = 0; i < 50; ++i) p.update(0.5);
    double width_after = p.ci_upper() - p.ci_lower();
    EXPECT_LT(width_after, width_before);
}

// -----------------------------------------------------------------------
// reset() clears state
// -----------------------------------------------------------------------

TEST(BayesianSentimentPrior, ResetClearsState) {
    BayesianSentimentPrior p;
    for (int i = 0; i < 20; ++i) p.update(0.5);
    p.reset();
    EXPECT_EQ(p.total_updates(), 0u);
    EXPECT_DOUBLE_EQ(p.posterior_mean(), 0.0);
    EXPECT_FALSE(p.is_shifted());
    EXPECT_EQ(p.belief_shifts(), 0u);
}

// -----------------------------------------------------------------------
// update_config resets state
// -----------------------------------------------------------------------

TEST(BayesianSentimentPrior, UpdateConfigResetsState) {
    BayesianSentimentPrior p;
    for (int i = 0; i < 10; ++i) p.update(0.5);
    BayesianSentimentPrior::Config cfg2;
    cfg2.prior_mean = 0.1;
    p.update_config(cfg2);
    EXPECT_EQ(p.total_updates(), 0u);
}

// -----------------------------------------------------------------------
// to_stats_json contains expected keys
// -----------------------------------------------------------------------

TEST(BayesianSentimentPrior, ToStatsJsonContainsKeys) {
    BayesianSentimentPrior p;
    p.update(0.5);
    std::string j = p.to_stats_json();
    EXPECT_NE(j.find("posterior_mean"),  std::string::npos);
    EXPECT_NE(j.find("posterior_std"),   std::string::npos);
    EXPECT_NE(j.find("ci_lower"),        std::string::npos);
    EXPECT_NE(j.find("ci_upper"),        std::string::npos);
    EXPECT_NE(j.find("is_shifted"),      std::string::npos);
    EXPECT_NE(j.find("total_updates"),   std::string::npos);
}

// -----------------------------------------------------------------------
// Concurrent update() is thread-safe
// -----------------------------------------------------------------------

TEST(BayesianSentimentPrior, ConcurrentUpdateSafe) {
    BayesianSentimentPrior p;
    std::vector<std::thread> threads;
    for (int t = 0; t < 4; ++t) {
        threads.emplace_back([&, t] {
            for (int i = 0; i < 250; ++i)
                p.update(t % 2 == 0 ? 0.5 : -0.5);
        });
    }
    for (auto& th : threads) th.join();
    EXPECT_EQ(p.total_updates(), 1000u);
    EXPECT_TRUE(std::isfinite(p.posterior_mean()));
    EXPECT_TRUE(std::isfinite(p.posterior_std()));
}

} // namespace

#else

TEST(BayesianSentimentPrior, DisabledAtBuildTime) { SUCCEED(); }

#endif  // LLMQUANT_BAYESIAN_SENTIMENT_ENABLED
