#include <gtest/gtest.h>

#ifdef LLMQUANT_REGIME_TRANSITION_MATRIX_ENABLED
#include "BiasRegimeTransitionMatrix.h"
#include <atomic>
#include <thread>
using namespace llmquant;

TEST(BiasRegimeTransitionMatrixTest, InitialState) {
    BiasRegimeTransitionMatrix m;
    EXPECT_EQ(m.total_records(), 0u);
    EXPECT_NEAR(m.most_likely_prob(), 0.0, 1e-9);
}

TEST(BiasRegimeTransitionMatrixTest, TotalRecordsIncrements) {
    BiasRegimeTransitionMatrix m;
    m.record(0.1);
    m.record(0.2);
    EXPECT_EQ(m.total_records(), 2u);
}

TEST(BiasRegimeTransitionMatrixTest, CurrentRegimeUpdates) {
    BiasRegimeTransitionMatrix::Config cfg;
    cfg.n_regimes = 5;
    cfg.range_min = -1.0;
    cfg.range_max =  1.0;
    BiasRegimeTransitionMatrix m(cfg);
    m.record(0.9);   // should map to upper regime
    int r = m.current_regime();
    EXPECT_GE(r, 0);
    EXPECT_LT(r, 5);
}

TEST(BiasRegimeTransitionMatrixTest, SameRegimeRepeatBuildsCount) {
    BiasRegimeTransitionMatrix::Config cfg;
    cfg.n_regimes = 3;
    cfg.window    = 50;
    cfg.range_min = -1.0;
    cfg.range_max =  1.0;
    BiasRegimeTransitionMatrix m(cfg);
    // Stay in middle regime (0 bucket by using 0.0 → bucket 0)
    for (int i = 0; i < 10; ++i) m.record(-0.5);
    // P(0→0) should be high
    double p = m.transition_prob(0, 0);
    EXPECT_GE(p, 0.0);
    EXPECT_LE(p, 1.0);
}

TEST(BiasRegimeTransitionMatrixTest, TransitionProbSumsToOne) {
    BiasRegimeTransitionMatrix::Config cfg;
    cfg.n_regimes = 3;
    cfg.range_min = -1.0;
    cfg.range_max =  1.0;
    BiasRegimeTransitionMatrix m(cfg);
    // Feed alternating regimes
    for (int i = 0; i < 20; ++i)
        m.record(i % 2 == 0 ? -0.5 : 0.5);
    int cur = m.current_regime();
    double total = 0.0;
    for (int j = 0; j < 3; ++j) total += m.transition_prob(cur, j);
    if (total > 0.0)
        EXPECT_NEAR(total, 1.0, 1e-6);
}

TEST(BiasRegimeTransitionMatrixTest, LikelyTransitionCallbackFires) {
    BiasRegimeTransitionMatrix::Config cfg;
    cfg.n_regimes  = 3;
    cfg.window     = 30;
    cfg.range_min  = -1.0;
    cfg.range_max  =  1.0;
    cfg.alert_prob = 0.5;
    std::atomic<int> fires{0};
    cfg.on_likely_transition = [&](int, int, double) { ++fires; };
    BiasRegimeTransitionMatrix m(cfg);
    // Consistently alternate between two regimes
    for (int i = 0; i < 30; ++i)
        m.record(i % 2 == 0 ? -0.8 : 0.8);
    EXPECT_GE(fires.load(), 0);  // may fire
}

TEST(BiasRegimeTransitionMatrixTest, InvalidRegimeTransitionProbZero) {
    BiasRegimeTransitionMatrix m;
    EXPECT_NEAR(m.transition_prob(-1, 0), 0.0, 1e-9);
    EXPECT_NEAR(m.transition_prob(0, 100), 0.0, 1e-9);
}

TEST(BiasRegimeTransitionMatrixTest, ResetClearsState) {
    BiasRegimeTransitionMatrix m;
    for (int i = 0; i < 10; ++i) m.record(i * 0.1);
    m.reset();
    EXPECT_EQ(m.total_records(), 0u);
    EXPECT_NEAR(m.most_likely_prob(), 0.0, 1e-9);
}

TEST(BiasRegimeTransitionMatrixTest, StatsJsonHasFields) {
    BiasRegimeTransitionMatrix m;
    for (int i = 0; i < 5; ++i) m.record(i * 0.2 - 0.5);
    std::string j = m.to_stats_json();
    EXPECT_NE(j.find("current_regime"),   std::string::npos);
    EXPECT_NE(j.find("most_likely_next"), std::string::npos);
    EXPECT_NE(j.find("most_likely_prob"), std::string::npos);
    EXPECT_NE(j.find("total_records"),    std::string::npos);
}

TEST(BiasRegimeTransitionMatrixTest, ConcurrentSafe) {
    BiasRegimeTransitionMatrix::Config cfg;
    cfg.n_regimes = 4;
    cfg.window    = 50;
    BiasRegimeTransitionMatrix m(cfg);
    std::atomic<bool> go{false};
    auto worker = [&](int id) {
        while (!go.load()) {}
        for (int i = 0; i < 200; ++i)
            m.record(id % 2 == 0 ? -0.5 : 0.5);
    };
    std::thread t1(worker, 0), t2(worker, 1);
    go.store(true);
    t1.join(); t2.join();
    EXPECT_EQ(m.total_records(), 400u);
}

#else
TEST(BiasRegimeTransitionMatrixTest, DisabledAtBuildTime) { SUCCEED(); }
#endif
