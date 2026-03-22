#include <gtest/gtest.h>
#include "NarrativeRegimeMarkov.h"

using namespace llmquant;

TEST(NarrativeRegimeMarkov, DefaultConstruction) {
    NarrativeRegimeMarkov m(NarrativeRegimeMarkov::Config{});
    EXPECT_EQ(m.total_records(), 0u);
    EXPECT_GE(m.current_state(), 0);
}

TEST(NarrativeRegimeMarkov, StateAssignment) {
    NarrativeRegimeMarkov::Config cfg;
    cfg.n_regimes = 3;
    cfg.bias_range = 0.1;
    NarrativeRegimeMarkov m(cfg);
    m.record(0.0);
    int s = m.current_state();
    EXPECT_GE(s, 0);
    EXPECT_LT(s, 3);
}

TEST(NarrativeRegimeMarkov, SteadyStateSumsToOne) {
    NarrativeRegimeMarkov::Config cfg;
    cfg.n_regimes = 3;
    cfg.min_samples = 2;
    NarrativeRegimeMarkov m(cfg);
    for (int i = 0; i < 30; ++i) m.record(i % 3 == 0 ? -0.15 : (i % 3 == 1 ? 0.0 : 0.15));
    double sum = 0.0;
    for (int s = 0; s < 3; ++s) sum += m.steady_state(s);
    EXPECT_NEAR(sum, 1.0, 1e-6);
}

TEST(NarrativeRegimeMarkov, TransitionInRange) {
    NarrativeRegimeMarkov::Config cfg;
    cfg.n_regimes = 3;
    NarrativeRegimeMarkov m(cfg);
    for (int i = 0; i < 20; ++i) m.record(i % 2 == 0 ? -0.15 : 0.15);
    double t = m.transition(0, 2);
    EXPECT_GE(t, 0.0);
    EXPECT_LE(t, 1.0);
}

TEST(NarrativeRegimeMarkov, RegimeChangeCallback) {
    NarrativeRegimeMarkov::Config cfg;
    cfg.n_regimes = 3;
    cfg.bias_range = 0.1;
    cfg.min_samples = 2;
    int changes = 0;
    cfg.on_regime_change = [&](int, int) { ++changes; };
    NarrativeRegimeMarkov m(cfg);
    for (int i = 0; i < 20; ++i) m.record(i % 2 == 0 ? -0.2 : 0.2);
    EXPECT_GE(changes, 1);
}

TEST(NarrativeRegimeMarkov, Reset) {
    NarrativeRegimeMarkov::Config cfg;
    NarrativeRegimeMarkov m(cfg);
    for (int i = 0; i < 10; ++i) m.record(0.1);
    m.reset();
    EXPECT_EQ(m.total_records(), 0u);
    EXPECT_EQ(m.regime_change_events(), 0u);
}

TEST(NarrativeRegimeMarkov, ToStatsJson) {
    NarrativeRegimeMarkov m(NarrativeRegimeMarkov::Config{});
    for (int i = 0; i < 5; ++i) m.record(0.05);
    std::string j = m.to_stats_json();
    EXPECT_NE(j.find("current_state"), std::string::npos);
    EXPECT_NE(j.find("total_records"), std::string::npos);
}

TEST(NarrativeRegimeMarkov, OutOfBoundsSteadyState) {
    NarrativeRegimeMarkov m(NarrativeRegimeMarkov::Config{});
    EXPECT_DOUBLE_EQ(m.steady_state(-1), 0.0);
    EXPECT_DOUBLE_EQ(m.steady_state(999), 0.0);
}
