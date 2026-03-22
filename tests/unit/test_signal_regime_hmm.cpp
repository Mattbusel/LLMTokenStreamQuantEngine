#include <gtest/gtest.h>
#include "SignalRegimeHMM.h"
#include <cmath>

namespace llmquant::tests {

TEST(SignalRegimeHMM, DefaultConstruction) {
    SignalRegimeHMM hmm;
    EXPECT_GE(hmm.p_bullish(), 0.0);
    EXPECT_LE(hmm.p_bullish(), 1.0);
    EXPECT_EQ(hmm.total_records(), 0u);
}

TEST(SignalRegimeHMM, PositiveBiasIncreasesP_Bullish) {
    SignalRegimeHMM hmm;
    for (int i = 0; i < 30; ++i) hmm.record(0.5);
    EXPECT_GT(hmm.p_bullish(), 0.5);
}

TEST(SignalRegimeHMM, NegativeBiasDecreasesP_Bullish) {
    SignalRegimeHMM hmm;
    for (int i = 0; i < 30; ++i) hmm.record(-0.5);
    EXPECT_LT(hmm.p_bullish(), 0.5);
}

TEST(SignalRegimeHMM, ProbabilitySumsToOne) {
    SignalRegimeHMM hmm;
    for (int i = 0; i < 10; ++i) hmm.record(0.1);
    EXPECT_NEAR(hmm.p_bullish() + hmm.p_bearish(), 1.0, 1e-10);
}

TEST(SignalRegimeHMM, RegimeChangeCallback) {
    SignalRegimeHMM::Config cfg;
    cfg.min_samples = 5;
    int changes = 0;
    cfg.on_regime_change = [&](int, int) { ++changes; };
    SignalRegimeHMM hmm(cfg);
    for (int i = 0; i < 10; ++i) hmm.record(0.8);   // bullish
    for (int i = 0; i < 30; ++i) hmm.record(-0.8);  // bearish
    // At least one regime change should be detected
    EXPECT_GE(hmm.regime_changes(), 1u);
}

TEST(SignalRegimeHMM, ToStatsJson) {
    SignalRegimeHMM hmm;
    for (int i = 0; i < 5; ++i) hmm.record(0.2);
    auto json = hmm.to_stats_json();
    EXPECT_NE(json.find("p_bullish"), std::string::npos);
    EXPECT_NE(json.find("regime_changes"), std::string::npos);
}

TEST(SignalRegimeHMM, Reset) {
    SignalRegimeHMM hmm;
    for (int i = 0; i < 20; ++i) hmm.record(0.3);
    hmm.reset();
    EXPECT_EQ(hmm.total_records(), 0u);
    EXPECT_NEAR(hmm.p_bullish(), 0.5, 0.01);
}

TEST(SignalRegimeHMM, StateIsZeroOrOne) {
    SignalRegimeHMM hmm;
    for (int i = 0; i < 20; ++i) hmm.record(0.2);
    int s = hmm.current_state();
    EXPECT_TRUE(s == 0 || s == 1);
}

} // namespace llmquant::tests
