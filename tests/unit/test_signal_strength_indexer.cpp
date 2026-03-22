#include <gtest/gtest.h>

#ifdef LLMQUANT_SIGNAL_SSI_ENABLED
#include "SignalStrengthIndexer.h"
using namespace llmquant;

TEST(SignalStrengthIndexerTest, InitialSSI) {
    SignalStrengthIndexer ssi;
    // Before sufficient warmup, SSI defaults to 50.
    EXPECT_GE(ssi.ssi(), 0.0);
    EXPECT_LE(ssi.ssi(), 100.0);
}

TEST(SignalStrengthIndexerTest, ObservationCountIncrement) {
    SignalStrengthIndexer ssi;
    ssi.record(0.1);
    ssi.record(-0.1);
    EXPECT_EQ(ssi.observation_count(), 2u);
}

TEST(SignalStrengthIndexerTest, OverboughtAfterConsistentGains) {
    SignalStrengthIndexer::Config cfg;
    cfg.period               = 5;
    cfg.overbought_threshold = 70.0;
    cfg.oversold_threshold   = 30.0;
    int ob_count = 0;
    cfg.on_overbought = [&](double) { ++ob_count; };
    SignalStrengthIndexer ssi(cfg);
    // Feed consistently rising bias — all gains, no losses → RS → ∞ → SSI → 100.
    for (int i = 0; i < 30; ++i)
        ssi.record(static_cast<double>(i) * 0.1);
    EXPECT_GE(ob_count, 1);
    EXPECT_TRUE(ssi.is_overbought());
}

TEST(SignalStrengthIndexerTest, OversoldAfterConsistentLosses) {
    SignalStrengthIndexer::Config cfg;
    cfg.period               = 5;
    cfg.overbought_threshold = 70.0;
    cfg.oversold_threshold   = 30.0;
    int os_count = 0;
    cfg.on_oversold = [&](double) { ++os_count; };
    SignalStrengthIndexer ssi(cfg);
    for (int i = 0; i < 30; ++i)
        ssi.record(-static_cast<double>(i) * 0.1);
    EXPECT_GE(os_count, 1);
    EXPECT_TRUE(ssi.is_oversold());
}

TEST(SignalStrengthIndexerTest, SSIRangeValid) {
    SignalStrengthIndexer ssi;
    for (int i = 0; i < 50; ++i)
        ssi.record(i % 3 == 0 ? 0.5 : -0.3);
    EXPECT_GE(ssi.ssi(), 0.0);
    EXPECT_LE(ssi.ssi(), 100.0);
}

TEST(SignalStrengthIndexerTest, Reset) {
    SignalStrengthIndexer ssi;
    for (int i = 0; i < 20; ++i) ssi.record(0.5);
    ssi.reset();
    EXPECT_EQ(ssi.observation_count(), 0u);
}

TEST(SignalStrengthIndexerTest, StatsJson) {
    SignalStrengthIndexer ssi;
    ssi.record(0.1);
    std::string json = ssi.to_stats_json();
    EXPECT_NE(json.find("ssi"), std::string::npos);
}

#else
TEST(SignalStrengthIndexerTest, Disabled) { SUCCEED(); }
#endif
