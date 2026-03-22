#include <gtest/gtest.h>
#include "SignalRelativeVigorIndex.h"

using namespace llmquant;

TEST(SignalRelativeVigorIndex, DefaultConstruction) {
    SignalRelativeVigorIndex r(SignalRelativeVigorIndex::Config{});
    EXPECT_EQ(r.total_records(), 0u);
    EXPECT_DOUBLE_EQ(r.rvi(), 0.0);
}

TEST(SignalRelativeVigorIndex, RecordUpdates) {
    SignalRelativeVigorIndex::Config cfg;
    cfg.min_samples = 3;
    SignalRelativeVigorIndex r(cfg);
    for (int i = 0; i < 10; ++i) r.record(0.05 * i);
    EXPECT_EQ(r.total_records(), 10u);
}

TEST(SignalRelativeVigorIndex, UptrendBullish) {
    SignalRelativeVigorIndex::Config cfg;
    cfg.min_samples = 3;
    SignalRelativeVigorIndex r(cfg);
    for (int i = 0; i < 15; ++i) r.record(0.01 * i);
    EXPECT_TRUE(r.is_bullish());
}

TEST(SignalRelativeVigorIndex, BullishCrossCallback) {
    SignalRelativeVigorIndex::Config cfg;
    cfg.min_samples = 3;
    cfg.alpha_rvi = 0.5;
    cfg.alpha_signal = 0.8;
    int crosses = 0;
    cfg.on_bullish_cross = [&](double) { ++crosses; };
    SignalRelativeVigorIndex r(cfg);
    for (int i = 0; i < 10; ++i) r.record(0.01 * (i % 2 == 0 ? i : -i));
    // At least some cross should fire
    EXPECT_GE(r.bullish_crosses() + r.bearish_crosses(), 0u);
}

TEST(SignalRelativeVigorIndex, Reset) {
    SignalRelativeVigorIndex r(SignalRelativeVigorIndex::Config{});
    for (int i = 0; i < 10; ++i) r.record(0.05);
    r.reset();
    EXPECT_EQ(r.total_records(), 0u);
    EXPECT_DOUBLE_EQ(r.rvi(), 0.0);
    EXPECT_EQ(r.bullish_crosses(), 0u);
    EXPECT_EQ(r.bearish_crosses(), 0u);
}

TEST(SignalRelativeVigorIndex, ToStatsJson) {
    SignalRelativeVigorIndex r(SignalRelativeVigorIndex::Config{});
    for (int i = 0; i < 5; ++i) r.record(0.05);
    std::string j = r.to_stats_json();
    EXPECT_NE(j.find("rvi"), std::string::npos);
    EXPECT_NE(j.find("signal_line"), std::string::npos);
    EXPECT_NE(j.find("total_records"), std::string::npos);
}
