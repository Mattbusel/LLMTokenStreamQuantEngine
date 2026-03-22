#include <gtest/gtest.h>
#include "SignalZeroCrossRate.h"

namespace llmquant::tests {

TEST(SignalZeroCrossRate, DefaultConstruction) {
    SignalZeroCrossRate z;
    EXPECT_EQ(z.total_records(), 0u);
    EXPECT_DOUBLE_EQ(z.zcr(), 0.0);
}

TEST(SignalZeroCrossRate, HighZCRForAlternatingSignal) {
    SignalZeroCrossRate::Config cfg;
    cfg.window        = 10;
    cfg.min_samples   = 4;
    cfg.high_threshold = 0.5;
    SignalZeroCrossRate z(cfg);
    for (int i = 0; i < 20; ++i) z.record(i % 2 == 0 ? 1.0 : -1.0);
    EXPECT_GT(z.zcr(), 0.5);
}

TEST(SignalZeroCrossRate, LowZCRForConstantSignal) {
    SignalZeroCrossRate::Config cfg;
    cfg.window      = 10;
    cfg.min_samples = 4;
    cfg.low_threshold = 0.1;
    SignalZeroCrossRate z(cfg);
    for (int i = 0; i < 20; ++i) z.record(1.0);
    EXPECT_LT(z.zcr(), 0.1);
}

TEST(SignalZeroCrossRate, HighZCRCallback) {
    SignalZeroCrossRate::Config cfg;
    cfg.window         = 10;
    cfg.min_samples    = 4;
    cfg.high_threshold = 0.3;
    int events = 0;
    cfg.on_high_zcr = [&](double) { ++events; };
    SignalZeroCrossRate z(cfg);
    for (int i = 0; i < 30; ++i) z.record(i % 2 == 0 ? 1.0 : -1.0);
    EXPECT_GT(events, 0);
}

TEST(SignalZeroCrossRate, LowZCRCallback) {
    SignalZeroCrossRate::Config cfg;
    cfg.window       = 10;
    cfg.min_samples  = 4;
    cfg.low_threshold = 0.5; // easy to be below this with constant signal
    int events = 0;
    cfg.on_low_zcr = [&](double) { ++events; };
    SignalZeroCrossRate z(cfg);
    for (int i = 0; i < 30; ++i) z.record(1.0);
    EXPECT_GT(events, 0);
}

TEST(SignalZeroCrossRate, ZCRInRange) {
    SignalZeroCrossRate z;
    for (int i = 0; i < 30; ++i) z.record(i % 3 == 0 ? 1.0 : -0.5);
    EXPECT_GE(z.zcr(), 0.0);
    EXPECT_LE(z.zcr(), 1.0);
}

TEST(SignalZeroCrossRate, Reset) {
    SignalZeroCrossRate z;
    for (int i = 0; i < 20; ++i) z.record(0.1);
    z.reset();
    EXPECT_EQ(z.total_records(), 0u);
    EXPECT_DOUBLE_EQ(z.zcr(), 0.0);
}

TEST(SignalZeroCrossRate, ToStatsJson) {
    SignalZeroCrossRate z;
    for (int i = 0; i < 15; ++i) z.record(0.05);
    auto json = z.to_stats_json();
    EXPECT_NE(json.find("zcr"), std::string::npos);
    EXPECT_NE(json.find("high_zcr_events"), std::string::npos);
}

} // namespace llmquant::tests
