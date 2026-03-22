#include <gtest/gtest.h>
#include "SignalBollingerBands.h"
#include <cmath>

namespace llmquant::tests {

TEST(SignalBollingerBands, DefaultConstruction) {
    SignalBollingerBands bb;
    EXPECT_EQ(bb.total_records(), 0u);
    EXPECT_DOUBLE_EQ(bb.middle(), 0.0);
}

TEST(SignalBollingerBands, BandsSymmetricAroundMean) {
    SignalBollingerBands bb;
    for (int i = 0; i < 30; ++i) bb.record(0.0);
    // All zero → bands at zero
    EXPECT_NEAR(bb.middle(), 0.0, 1e-9);
    EXPECT_NEAR(bb.upper(),  0.0, 1e-6);
    EXPECT_NEAR(bb.lower(),  0.0, 1e-6);
}

TEST(SignalBollingerBands, UpperAboveLower) {
    SignalBollingerBands bb;
    for (int i = 0; i < 30; ++i) bb.record((double)(i % 5 - 2) * 0.1);
    EXPECT_GE(bb.upper(), bb.lower());
}

TEST(SignalBollingerBands, UpperTouchCallback) {
    SignalBollingerBands::Config cfg;
    cfg.window = 5; cfg.k = 0.5; cfg.min_samples = 3;
    int touches = 0;
    cfg.on_upper_touch = [&](double) { ++touches; };
    SignalBollingerBands bb(cfg);
    for (int i = 0; i < 5; ++i) bb.record(0.0);
    bb.record(10.0);  // large spike > upper band
    EXPECT_GT(touches, 0);
}

TEST(SignalBollingerBands, LowerTouchCallback) {
    SignalBollingerBands::Config cfg;
    cfg.window = 5; cfg.k = 0.5; cfg.min_samples = 3;
    int touches = 0;
    cfg.on_lower_touch = [&](double) { ++touches; };
    SignalBollingerBands bb(cfg);
    for (int i = 0; i < 5; ++i) bb.record(0.0);
    bb.record(-10.0); // large negative spike < lower band
    EXPECT_GT(touches, 0);
}

TEST(SignalBollingerBands, PctBBetweenZeroAndOne_Approx) {
    SignalBollingerBands bb;
    for (int i = 0; i < 30; ++i) bb.record((double)(i % 3 - 1) * 0.1);
    double pb = bb.pct_b();
    EXPECT_GE(pb, 0.0);
    EXPECT_LE(pb, 1.0);
}

TEST(SignalBollingerBands, SqueezeCallback) {
    SignalBollingerBands::Config cfg;
    cfg.squeeze_threshold = 10.0; // very wide → always squeeze
    cfg.min_samples = 3;
    int squeezes = 0;
    cfg.on_squeeze = [&](double) { ++squeezes; };
    SignalBollingerBands bb(cfg);
    for (int i = 0; i < 10; ++i) bb.record(0.01);
    EXPECT_GT(squeezes, 0);
}

TEST(SignalBollingerBands, Reset) {
    SignalBollingerBands bb;
    for (int i = 0; i < 20; ++i) bb.record(0.1);
    bb.reset();
    EXPECT_EQ(bb.total_records(), 0u);
    EXPECT_DOUBLE_EQ(bb.middle(), 0.0);
}

TEST(SignalBollingerBands, ToStatsJson) {
    SignalBollingerBands bb;
    for (int i = 0; i < 20; ++i) bb.record(0.05);
    auto json = bb.to_stats_json();
    EXPECT_NE(json.find("middle"), std::string::npos);
    EXPECT_NE(json.find("pct_b"), std::string::npos);
}

} // namespace llmquant::tests
