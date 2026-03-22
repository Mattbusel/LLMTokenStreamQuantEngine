#include <gtest/gtest.h>

#ifdef LLMQUANT_ENTROPY_METER_ENABLED

#include "TokenEntropyMeter.h"

#include <cmath>

using namespace llmquant;

namespace {

// ============================================================
// Construction
// ============================================================

TEST(TokenEntropyMeter, DefaultConstructsWithZeroEntropy) {
    TokenEntropyMeter meter;
    EXPECT_NEAR(meter.get_normalized_entropy(), 0.0, 1e-9);
    EXPECT_NEAR(meter.get_raw_entropy(),         0.0, 1e-9);
    EXPECT_EQ(meter.total_samples(), 0u);
}

// ============================================================
// Single bin — maximum conviction
// ============================================================

TEST(TokenEntropyMeter, UniformInputGivesHighEntropy) {
    TokenEntropyMeter::Config cfg;
    cfg.window_size = 32;
    TokenEntropyMeter meter(cfg);

    // Spread samples evenly across the full range.
    for (int i = 0; i < 32; ++i) {
        double v = -1.0 + 2.0 * static_cast<double>(i) / 31.0;
        meter.add(v);
    }
    // Normalized entropy should be close to 1.0.
    EXPECT_GT(meter.get_normalized_entropy(), 0.8);
}

TEST(TokenEntropyMeter, AllSameValueGivesLowEntropy) {
    TokenEntropyMeter::Config cfg;
    cfg.window_size = 32;
    TokenEntropyMeter meter(cfg);

    for (int i = 0; i < 32; ++i)
        meter.add(0.9);  // all in the same positive bin

    EXPECT_LT(meter.get_normalized_entropy(), 0.25);
}

// ============================================================
// Entropy is normalised in [0, 1]
// ============================================================

TEST(TokenEntropyMeter, NormalizedEntropyIsBounded) {
    TokenEntropyMeter meter;
    for (int i = 0; i < 100; ++i)
        meter.add(-1.0 + 2.0 * (i % 17) / 16.0);

    double ne = meter.get_normalized_entropy();
    EXPECT_GE(ne, 0.0);
    EXPECT_LE(ne, 1.0 + 1e-9);
}

// ============================================================
// Sliding window evicts old samples
// ============================================================

TEST(TokenEntropyMeter, WindowEvictsOldSamples) {
    TokenEntropyMeter::Config cfg;
    cfg.window_size = 8;
    TokenEntropyMeter meter(cfg);

    // Fill window with scattered values → high entropy.
    for (int i = 0; i < 8; ++i)
        meter.add(-1.0 + 2.0 * static_cast<double>(i) / 7.0);
    double high_ent = meter.get_normalized_entropy();

    // Overwrite window with all-same values → low entropy.
    for (int i = 0; i < 8; ++i)
        meter.add(0.95);
    double low_ent = meter.get_normalized_entropy();

    EXPECT_GT(high_ent, low_ent);
}

// ============================================================
// window_count
// ============================================================

TEST(TokenEntropyMeter, WindowCountCapsAtWindowSize) {
    TokenEntropyMeter::Config cfg;
    cfg.window_size = 10;
    TokenEntropyMeter meter(cfg);

    for (int i = 0; i < 25; ++i)
        meter.add(static_cast<double>(i % 5) * 0.4 - 1.0);

    EXPECT_EQ(meter.window_count(), 10u);
    EXPECT_EQ(meter.total_samples(), 25u);
}

// ============================================================
// reset
// ============================================================

TEST(TokenEntropyMeter, ResetClearsState) {
    TokenEntropyMeter meter;
    for (int i = 0; i < 20; ++i) meter.add(0.5);
    meter.reset();
    EXPECT_NEAR(meter.get_normalized_entropy(), 0.0, 1e-9);
    EXPECT_EQ(meter.window_count(), 0u);
}

// ============================================================
// update_config
// ============================================================

TEST(TokenEntropyMeter, UpdateConfigResizesWindow) {
    TokenEntropyMeter meter;
    for (int i = 0; i < 32; ++i) meter.add(0.5);
    TokenEntropyMeter::Config cfg;
    cfg.window_size = 4;
    meter.update_config(cfg);
    EXPECT_EQ(meter.window_count(), 0u);  // reset on resize
}

// ============================================================
// Clamping out-of-range values
// ============================================================

TEST(TokenEntropyMeter, OutOfRangeValuesAreClamped) {
    TokenEntropyMeter meter;
    // Should not crash or produce NaN.
    meter.add(-999.0);
    meter.add(+999.0);
    double ne = meter.get_normalized_entropy();
    EXPECT_FALSE(std::isnan(ne));
    EXPECT_GE(ne, 0.0);
}

// ============================================================
// to_stats_json
// ============================================================

TEST(TokenEntropyMeter, ToStatsJsonContainsKeys) {
    TokenEntropyMeter meter;
    meter.add(0.3);
    std::string json = meter.to_stats_json();
    EXPECT_NE(json.find("normalized_entropy"), std::string::npos);
    EXPECT_NE(json.find("raw_entropy"),        std::string::npos);
    EXPECT_NE(json.find("window_count"),       std::string::npos);
    EXPECT_NE(json.find("total_samples"),      std::string::npos);
}

// ============================================================
// Two-peak bimodal — intermediate entropy
// ============================================================

TEST(TokenEntropyMeter, BimodalDistributionHasMidEntropy) {
    TokenEntropyMeter::Config cfg;
    cfg.window_size = 64;
    TokenEntropyMeter meter(cfg);

    // Half strong-bullish, half strong-bearish.
    for (int i = 0; i < 32; ++i) meter.add( 0.9);
    for (int i = 0; i < 32; ++i) meter.add(-0.9);

    double ne = meter.get_normalized_entropy();
    // Should be above low-entropy but below near-uniform.
    EXPECT_GT(ne, 0.05);
    EXPECT_LT(ne, 0.85);
}

} // namespace

#else  // LLMQUANT_ENTROPY_METER_ENABLED not defined

TEST(TokenEntropyMeter, DisabledAtBuildTime) {
    SUCCEED();
}

#endif  // LLMQUANT_ENTROPY_METER_ENABLED
