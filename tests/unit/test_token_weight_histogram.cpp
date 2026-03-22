#include <gtest/gtest.h>

#ifdef LLMQUANT_WEIGHT_HISTOGRAM_ENABLED
#include "TokenWeightHistogram.h"
using namespace llmquant;

TEST(TokenWeightHistogramTest, InitialState) {
    TokenWeightHistogram h;
    EXPECT_EQ(h.total(), 0u);
    EXPECT_GE(h.entropy(), 0.0);
}

TEST(TokenWeightHistogramTest, TotalCountIncrement) {
    TokenWeightHistogram h;
    h.record(0.1);
    h.record(-0.2);
    h.record(0.5);
    EXPECT_EQ(h.total(), 3u);
}

TEST(TokenWeightHistogramTest, BucketAccumulates) {
    TokenWeightHistogram h;
    // Feed 5 values all in the positive half — should cluster into the same upper buckets.
    for (int i = 0; i < 5; ++i) h.record(0.9);
    // Mode should be in the upper buckets (index >= 15 of 20).
    EXPECT_GE(h.mode_bucket(), 15);
}

TEST(TokenWeightHistogramTest, ModeShiftsToNegative) {
    TokenWeightHistogram h;
    for (int i = 0; i < 3; ++i) h.record(0.8);
    for (int i = 0; i < 10; ++i) h.record(-0.8);
    EXPECT_LT(h.mode_bucket(), 5); // negative side
}

TEST(TokenWeightHistogramTest, DistributionShiftCallback) {
    TokenWeightHistogram::Config cfg;
    int shifts = 0;
    cfg.on_distribution_shift = [&](int) { ++shifts; };
    TokenWeightHistogram h(cfg);
    // Seed a mode in positive territory.
    for (int i = 0; i < 5; ++i) h.record(0.9);
    // Now shift mode to negative.
    for (int i = 0; i < 20; ++i) h.record(-0.9);
    EXPECT_GE(shifts, 1);
}

TEST(TokenWeightHistogramTest, EntropyRangeValid) {
    TokenWeightHistogram h;
    for (int i = 0; i < 100; ++i) h.record((i % 5 == 0) ? 0.5 : -0.3);
    EXPECT_GE(h.entropy(), 0.0);
    EXPECT_LE(h.entropy(), std::log2(TokenWeightHistogram::kBuckets) + 1e-9);
}

TEST(TokenWeightHistogramTest, ClampingOutOfRange) {
    TokenWeightHistogram h;
    h.record(5.0);   // clamped to 1.0
    h.record(-5.0);  // clamped to -1.0
    EXPECT_EQ(h.total(), 2u);
}

TEST(TokenWeightHistogramTest, BucketCountAccessor) {
    TokenWeightHistogram h;
    for (int i = 0; i < 7; ++i) h.record(0.99); // all into bucket 19
    EXPECT_EQ(h.bucket_count(19), 7u);
    EXPECT_EQ(h.bucket_count(0), 0u);
    EXPECT_EQ(h.bucket_count(-1), 0u);  // out-of-range returns 0
}

TEST(TokenWeightHistogramTest, Reset) {
    TokenWeightHistogram h;
    for (int i = 0; i < 10; ++i) h.record(0.5);
    h.reset();
    EXPECT_EQ(h.total(), 0u);
    for (int i = 0; i < TokenWeightHistogram::kBuckets; ++i)
        EXPECT_EQ(h.bucket_count(i), 0u);
}

TEST(TokenWeightHistogramTest, StatsJson) {
    TokenWeightHistogram h;
    h.record(0.3);
    std::string json = h.to_stats_json();
    EXPECT_NE(json.find("mode_bucket"), std::string::npos);
    EXPECT_NE(json.find("entropy"), std::string::npos);
    EXPECT_NE(json.find("buckets"), std::string::npos);
}

#else
TEST(TokenWeightHistogramTest, Disabled) { SUCCEED(); }
#endif
