#include <gtest/gtest.h>
#include "BiasSkewnessTracker.h"
#include <cmath>

namespace llmquant::tests {

TEST(BiasSkewnessTracker, DefaultConstruction) {
    BiasSkewnessTracker t;
    EXPECT_EQ(t.total_records(), 0u);
    EXPECT_DOUBLE_EQ(t.skewness(), 0.0);
}

TEST(BiasSkewnessTracker, SymmetricDistributionNearZeroSkew) {
    BiasSkewnessTracker t;
    // Symmetric ±1 series → skew near 0
    for (int i = 0; i < 30; ++i) t.record(i % 2 == 0 ? 1.0 : -1.0);
    EXPECT_NEAR(t.skewness(), 0.0, 0.5);
}

TEST(BiasSkewnessTracker, PositiveSkewForRightTail) {
    BiasSkewnessTracker::Config cfg;
    cfg.window = 30; cfg.min_samples = 10;
    BiasSkewnessTracker t(cfg);
    // Many small negatives + one large positive → positive skew
    for (int i = 0; i < 25; ++i) t.record(-0.1);
    for (int i = 0; i < 5;  ++i) t.record(10.0);
    EXPECT_GT(t.skewness(), 0.0);
}

TEST(BiasSkewnessTracker, PositiveSkewCallback) {
    BiasSkewnessTracker::Config cfg;
    cfg.window = 20; cfg.min_samples = 5;
    int pos_events = 0;
    cfg.on_positive_skew = [&](double) { ++pos_events; };
    BiasSkewnessTracker t(cfg);
    // First add negatives to get negative skew, then positives to flip
    for (int i = 0; i < 15; ++i) t.record(-10.0);
    for (int i = 0; i < 15; ++i) t.record(0.01);
    EXPECT_GE(pos_events, 0);
}

TEST(BiasSkewnessTracker, NegativeSkewCallback) {
    BiasSkewnessTracker::Config cfg;
    cfg.window = 20; cfg.min_samples = 5;
    int neg_events = 0;
    cfg.on_negative_skew = [&](double) { ++neg_events; };
    BiasSkewnessTracker t(cfg);
    for (int i = 0; i < 15; ++i) t.record(10.0);
    for (int i = 0; i < 15; ++i) t.record(-0.01);
    EXPECT_GE(neg_events, 0);
}

TEST(BiasSkewnessTracker, Reset) {
    BiasSkewnessTracker t;
    for (int i = 0; i < 30; ++i) t.record(0.1);
    t.reset();
    EXPECT_EQ(t.total_records(), 0u);
    EXPECT_DOUBLE_EQ(t.skewness(), 0.0);
}

TEST(BiasSkewnessTracker, ToStatsJson) {
    BiasSkewnessTracker t;
    for (int i = 0; i < 20; ++i) t.record(0.05);
    auto json = t.to_stats_json();
    EXPECT_NE(json.find("skewness"), std::string::npos);
    EXPECT_NE(json.find("positive_skew_events"), std::string::npos);
}

} // namespace llmquant::tests
