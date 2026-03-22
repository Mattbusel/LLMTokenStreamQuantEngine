#include <gtest/gtest.h>
#include "BiasEntropyRate.h"
#include <cmath>

namespace llmquant::tests {

TEST(BiasEntropyRate, DefaultConstruction) {
    BiasEntropyRate e;
    EXPECT_EQ(e.total_records(), 0u);
    EXPECT_DOUBLE_EQ(e.entropy(), 0.0);
}

TEST(BiasEntropyRate, EntropyInRange) {
    BiasEntropyRate e;
    for (int i = 0; i < 30; ++i) e.record(i % 5 == 0 ? 1.0 : -0.1);
    double h = e.entropy();
    EXPECT_GE(h, 0.0);
    EXPECT_LE(h, 1.0);
}

TEST(BiasEntropyRate, HighEntropyForRandom) {
    BiasEntropyRate::Config cfg;
    cfg.window = 32; cfg.n_buckets = 8; cfg.min_samples = 10;
    BiasEntropyRate e(cfg);
    // Wide spread → high entropy
    for (int i = 0; i < 40; ++i) e.record(static_cast<double>(i % 8) - 3.5);
    EXPECT_GT(e.entropy(), 0.5);
}

TEST(BiasEntropyRate, LowEntropyForConstant) {
    BiasEntropyRate::Config cfg;
    cfg.window = 32; cfg.n_buckets = 8; cfg.min_samples = 10;
    BiasEntropyRate e(cfg);
    // All same value → one bucket → zero entropy
    for (int i = 0; i < 40; ++i) e.record(1.0);
    EXPECT_NEAR(e.entropy(), 0.0, 0.01);
}

TEST(BiasEntropyRate, HighEntropyCallback) {
    BiasEntropyRate::Config cfg;
    cfg.window = 20; cfg.n_buckets = 4; cfg.min_samples = 5;
    cfg.high_threshold = 0.1; // easy to trigger
    int events = 0;
    cfg.on_high_entropy = [&](double) { ++events; };
    BiasEntropyRate e(cfg);
    for (int i = 0; i < 30; ++i) e.record(static_cast<double>(i % 4));
    EXPECT_GT(events, 0);
}

TEST(BiasEntropyRate, LowEntropyCallback) {
    BiasEntropyRate::Config cfg;
    cfg.window = 20; cfg.n_buckets = 4; cfg.min_samples = 5;
    cfg.low_threshold = 1.0; // always below (entropy ≤ 1)
    int events = 0;
    cfg.on_low_entropy = [&](double) { ++events; };
    BiasEntropyRate e(cfg);
    for (int i = 0; i < 30; ++i) e.record(1.0);
    EXPECT_GT(events, 0);
}

TEST(BiasEntropyRate, Reset) {
    BiasEntropyRate e;
    for (int i = 0; i < 20; ++i) e.record(0.1);
    e.reset();
    EXPECT_EQ(e.total_records(), 0u);
    EXPECT_DOUBLE_EQ(e.entropy(), 0.0);
}

TEST(BiasEntropyRate, ToStatsJson) {
    BiasEntropyRate e;
    for (int i = 0; i < 20; ++i) e.record(0.05);
    auto json = e.to_stats_json();
    EXPECT_NE(json.find("entropy"), std::string::npos);
    EXPECT_NE(json.find("high_entropy_events"), std::string::npos);
}

} // namespace llmquant::tests
