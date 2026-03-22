#include <gtest/gtest.h>
#include "SignalAccelerationMeter.h"
#include <cmath>

namespace llmquant::tests {

TEST(SignalAccelerationMeter, DefaultConstruction) {
    SignalAccelerationMeter m;
    EXPECT_EQ(m.total_records(), 0u);
    EXPECT_DOUBLE_EQ(m.velocity(), 0.0);
    EXPECT_DOUBLE_EQ(m.acceleration(), 0.0);
}

TEST(SignalAccelerationMeter, VelocityNonZeroAfterRecords) {
    SignalAccelerationMeter m;
    for (int i = 0; i < 20; ++i) m.record(static_cast<double>(i) * 0.1);
    EXPECT_NE(m.velocity(), 0.0);
}

TEST(SignalAccelerationMeter, MaxAccelIncreases) {
    SignalAccelerationMeter m;
    for (int i = 0; i < 30; ++i) m.record(i % 2 == 0 ? 1.0 : -1.0);
    EXPECT_GE(m.max_accel(), 0.0);
}

TEST(SignalAccelerationMeter, InflectionCallback) {
    SignalAccelerationMeter::Config cfg;
    cfg.min_samples = 3;
    int inflections = 0;
    cfg.on_inflection = [&](double, double) { ++inflections; };
    SignalAccelerationMeter m(cfg);
    // Alternating velocity → acceleration sign changes
    for (int i = 0; i < 30; ++i) m.record(i % 3 == 0 ? 1.0 : -0.5);
    EXPECT_GT(inflections, 0);
}

TEST(SignalAccelerationMeter, SurgeCallback) {
    SignalAccelerationMeter::Config cfg;
    cfg.surge_threshold = 0.001;
    cfg.min_samples = 2;
    int surges = 0;
    cfg.on_surge = [&](double) { ++surges; };
    SignalAccelerationMeter m(cfg);
    for (int i = 0; i < 20; ++i) m.record(i % 2 == 0 ? 5.0 : -5.0);
    EXPECT_GT(surges, 0);
}

TEST(SignalAccelerationMeter, Reset) {
    SignalAccelerationMeter m;
    for (int i = 0; i < 20; ++i) m.record(0.1);
    m.reset();
    EXPECT_EQ(m.total_records(), 0u);
    EXPECT_DOUBLE_EQ(m.velocity(), 0.0);
    EXPECT_DOUBLE_EQ(m.acceleration(), 0.0);
    EXPECT_DOUBLE_EQ(m.max_accel(), 0.0);
}

TEST(SignalAccelerationMeter, ToStatsJson) {
    SignalAccelerationMeter m;
    for (int i = 0; i < 15; ++i) m.record(0.05);
    auto json = m.to_stats_json();
    EXPECT_NE(json.find("velocity"), std::string::npos);
    EXPECT_NE(json.find("acceleration"), std::string::npos);
}

} // namespace llmquant::tests
