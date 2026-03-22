#include <gtest/gtest.h>
#include "NarrativeClusterDetector.h"

namespace llmquant::tests {

TEST(NarrativeClusterDetector, DefaultConstruction) {
    NarrativeClusterDetector d;
    EXPECT_EQ(d.total_records(), 0u);
    EXPECT_EQ(d.active_cluster(), 0);
}

TEST(NarrativeClusterDetector, RecordsAccumulate) {
    NarrativeClusterDetector d;
    for (int i = 0; i < 20; ++i) d.record(static_cast<double>(i % 3));
    EXPECT_EQ(d.total_records(), 20u);
}

TEST(NarrativeClusterDetector, ActiveClusterInRange) {
    NarrativeClusterDetector::Config cfg;
    cfg.k = 3;
    NarrativeClusterDetector d(cfg);
    for (int i = 0; i < 30; ++i) d.record(static_cast<double>(i % 3));
    EXPECT_GE(d.active_cluster(), 0);
    EXPECT_LT(d.active_cluster(), 3);
}

TEST(NarrativeClusterDetector, CentroidAccessible) {
    NarrativeClusterDetector::Config cfg;
    cfg.k = 3;
    NarrativeClusterDetector d(cfg);
    for (int i = 0; i < 15; ++i) d.record(static_cast<double>(i));
    // Centroid should be finite
    double c = d.centroid(0);
    EXPECT_FALSE(std::isnan(c));
}

TEST(NarrativeClusterDetector, ClusterChangeCallback) {
    NarrativeClusterDetector::Config cfg;
    cfg.k           = 3;
    cfg.alpha       = 0.5;
    cfg.min_samples = 3;
    int changes = 0;
    cfg.on_cluster_change = [&](int, int, double) { ++changes; };
    NarrativeClusterDetector d(cfg);
    // Feed very different values to force cluster switches
    for (int i = 0; i < 30; ++i) d.record(i % 3 == 0 ? 100.0 : (i % 3 == 1 ? -100.0 : 0.0));
    EXPECT_GT(changes, 0);
}

TEST(NarrativeClusterDetector, Reset) {
    NarrativeClusterDetector d;
    for (int i = 0; i < 20; ++i) d.record(0.1);
    d.reset();
    EXPECT_EQ(d.total_records(), 0u);
    EXPECT_EQ(d.cluster_change_events(), 0u);
}

TEST(NarrativeClusterDetector, ToStatsJson) {
    NarrativeClusterDetector d;
    for (int i = 0; i < 15; ++i) d.record(0.05);
    auto json = d.to_stats_json();
    EXPECT_NE(json.find("active_cluster"), std::string::npos);
    EXPECT_NE(json.find("cluster_change_events"), std::string::npos);
}

} // namespace llmquant::tests
