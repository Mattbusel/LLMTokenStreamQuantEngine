#include <gtest/gtest.h>

#ifdef LLMQUANT_ANOMALY_DETECTOR_ENABLED
#include "AnomalyDetector.h"
#include <cmath>

using namespace llmquant;

TEST(AnomalyDetectorTest, ConstructsOk) {
    AnomalyDetector det;
    EXPECT_EQ(det.sample_count(), 0);
    EXPECT_EQ(det.total_records(), 0u);
}

TEST(AnomalyDetectorTest, ReturnsNanBeforeMinSamples) {
    AnomalyDetector det;
    double z = det.record(1.0);
    EXPECT_TRUE(std::isnan(z));
}

TEST(AnomalyDetectorTest, AccumulatesSamples) {
    AnomalyDetector det;
    for (int i = 0; i < 15; ++i) det.record(static_cast<double>(i));
    EXPECT_EQ(det.sample_count(), 15);
    EXPECT_EQ(det.total_records(), 15u);
}

TEST(AnomalyDetectorTest, RollingWindowCapEnforced) {
    AnomalyDetector::Config cfg;
    cfg.window_size = 10;
    cfg.min_samples = 2;
    AnomalyDetector det(cfg);
    for (int i = 0; i < 30; ++i) det.record(1.0);
    EXPECT_EQ(det.sample_count(), 10);
}

TEST(AnomalyDetectorTest, RollingMeanCorrect) {
    AnomalyDetector::Config cfg;
    cfg.window_size = 20;
    cfg.min_samples = 3;
    AnomalyDetector det(cfg);
    det.record(1.0);
    det.record(2.0);
    det.record(3.0);
    EXPECT_NEAR(det.rolling_mean(), 2.0, 1e-9);
}

TEST(AnomalyDetectorTest, RollingStddevCorrect) {
    AnomalyDetector::Config cfg;
    cfg.window_size = 20;
    cfg.min_samples = 3;
    AnomalyDetector det(cfg);
    // Feed 100 identical values: stddev should be 0.
    for (int i = 0; i < 20; ++i) det.record(5.0);
    EXPECT_NEAR(det.rolling_stddev(), 0.0, 1e-9);
}

TEST(AnomalyDetectorTest, SoftAnomalyCallbackFires) {
    AnomalyDetector::Config cfg;
    cfg.window_size    = 50;
    cfg.min_samples    = 5;
    cfg.soft_threshold = 2.0;
    cfg.hard_threshold = 100.0; // disable hard
    int soft_count = 0;
    cfg.soft_cb = [&](const AnomalyDetector::AnomalyEvent&) { ++soft_count; };
    AnomalyDetector det(cfg);

    // Feed alternating ±1 background (mean=0, stddev=1) so z is computable,
    // then inject a spike far above the soft threshold.
    for (int i = 0; i < 10; ++i) det.record(i % 2 == 0 ? -1.0 : 1.0);
    det.record(50.0); // z ≈ 50 — well above soft=2, below hard=100

    EXPECT_GE(soft_count, 1);
    EXPECT_EQ(det.soft_anomalies(), static_cast<uint64_t>(soft_count));
}

TEST(AnomalyDetectorTest, HardAnomalyCallbackFires) {
    AnomalyDetector::Config cfg;
    cfg.window_size    = 50;
    cfg.min_samples    = 5;
    cfg.soft_threshold = 2.0;
    cfg.hard_threshold = 3.0;
    int hard_count = 0;
    cfg.hard_cb = [&](const AnomalyDetector::AnomalyEvent& e) {
        EXPECT_EQ(e.severity, AnomalyDetector::Severity::Hard);
        ++hard_count;
    };
    AnomalyDetector det(cfg);

    for (int i = 0; i < 10; ++i) det.record(i % 2 == 0 ? -1.0 : 1.0);
    det.record(100.0); // z ≈ 100 >> 3

    EXPECT_GE(hard_count, 1);
}

TEST(AnomalyDetectorTest, SoftNotFiredWhenHardExceeded) {
    AnomalyDetector::Config cfg;
    cfg.window_size    = 50;
    cfg.min_samples    = 5;
    cfg.soft_threshold = 2.0;
    cfg.hard_threshold = 3.0;
    int soft_count = 0, hard_count = 0;
    cfg.soft_cb = [&](const AnomalyDetector::AnomalyEvent&) { ++soft_count; };
    cfg.hard_cb = [&](const AnomalyDetector::AnomalyEvent&) { ++hard_count; };
    AnomalyDetector det(cfg);

    for (int i = 0; i < 10; ++i) det.record(i % 2 == 0 ? -1.0 : 1.0);
    det.record(100.0); // z ≈ 100 >> 3 — hard anomaly only

    EXPECT_EQ(soft_count, 0);
    EXPECT_GE(hard_count, 1);
}

TEST(AnomalyDetectorTest, ResetClearsWindow) {
    AnomalyDetector det;
    for (int i = 0; i < 20; ++i) det.record(1.0);
    det.reset();
    EXPECT_EQ(det.sample_count(), 0);
    EXPECT_NEAR(det.rolling_mean(), 0.0, 1e-9);
    EXPECT_TRUE(std::isnan(det.record(1.0)));
}

TEST(AnomalyDetectorTest, LastZScoreUpdates) {
    AnomalyDetector::Config cfg;
    cfg.min_samples = 5;
    AnomalyDetector det(cfg);
    for (int i = 0; i < 10; ++i) det.record(1.0);
    double z = det.record(1.0);
    EXPECT_DOUBLE_EQ(z, det.last_z_score());
}

TEST(AnomalyDetectorTest, UpdateConfigResetsWindow) {
    AnomalyDetector det;
    for (int i = 0; i < 20; ++i) det.record(1.0);
    AnomalyDetector::Config cfg2;
    cfg2.window_size = 10;
    det.update_config(cfg2);
    EXPECT_EQ(det.sample_count(), 0);
}

TEST(AnomalyDetectorTest, StatsJsonContainsRequiredKeys) {
    AnomalyDetector det;
    for (int i = 0; i < 15; ++i) det.record(static_cast<double>(i));
    std::string json = det.to_stats_json();
    EXPECT_NE(json.find("sample_count"), std::string::npos);
    EXPECT_NE(json.find("rolling_mean"), std::string::npos);
    EXPECT_NE(json.find("last_z_score"), std::string::npos);
    EXPECT_NE(json.find("soft_anomalies"), std::string::npos);
    EXPECT_NE(json.find("hard_anomalies"), std::string::npos);
}

TEST(AnomalyDetectorTest, AnomalyEventFieldsPopulated) {
    AnomalyDetector::Config cfg;
    cfg.window_size    = 50;
    cfg.min_samples    = 5;
    cfg.soft_threshold = 2.0;
    cfg.hard_threshold = 100.0;
    AnomalyDetector::AnomalyEvent captured{};
    cfg.soft_cb = [&](const AnomalyDetector::AnomalyEvent& e) { captured = e; };
    AnomalyDetector det(cfg);

    // Alternating ±1 background: mean=0, stddev=1 → z(50) ≈ 50.
    for (int i = 0; i < 10; ++i) det.record(i % 2 == 0 ? -1.0 : 1.0);
    det.record(50.0);

    EXPECT_DOUBLE_EQ(captured.value, 50.0);
    EXPECT_GT(std::abs(captured.z_score), 2.0);
    EXPECT_NEAR(captured.mean, 0.0, 1.0);
    EXPECT_GT(captured.stddev, 0.0);
    EXPECT_EQ(captured.severity, AnomalyDetector::Severity::Soft);
}

TEST(AnomalyDetectorTest, CustomNameAppearsInJson) {
    AnomalyDetector::Config cfg;
    cfg.name = "entropy_stream";
    AnomalyDetector det(cfg);
    EXPECT_NE(det.to_stats_json().find("entropy_stream"), std::string::npos);
}

#else
TEST(AnomalyDetectorTest, FeatureFlagDisabled) { SUCCEED(); }
#endif
