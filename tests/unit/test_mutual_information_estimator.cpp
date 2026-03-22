#include <gtest/gtest.h>

#ifdef LLMQUANT_MUTUAL_INFORMATION_ENABLED

#include "MutualInformationEstimator.h"

#include <cmath>
#include <thread>
#include <vector>

using namespace llmquant;

namespace {

// ============================================================
// Construction — initial state
// ============================================================

TEST(MutualInformationEstimator, DefaultStateIsZero) {
    MutualInformationEstimator est;
    EXPECT_DOUBLE_EQ(est.mi(), 0.0);
    EXPECT_DOUBLE_EQ(est.normalized_mi(), 0.0);
    EXPECT_EQ(est.sample_count(), 0);
}

// ============================================================
// Insufficient samples → MI stays 0
// ============================================================

TEST(MutualInformationEstimator, InsufficientSamplesKeepsMiZero) {
    MutualInformationEstimator::Config cfg;
    cfg.min_samples = 20;
    MutualInformationEstimator est(cfg);

    for (int i = 0; i < 5; ++i)
        est.record(static_cast<double>(i) * 0.1, static_cast<double>(i) * 0.01);

    EXPECT_DOUBLE_EQ(est.mi(), 0.0);
    EXPECT_EQ(est.sample_count(), 5);
}

// ============================================================
// Perfectly correlated inputs → high NMI
// ============================================================

TEST(MutualInformationEstimator, PerfectCorrelationHighNmi) {
    MutualInformationEstimator::Config cfg;
    cfg.bins        = 10;
    cfg.window_size = 100;
    cfg.min_samples = 10;
    MutualInformationEstimator est(cfg);

    // sentiment and return are identical → maximum mutual information
    for (int i = 0; i < 100; ++i) {
        double v = (i % 10) * 0.1 - 0.45;  // cycles through 10 distinct values
        est.record(v, v * 0.1);
    }

    EXPECT_GT(est.mi(), 0.0);
    EXPECT_GT(est.normalized_mi(), 0.3);
}

// ============================================================
// Independent uniform inputs → near-zero NMI
// ============================================================

TEST(MutualInformationEstimator, IndependentInputsLowNmi) {
    MutualInformationEstimator::Config cfg;
    cfg.bins        = 5;
    cfg.window_size = 200;
    cfg.min_samples = 50;
    MutualInformationEstimator est(cfg);

    // Two independent cycling sequences
    for (int i = 0; i < 200; ++i) {
        double s = ((i * 7) % 5) * 0.4 - 0.8;
        double r = ((i * 3) % 5) * 0.04 - 0.08;
        est.record(s, r);
    }

    EXPECT_LT(est.normalized_mi(), 0.3);
}

// ============================================================
// Window eviction keeps sample_count bounded
// ============================================================

TEST(MutualInformationEstimator, SampleCountBoundedByWindowSize) {
    MutualInformationEstimator::Config cfg;
    cfg.window_size = 50;
    cfg.min_samples = 10;
    MutualInformationEstimator est(cfg);

    for (int i = 0; i < 200; ++i)
        est.record(0.1 * (i % 10 - 5), 0.01 * (i % 7 - 3));

    EXPECT_LE(est.sample_count(), 50);
}

// ============================================================
// reset() clears state
// ============================================================

TEST(MutualInformationEstimator, ResetClearsState) {
    MutualInformationEstimator::Config cfg;
    cfg.min_samples = 5;
    MutualInformationEstimator est(cfg);

    for (int i = 0; i < 20; ++i)
        est.record(static_cast<double>(i % 10) * 0.1 - 0.45,
                   static_cast<double>(i % 10) * 0.01 - 0.045);

    EXPECT_GT(est.mi(), 0.0);
    est.reset();
    EXPECT_DOUBLE_EQ(est.mi(), 0.0);
    EXPECT_EQ(est.sample_count(), 0);
}

// ============================================================
// update_config resets state
// ============================================================

TEST(MutualInformationEstimator, UpdateConfigResetsState) {
    MutualInformationEstimator::Config cfg;
    cfg.min_samples = 5;
    MutualInformationEstimator est(cfg);

    for (int i = 0; i < 20; ++i)
        est.record(0.1 * (i % 5 - 2), 0.01 * (i % 5 - 2));

    MutualInformationEstimator::Config cfg2;
    cfg2.bins        = 8;
    cfg2.window_size = 100;
    cfg2.min_samples = 5;
    est.update_config(cfg2);

    EXPECT_DOUBLE_EQ(est.mi(), 0.0);
    EXPECT_EQ(est.sample_count(), 0);
}

// ============================================================
// NMI is in [0, 1]
// ============================================================

TEST(MutualInformationEstimator, NmiInUnitInterval) {
    MutualInformationEstimator::Config cfg;
    cfg.min_samples = 5;
    cfg.bins        = 6;
    cfg.window_size = 100;
    MutualInformationEstimator est(cfg);

    for (int i = 0; i < 80; ++i)
        est.record(static_cast<double>(i % 6) * 0.3 - 0.75,
                   static_cast<double>(i % 6) * 0.03 - 0.075);

    double nmi = est.normalized_mi();
    EXPECT_GE(nmi, 0.0);
    EXPECT_LE(nmi, 1.0);
}

// ============================================================
// to_stats_json contains required keys
// ============================================================

TEST(MutualInformationEstimator, ToStatsJsonContainsKeys) {
    MutualInformationEstimator est;
    est.record(0.1, 0.01);
    std::string json = est.to_stats_json();
    EXPECT_NE(json.find("mi"),            std::string::npos);
    EXPECT_NE(json.find("normalized_mi"), std::string::npos);
    EXPECT_NE(json.find("sample_count"),  std::string::npos);
    EXPECT_NE(json.find("bins"),          std::string::npos);
    EXPECT_NE(json.find("total_records"), std::string::npos);
}

// ============================================================
// Thread safety: concurrent record() calls do not crash
// ============================================================

TEST(MutualInformationEstimator, ConcurrentRecordSafe) {
    MutualInformationEstimator::Config cfg;
    cfg.window_size = 64;
    cfg.min_samples = 10;
    MutualInformationEstimator est(cfg);

    std::vector<std::thread> threads;
    for (int t = 0; t < 4; ++t) {
        threads.emplace_back([&, t] {
            for (int i = 0; i < 100; ++i) {
                double s = static_cast<double>((i + t) % 10) * 0.1 - 0.45;
                double r = static_cast<double>((i + t) % 7) * 0.02 - 0.06;
                est.record(s, r);
            }
        });
    }
    for (auto& th : threads) th.join();

    // No crash; NMI stays in valid range.
    EXPECT_GE(est.normalized_mi(), 0.0);
    EXPECT_LE(est.normalized_mi(), 1.0);
}

} // namespace

#else  // LLMQUANT_MUTUAL_INFO_ENABLED not defined

TEST(MutualInformationEstimator, DisabledAtBuildTime) {
    SUCCEED();
}

#endif  // LLMQUANT_MUTUAL_INFORMATION_ENABLED
