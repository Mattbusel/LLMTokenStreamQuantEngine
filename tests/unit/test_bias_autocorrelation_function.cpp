#include <gtest/gtest.h>

#ifdef LLMQUANT_BIAS_ACF_ENABLED

#include "BiasAutocorrelationFunction.h"

#include <atomic>
#include <cmath>
#include <thread>
#include <vector>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

using namespace llmquant;

namespace {

TEST(BiasAutocorrelationFunction, DefaultState) {
    BiasAutocorrelationFunction acf;
    EXPECT_TRUE(std::isnan(acf.r1()));
    EXPECT_EQ(acf.dominant_lag(), 0);
    EXPECT_FALSE(acf.is_periodic());
    EXPECT_EQ(acf.total_records(), 0u);
}

TEST(BiasAutocorrelationFunction, TotalRecordsIncrements) {
    BiasAutocorrelationFunction acf;
    acf.record(0.5);
    acf.record(-0.3);
    EXPECT_EQ(acf.total_records(), 2u);
}

TEST(BiasAutocorrelationFunction, NaNBeforeMinSamples) {
    BiasAutocorrelationFunction::Config cfg;
    cfg.window      = 16;
    cfg.min_samples = 16;
    BiasAutocorrelationFunction acf(cfg);
    for (int i = 0; i < 15; ++i) acf.record(static_cast<double>(i % 5));
    EXPECT_TRUE(std::isnan(acf.r1()));
}

TEST(BiasAutocorrelationFunction, TrendingSignalPositiveR1) {
    BiasAutocorrelationFunction::Config cfg;
    cfg.window      = 32;
    cfg.min_samples = 32;
    cfg.max_lag     = 8;
    BiasAutocorrelationFunction acf(cfg);
    // Slowly increasing signal → strong positive lag-1 autocorrelation
    for (int i = 0; i < 32; ++i)
        acf.record(static_cast<double>(i) * 0.1);
    EXPECT_GT(acf.r1(), 0.5);
}

TEST(BiasAutocorrelationFunction, AlternatingSignalNegativeR1) {
    BiasAutocorrelationFunction::Config cfg;
    cfg.window      = 32;
    cfg.min_samples = 32;
    cfg.max_lag     = 8;
    BiasAutocorrelationFunction acf(cfg);
    // Alternating ±1 → strong negative lag-1 autocorrelation
    for (int i = 0; i < 32; ++i)
        acf.record((i % 2 == 0) ? 1.0 : -1.0);
    EXPECT_LT(acf.r1(), -0.5);
}

TEST(BiasAutocorrelationFunction, PeriodicPatternDetected) {
    BiasAutocorrelationFunction::Config cfg;
    cfg.window              = 64;
    cfg.min_samples         = 64;
    cfg.max_lag             = 16;
    cfg.periodic_threshold  = 0.3;
    std::atomic<int> fires{0};
    cfg.on_periodic_pattern = [&](int, double) { ++fires; };
    BiasAutocorrelationFunction acf(cfg);
    // Sine wave with period 8: strong autocorrelation at lag 8
    for (int i = 0; i < 64; ++i)
        acf.record(std::sin(2.0 * M_PI * i / 8.0));
    EXPECT_GE(fires.load(), 1);
    EXPECT_GE(acf.periodic_events(), 1u);
    EXPECT_TRUE(acf.is_periodic());
}

TEST(BiasAutocorrelationFunction, ConstantSignalZeroR1) {
    BiasAutocorrelationFunction::Config cfg;
    cfg.window      = 16;
    cfg.min_samples = 16;
    cfg.max_lag     = 4;
    BiasAutocorrelationFunction acf(cfg);
    for (int i = 0; i < 16; ++i) acf.record(1.0);
    // Constant signal → variance ≈ 0, r1 set to 0 by convention
    EXPECT_TRUE(std::isfinite(acf.r1()));
    EXPECT_NEAR(acf.r1(), 0.0, 1e-6);
}

TEST(BiasAutocorrelationFunction, AcfSnapshotLength) {
    BiasAutocorrelationFunction::Config cfg;
    cfg.window      = 16;
    cfg.min_samples = 16;
    cfg.max_lag     = 6;
    BiasAutocorrelationFunction acf(cfg);
    for (int i = 0; i < 16; ++i) acf.record(static_cast<double>(i % 4));
    auto snap = acf.acf_snapshot();
    EXPECT_EQ(static_cast<int>(snap.size()), cfg.max_lag);
}

TEST(BiasAutocorrelationFunction, ResetClearsState) {
    BiasAutocorrelationFunction acf;
    for (int i = 0; i < 32; ++i) acf.record(static_cast<double>(i));
    acf.reset();
    EXPECT_EQ(acf.total_records(), 0u);
    EXPECT_TRUE(std::isnan(acf.r1()));
    EXPECT_FALSE(acf.is_periodic());
}

TEST(BiasAutocorrelationFunction, UpdateConfigResetsState) {
    BiasAutocorrelationFunction acf;
    for (int i = 0; i < 32; ++i) acf.record(static_cast<double>(i));
    BiasAutocorrelationFunction::Config cfg2;
    cfg2.window = 16;
    acf.update_config(cfg2);
    EXPECT_EQ(acf.total_records(), 0u);
    EXPECT_TRUE(std::isnan(acf.r1()));
}

TEST(BiasAutocorrelationFunction, ToStatsJsonContainsKeys) {
    BiasAutocorrelationFunction::Config cfg;
    cfg.window      = 16;
    cfg.min_samples = 16;
    BiasAutocorrelationFunction acf(cfg);
    for (int i = 0; i < 16; ++i) acf.record(static_cast<double>(i));
    std::string j = acf.to_stats_json();
    EXPECT_NE(j.find("r1"),               std::string::npos);
    EXPECT_NE(j.find("dominant_lag"),     std::string::npos);
    EXPECT_NE(j.find("dominant_acf"),     std::string::npos);
    EXPECT_NE(j.find("is_periodic"),      std::string::npos);
    EXPECT_NE(j.find("periodic_events"),  std::string::npos);
    EXPECT_NE(j.find("total_records"),    std::string::npos);
}

TEST(BiasAutocorrelationFunction, ConcurrentRecordSafe) {
    BiasAutocorrelationFunction acf;
    std::vector<std::thread> threads;
    for (int t = 0; t < 4; ++t) {
        threads.emplace_back([&, t] {
            for (int i = 0; i < 250; ++i)
                acf.record(static_cast<double>((i + t) % 10) * 0.1);
        });
    }
    for (auto& th : threads) th.join();
    EXPECT_EQ(acf.total_records(), 1000u);
}

} // namespace

#else

TEST(BiasAutocorrelationFunction, DisabledAtBuildTime) { SUCCEED(); }

#endif  // LLMQUANT_BIAS_ACF_ENABLED
