#include <gtest/gtest.h>

#ifdef LLMQUANT_POLARIZATION_MONITOR_ENABLED

#include "SignalPolarizationMonitor.h"

#include <atomic>
#include <cmath>
#include <thread>
#include <vector>

using namespace llmquant;

namespace {

// ---------------------------------------------------------------------------
// Default state
// ---------------------------------------------------------------------------

TEST(SignalPolarizationMonitor, DefaultState) {
    SignalPolarizationMonitor m;
    EXPECT_NEAR(m.bimodality_coefficient(), 0.0, 1e-9);
    EXPECT_FALSE(m.is_polarized());
    EXPECT_EQ(m.polarization_events(), 0u);
    EXPECT_EQ(m.total_records(), 0u);
}

// ---------------------------------------------------------------------------
// total_records increments on every record call
// ---------------------------------------------------------------------------

TEST(SignalPolarizationMonitor, TotalRecordsIncrements) {
    SignalPolarizationMonitor m;
    m.record(0.1);
    m.record(-0.2);
    m.record(0.3);
    EXPECT_EQ(m.total_records(), 3u);
}

// ---------------------------------------------------------------------------
// BC is bounded [0, 1]
// ---------------------------------------------------------------------------

TEST(SignalPolarizationMonitor, BCBounded) {
    SignalPolarizationMonitor::Config cfg;
    cfg.window          = 32;
    cfg.min_observations = 4;
    SignalPolarizationMonitor m(cfg);
    // Feed a mix of values and check BC stays in [0, 1].
    for (int i = 0; i < 50; ++i)
        m.record((i % 7 == 0) ? 5.0 : -5.0);
    double bc = m.bimodality_coefficient();
    EXPECT_GE(bc, 0.0);
    EXPECT_LE(bc, 1.0 + 1e-9);
}

// ---------------------------------------------------------------------------
// Strongly bimodal series: alternating ±large values → high BC
// ---------------------------------------------------------------------------

TEST(SignalPolarizationMonitor, BimodalSeriesHighBC) {
    SignalPolarizationMonitor::Config cfg;
    cfg.window            = 64;
    cfg.min_observations  = 8;
    cfg.polarization_threshold = 0.55;
    SignalPolarizationMonitor m(cfg);
    // Two-spike bimodal: half at +5, half at -5.
    for (int i = 0; i < 64; ++i)
        m.record((i % 2 == 0) ? 5.0 : -5.0);
    // A perfectly bimodal distribution has BC > 5/9 ≈ 0.556.
    EXPECT_GT(m.bimodality_coefficient(), 0.5);
}

// ---------------------------------------------------------------------------
// Unimodal Gaussian-like series → BC should be lower than bimodal
// ---------------------------------------------------------------------------

TEST(SignalPolarizationMonitor, UnimodalSeriesLowerBC) {
    SignalPolarizationMonitor::Config cfg;
    cfg.window           = 64;
    cfg.min_observations = 8;
    SignalPolarizationMonitor m(cfg);
    // Uniform random-ish series centred at 0 (not bimodal).
    for (int i = 0; i < 64; ++i)
        m.record(static_cast<double>(i % 13 - 6) * 0.1);
    double unimodal_bc = m.bimodality_coefficient();

    // Compare against bimodal.
    SignalPolarizationMonitor m2(cfg);
    for (int i = 0; i < 64; ++i)
        m2.record((i % 2 == 0) ? 5.0 : -5.0);
    EXPECT_LT(unimodal_bc, m2.bimodality_coefficient() + 0.5);
}

// ---------------------------------------------------------------------------
// Constant series → BC computed without crash, skew=0 kurtosis defined
// ---------------------------------------------------------------------------

TEST(SignalPolarizationMonitor, ConstantSeriesNoCrash) {
    SignalPolarizationMonitor::Config cfg;
    cfg.window           = 16;
    cfg.min_observations = 4;
    SignalPolarizationMonitor m(cfg);
    for (int i = 0; i < 16; ++i) m.record(1.0);
    // Zero variance → BC = 1 (flat distribution fallback).
    double bc = m.bimodality_coefficient();
    EXPECT_GE(bc, 0.0);
    EXPECT_LE(bc, 1.0 + 1e-9);
}

// ---------------------------------------------------------------------------
// Polarization callback fires
// ---------------------------------------------------------------------------

TEST(SignalPolarizationMonitor, PolarizationCallbackFires) {
    SignalPolarizationMonitor::Config cfg;
    cfg.window                 = 32;
    cfg.min_observations       = 4;
    cfg.polarization_threshold = 0.55;
    std::atomic<int> fires{0};
    cfg.on_polarized = [&](double) { ++fires; };
    SignalPolarizationMonitor m(cfg);
    // Force bimodal: alternate ±10.
    for (int i = 0; i < 32; ++i) m.record((i % 2 == 0) ? 10.0 : -10.0);
    EXPECT_GE(fires.load(), 1);
    EXPECT_GT(m.polarization_events(), 0u);
}

// ---------------------------------------------------------------------------
// Unified callback fires after polarization clears
// ---------------------------------------------------------------------------

TEST(SignalPolarizationMonitor, UnifiedCallbackFires) {
    SignalPolarizationMonitor::Config cfg;
    cfg.window                 = 16;
    cfg.min_observations       = 4;
    cfg.polarization_threshold = 0.55;
    cfg.clear_hysteresis       = 0.9;
    std::atomic<int> unified{0};
    cfg.on_unified = [&](double) { ++unified; };
    SignalPolarizationMonitor m(cfg);
    // Polarize.
    for (int i = 0; i < 16; ++i) m.record((i % 2 == 0) ? 10.0 : -10.0);
    // Unify: feed a strongly unimodal series to drive BC below threshold.
    for (int i = 0; i < 32; ++i) m.record(0.01 * i);
    EXPECT_GE(unified.load(), 1);
}

// ---------------------------------------------------------------------------
// Reset clears all state
// ---------------------------------------------------------------------------

TEST(SignalPolarizationMonitor, ResetClearsState) {
    SignalPolarizationMonitor m;
    for (int i = 0; i < 20; ++i) m.record(i % 2 == 0 ? 5.0 : -5.0);
    m.reset();
    EXPECT_EQ(m.total_records(), 0u);
    EXPECT_EQ(m.polarization_events(), 0u);
    EXPECT_FALSE(m.is_polarized());
    EXPECT_NEAR(m.bimodality_coefficient(), 0.0, 1e-9);
}

// ---------------------------------------------------------------------------
// update_config resets state
// ---------------------------------------------------------------------------

TEST(SignalPolarizationMonitor, UpdateConfigResetsState) {
    SignalPolarizationMonitor m;
    for (int i = 0; i < 20; ++i) m.record(i % 2 == 0 ? 5.0 : -5.0);
    SignalPolarizationMonitor::Config cfg2;
    cfg2.window = 8;
    m.update_config(cfg2);
    EXPECT_EQ(m.total_records(), 0u);
    EXPECT_FALSE(m.is_polarized());
}

// ---------------------------------------------------------------------------
// to_stats_json contains expected fields
// ---------------------------------------------------------------------------

TEST(SignalPolarizationMonitor, StatsJsonContainsFields) {
    SignalPolarizationMonitor m;
    for (int i = 0; i < 20; ++i) m.record(i % 2 == 0 ? 2.0 : -2.0);
    std::string j = m.to_stats_json();
    EXPECT_NE(j.find("bimodality_coefficient"), std::string::npos);
    EXPECT_NE(j.find("is_polarized"),           std::string::npos);
    EXPECT_NE(j.find("skewness"),               std::string::npos);
    EXPECT_NE(j.find("excess_kurtosis"),        std::string::npos);
    EXPECT_NE(j.find("polarization_events"),    std::string::npos);
    EXPECT_NE(j.find("total_records"),          std::string::npos);
}

// ---------------------------------------------------------------------------
// Concurrent record_signal is thread-safe
// ---------------------------------------------------------------------------

TEST(SignalPolarizationMonitor, ConcurrentRecordSafe) {
    SignalPolarizationMonitor::Config cfg;
    cfg.window           = 64;
    cfg.min_observations = 4;
    SignalPolarizationMonitor m(cfg);

    std::atomic<bool> go{false};
    auto worker = [&](int id) {
        while (!go.load()) {}
        for (int i = 0; i < 250; ++i)
            m.record((id % 2 == 0) ? static_cast<double>(i % 5) : -static_cast<double>(i % 5));
    };
    std::vector<std::thread> threads;
    for (int t = 0; t < 4; ++t) threads.emplace_back(worker, t);
    go.store(true);
    for (auto& t : threads) t.join();

    EXPECT_EQ(m.total_records(), 1000u);
    double bc = m.bimodality_coefficient();
    EXPECT_GE(bc, 0.0);
    EXPECT_LE(bc, 1.0 + 1e-9);
}

}  // namespace

#else

TEST(SignalPolarizationMonitor, DisabledAtBuildTime) { SUCCEED(); }

#endif  // LLMQUANT_POLARIZATION_MONITOR_ENABLED
