#include <gtest/gtest.h>

#ifdef LLMQUANT_WAVELET_DECOMPOSER_ENABLED

#include "WaveletSignalDecomposer.h"

#include <atomic>
#include <cmath>
#include <thread>
#include <vector>

using namespace llmquant;

namespace {

// -----------------------------------------------------------------------
// Default state
// -----------------------------------------------------------------------

TEST(WaveletSignalDecomposer, DefaultState) {
    WaveletSignalDecomposer wd;
    EXPECT_DOUBLE_EQ(wd.approx_mean(), 0.0);
    EXPECT_DOUBLE_EQ(wd.max_detail_energy(), 0.0);
    EXPECT_FALSE(wd.is_spiking());
    EXPECT_EQ(wd.total_records(), 0u);
    EXPECT_EQ(wd.spike_events(), 0u);
}

// -----------------------------------------------------------------------
// Total records increments
// -----------------------------------------------------------------------

TEST(WaveletSignalDecomposer, TotalRecordsIncrements) {
    WaveletSignalDecomposer wd;
    wd.record(1.0);
    wd.record(2.0);
    EXPECT_EQ(wd.total_records(), 2u);
}

// -----------------------------------------------------------------------
// Constant signal → zero detail energy (transform runs after window filled)
// -----------------------------------------------------------------------

TEST(WaveletSignalDecomposer, ConstantSignalZeroDetailEnergy) {
    WaveletSignalDecomposer::Config cfg;
    cfg.levels = 3;
    cfg.window = 8;
    WaveletSignalDecomposer wd(cfg);
    for (int i = 0; i < 8; ++i) wd.record(5.0);
    // Haar detail of constant signal is exactly 0.
    EXPECT_NEAR(wd.detail_energy(0), 0.0, 1e-10);
    EXPECT_NEAR(wd.approx_mean(),    5.0, 1e-9);
}

// -----------------------------------------------------------------------
// Alternating ±1 signal → high detail energy at level 0
// -----------------------------------------------------------------------

TEST(WaveletSignalDecomposer, AlternatingSignalHighDetailEnergy) {
    WaveletSignalDecomposer::Config cfg;
    cfg.levels = 3;
    cfg.window = 8;
    cfg.spike_threshold = 0.01;  // low threshold to observe spike
    WaveletSignalDecomposer wd(cfg);
    for (int i = 0; i < 8; ++i) wd.record(i % 2 == 0 ? 1.0 : -1.0);
    EXPECT_GT(wd.detail_energy(0), 0.0);
    EXPECT_GT(wd.max_detail_energy(), 0.0);
}

// -----------------------------------------------------------------------
// Spike callback fires when energy exceeds threshold
// -----------------------------------------------------------------------

TEST(WaveletSignalDecomposer, SpikeCbFires) {
    WaveletSignalDecomposer::Config cfg;
    cfg.levels = 2;
    cfg.window = 4;
    cfg.spike_threshold = 0.01;
    std::atomic<int> fires{0};
    cfg.on_high_freq_spike = [&](int, double) { ++fires; };
    WaveletSignalDecomposer wd(cfg);
    // Alternating high-amplitude signal → large detail coefficients
    for (int i = 0; i < 16; ++i) wd.record(i % 2 == 0 ? 1.0 : -1.0);
    EXPECT_GT(fires.load(), 0);
    EXPECT_GT(wd.spike_events(), 0u);
}

// -----------------------------------------------------------------------
// Approx mean converges to true mean after many records
// -----------------------------------------------------------------------

TEST(WaveletSignalDecomposer, ApproxMeanConverges) {
    WaveletSignalDecomposer::Config cfg;
    cfg.levels = 3;
    cfg.window = 8;
    WaveletSignalDecomposer wd(cfg);
    // Feed values in [0.9, 1.1] — mean ≈ 1.0
    for (int i = 0; i < 32; ++i)
        wd.record(1.0 + 0.1 * (i % 3 == 0 ? 1.0 : -0.5));
    EXPECT_NEAR(wd.approx_mean(), 1.0, 0.3);
}

// -----------------------------------------------------------------------
// detail_energy returns 0 for invalid level
// -----------------------------------------------------------------------

TEST(WaveletSignalDecomposer, DetailEnergyInvalidLevel) {
    WaveletSignalDecomposer wd;
    EXPECT_DOUBLE_EQ(wd.detail_energy(-1), 0.0);
    EXPECT_DOUBLE_EQ(wd.detail_energy(99), 0.0);
}

// -----------------------------------------------------------------------
// reset() clears state
// -----------------------------------------------------------------------

TEST(WaveletSignalDecomposer, ResetClearsState) {
    WaveletSignalDecomposer::Config cfg;
    cfg.levels = 2;
    cfg.window = 4;
    WaveletSignalDecomposer wd(cfg);
    for (int i = 0; i < 8; ++i) wd.record(static_cast<double>(i));
    wd.reset();
    EXPECT_EQ(wd.total_records(), 0u);
    EXPECT_DOUBLE_EQ(wd.max_detail_energy(), 0.0);
    EXPECT_FALSE(wd.is_spiking());
    EXPECT_EQ(wd.spike_events(), 0u);
}

// -----------------------------------------------------------------------
// update_config resets state
// -----------------------------------------------------------------------

TEST(WaveletSignalDecomposer, UpdateConfigResetsState) {
    WaveletSignalDecomposer wd;
    for (int i = 0; i < 10; ++i) wd.record(1.0);
    WaveletSignalDecomposer::Config cfg2;
    cfg2.levels = 2;
    cfg2.window = 4;
    wd.update_config(cfg2);
    EXPECT_EQ(wd.total_records(), 0u);
    EXPECT_DOUBLE_EQ(wd.max_detail_energy(), 0.0);
}

// -----------------------------------------------------------------------
// to_stats_json contains expected keys
// -----------------------------------------------------------------------

TEST(WaveletSignalDecomposer, ToStatsJsonContainsKeys) {
    WaveletSignalDecomposer::Config cfg;
    cfg.levels = 2;
    cfg.window = 4;
    WaveletSignalDecomposer wd(cfg);
    for (int i = 0; i < 8; ++i) wd.record(static_cast<double>(i));
    std::string j = wd.to_stats_json();
    EXPECT_NE(j.find("approx_mean"),       std::string::npos);
    EXPECT_NE(j.find("max_detail_energy"), std::string::npos);
    EXPECT_NE(j.find("is_spiking"),        std::string::npos);
    EXPECT_NE(j.find("detail_energies"),   std::string::npos);
    EXPECT_NE(j.find("total_records"),     std::string::npos);
}

// -----------------------------------------------------------------------
// Concurrent record() is thread-safe
// -----------------------------------------------------------------------

TEST(WaveletSignalDecomposer, ConcurrentRecordSafe) {
    WaveletSignalDecomposer::Config cfg;
    cfg.levels = 2;
    cfg.window = 4;
    WaveletSignalDecomposer wd(cfg);
    std::vector<std::thread> threads;
    for (int t = 0; t < 4; ++t) {
        threads.emplace_back([&, t] {
            for (int i = 0; i < 250; ++i)
                wd.record(static_cast<double>(t % 2 == 0 ? i : -i) * 0.01);
        });
    }
    for (auto& th : threads) th.join();
    EXPECT_EQ(wd.total_records(), 1000u);
    EXPECT_TRUE(std::isfinite(wd.approx_mean()));
}

} // namespace

#else

TEST(WaveletSignalDecomposer, DisabledAtBuildTime) { SUCCEED(); }

#endif  // LLMQUANT_WAVELET_DECOMPOSER_ENABLED
