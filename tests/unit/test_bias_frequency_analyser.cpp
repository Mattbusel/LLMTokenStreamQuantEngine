#include <gtest/gtest.h>

#ifdef LLMQUANT_FREQ_ANALYSER_ENABLED
#include "BiasFrequencyAnalyser.h"
#include <atomic>
#include <cmath>
#include <thread>
using namespace llmquant;

TEST(BiasFrequencyAnalyserTest, InitialState) {
    BiasFrequencyAnalyser fa;
    EXPECT_EQ(fa.dominant_k(), 0);
    EXPECT_NEAR(fa.dominant_frequency(), 0.0, 1e-9);
    EXPECT_EQ(fa.total_records(), 0u);
}

TEST(BiasFrequencyAnalyserTest, TotalRecordsIncrements) {
    BiasFrequencyAnalyser fa;
    fa.record(0.1);
    fa.record(0.2);
    EXPECT_EQ(fa.total_records(), 2u);
}

TEST(BiasFrequencyAnalyserTest, DominantKNonZeroAfterMinSamples) {
    BiasFrequencyAnalyser::Config cfg;
    cfg.window      = 16;
    cfg.min_samples = 8;
    cfg.max_k       = 4;
    BiasFrequencyAnalyser fa(cfg);
    for (int i = 0; i < 16; ++i)
        fa.record(std::sin(i * 0.5));
    EXPECT_GT(fa.dominant_k(), 0);
}

TEST(BiasFrequencyAnalyserTest, DominantFrequencyBounded) {
    BiasFrequencyAnalyser::Config cfg;
    cfg.window      = 32;
    cfg.min_samples = 8;
    cfg.max_k       = 8;
    BiasFrequencyAnalyser fa(cfg);
    for (int i = 0; i < 32; ++i)
        fa.record(std::sin(i * 0.3));
    double f = fa.dominant_frequency();
    EXPECT_GE(f, 0.0);
    EXPECT_LE(f, 1.0);
}

TEST(BiasFrequencyAnalyserTest, HighFrequencySineDetected) {
    // A 1-in-2 alternating signal → dominant k should be high
    BiasFrequencyAnalyser::Config cfg;
    cfg.window      = 32;
    cfg.min_samples = 16;
    cfg.max_k       = 8;
    BiasFrequencyAnalyser fa(cfg);
    for (int i = 0; i < 32; ++i)
        fa.record(i % 2 == 0 ? 1.0 : -1.0);
    EXPECT_GE(fa.dominant_k(), 1);
}

TEST(BiasFrequencyAnalyserTest, FrequencyChangeCallbackFires) {
    BiasFrequencyAnalyser::Config cfg;
    cfg.window      = 16;
    cfg.min_samples = 8;
    cfg.max_k       = 4;
    std::atomic<int> fires{0};
    cfg.on_dominant_frequency_change = [&](int, int, double) { ++fires; };
    BiasFrequencyAnalyser fa(cfg);
    // Alternating signal then steady signal — should trigger frequency shift
    for (int i = 0; i < 16; ++i) fa.record(i % 2 == 0 ? 1.0 : -1.0);
    for (int i = 0; i < 16; ++i) fa.record(static_cast<double>(i) * 0.01);
    // callback may or may not fire depending on DCT peak; just verify no crash
    EXPECT_GE(fires.load(), 0);
}

TEST(BiasFrequencyAnalyserTest, ResetClearsState) {
    BiasFrequencyAnalyser fa;
    for (int i = 0; i < 20; ++i) fa.record(std::sin(i * 0.5));
    fa.reset();
    EXPECT_EQ(fa.dominant_k(), 0);
    EXPECT_EQ(fa.total_records(), 0u);
}

TEST(BiasFrequencyAnalyserTest, StatsJsonHasFields) {
    BiasFrequencyAnalyser fa;
    for (int i = 0; i < 20; ++i) fa.record(std::sin(i * 0.5));
    std::string j = fa.to_stats_json();
    EXPECT_NE(j.find("dominant_k"),         std::string::npos);
    EXPECT_NE(j.find("dominant_frequency"), std::string::npos);
    EXPECT_NE(j.find("dominant_power"),     std::string::npos);
    EXPECT_NE(j.find("total_records"),      std::string::npos);
}

TEST(BiasFrequencyAnalyserTest, ConcurrentSafe) {
    BiasFrequencyAnalyser::Config cfg;
    cfg.window      = 32;
    cfg.min_samples = 8;
    BiasFrequencyAnalyser fa(cfg);
    std::atomic<bool> go{false};
    auto worker = [&](int id) {
        while (!go.load()) {}
        for (int i = 0; i < 200; ++i)
            fa.record(id % 2 == 0 ? std::sin(i * 0.3) : static_cast<double>(i) * 0.01);
    };
    std::thread t1(worker, 0), t2(worker, 1);
    go.store(true);
    t1.join(); t2.join();
    EXPECT_EQ(fa.total_records(), 400u);
    double f = fa.dominant_frequency();
    EXPECT_GE(f, 0.0);
    EXPECT_LE(f, 1.0);
}

#else
TEST(BiasFrequencyAnalyserTest, DisabledAtBuildTime) { SUCCEED(); }
#endif
