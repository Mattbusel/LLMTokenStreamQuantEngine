#include <gtest/gtest.h>
#include <thread>
#include <vector>

#ifdef LLMQUANT_SIGNAL_COMPRESSOR_ENABLED
#include "SignalCompressor.h"
using namespace llmquant;

TEST(SignalCompressor, ConstructDefault) {
    SignalCompressor sc;
    EXPECT_EQ(sc.lz_complexity(), 0);
    EXPECT_DOUBLE_EQ(sc.normalised_complexity(), 0.0);
    EXPECT_EQ(sc.total_records(), 0u);
}

TEST(SignalCompressor, TotalRecordsIncrement) {
    SignalCompressor sc;
    sc.record(0.1);
    sc.record(-0.2);
    sc.record(0.3);
    EXPECT_EQ(sc.total_records(), 3u);
}

TEST(SignalCompressor, BelowMinSamplesNoComputation) {
    SignalCompressor::Config cfg;
    cfg.min_samples = 32;
    SignalCompressor sc(cfg);
    for (int i = 0; i < 10; ++i) sc.record(0.1 * i);
    EXPECT_EQ(sc.lz_complexity(), 0);
    EXPECT_DOUBLE_EQ(sc.normalised_complexity(), 0.0);
}

TEST(SignalCompressor, LZComplexityPositiveAfterMinSamples) {
    SignalCompressor sc;
    for (int i = 0; i < 32; ++i) sc.record((i % 4 == 0) ? 0.5 : -0.5);
    EXPECT_GE(sc.lz_complexity(), 1);
}

TEST(SignalCompressor, NormalisedComplexityInRange) {
    SignalCompressor sc;
    for (int i = 0; i < 40; ++i) sc.record(0.1 * (i % 5));
    EXPECT_GE(sc.normalised_complexity(), 0.0);
}

TEST(SignalCompressor, ConstantSignalLowComplexity) {
    SignalCompressor sc;
    // Constant bias → only one phrase ever needed
    for (int i = 0; i < 64; ++i) sc.record(0.5);
    EXPECT_LE(sc.lz_complexity(), 4);
}

TEST(SignalCompressor, AlternatingSignalMorePhrases) {
    SignalCompressor::Config cfg;
    cfg.window      = 64;
    cfg.n_symbols   = 4;
    cfg.min_samples = 16;
    SignalCompressor sc(cfg);
    for (int i = 0; i < 64; ++i)
        sc.record((i % 4) * 0.5 - 0.75);
    EXPECT_GE(sc.lz_complexity(), 2);
}

TEST(SignalCompressor, CallbackFiredOnChange) {
    SignalCompressor::Config cfg;
    cfg.change_threshold = 0.0;   // fire on any norm change
    cfg.min_samples      = 4;
    int fired = 0;
    cfg.on_complexity_change = [&](double, double) { ++fired; };
    SignalCompressor sc(cfg);
    for (int i = 0; i < 32; ++i) sc.record(0.1 * (i % 7));
    EXPECT_GT(fired, 0);
}

TEST(SignalCompressor, CallbackReceivesOldAndNew) {
    SignalCompressor::Config cfg;
    cfg.change_threshold = 0.0;
    cfg.min_samples      = 4;
    double got_old = -99.0, got_new = -99.0;
    cfg.on_complexity_change = [&](double o, double n) {
        got_old = o;
        got_new = n;
    };
    SignalCompressor sc(cfg);
    for (int i = 0; i < 32; ++i) sc.record(0.1 * (i % 5));
    EXPECT_GE(got_old, 0.0);
    EXPECT_GE(got_new, 0.0);
}

TEST(SignalCompressor, Reset) {
    SignalCompressor sc;
    for (int i = 0; i < 32; ++i) sc.record(0.1 * i);
    sc.reset();
    EXPECT_EQ(sc.lz_complexity(), 0);
    EXPECT_DOUBLE_EQ(sc.normalised_complexity(), 0.0);
    EXPECT_EQ(sc.total_records(), 0u);
}

TEST(SignalCompressor, UpdateConfigResetsState) {
    SignalCompressor sc;
    for (int i = 0; i < 32; ++i) sc.record(0.1);
    SignalCompressor::Config cfg;
    cfg.window = 32;
    sc.update_config(cfg);
    EXPECT_EQ(sc.total_records(), 0u);
    EXPECT_EQ(sc.lz_complexity(), 0);
}

TEST(SignalCompressor, ToStatsJsonFields) {
    SignalCompressor sc;
    for (int i = 0; i < 20; ++i) sc.record(0.1 * (i % 5));
    auto js = sc.to_stats_json();
    EXPECT_NE(js.find("lz_complexity"),         std::string::npos);
    EXPECT_NE(js.find("normalised_complexity"),  std::string::npos);
    EXPECT_NE(js.find("total_records"),          std::string::npos);
}

TEST(SignalCompressor, ConcurrentRecord) {
    SignalCompressor sc;
    std::vector<std::thread> threads;
    for (int t = 0; t < 4; ++t)
        threads.emplace_back([&] {
            for (int i = 0; i < 50; ++i) sc.record(0.1 * (i % 8));
        });
    for (auto& th : threads) th.join();
    EXPECT_GE(sc.total_records(), 100u);
}

TEST(SignalCompressor, OutOfRangeBiasClampedToBin) {
    SignalCompressor sc;
    for (int i = 0; i < 20; ++i) sc.record(100.0);   // far above range_max
    for (int i = 0; i < 20; ++i) sc.record(-100.0);  // far below range_min
    // Should not crash or produce negative complexity
    EXPECT_GE(sc.lz_complexity(), 0);
}

#else
TEST(SignalCompressor, DisabledAtBuildTime) { SUCCEED(); }
#endif
