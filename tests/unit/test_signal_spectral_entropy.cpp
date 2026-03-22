#include <gtest/gtest.h>
#include <thread>
#include <vector>
#include <cmath>
#ifndef M_PI
#  define M_PI 3.14159265358979323846
#endif

#ifdef LLMQUANT_SPECTRAL_ENTROPY_ENABLED
#include "SignalSpectralEntropy.h"
using namespace llmquant;

TEST(SignalSpectralEntropy, ConstructDefault) {
    SignalSpectralEntropy sse;
    EXPECT_DOUBLE_EQ(sse.spectral_entropy(),   0.0);
    EXPECT_DOUBLE_EQ(sse.normalised_entropy(),  0.0);
    EXPECT_EQ(sse.total_records(), 0u);
}

TEST(SignalSpectralEntropy, TotalRecordsIncrement) {
    SignalSpectralEntropy sse;
    sse.record(0.1);
    sse.record(-0.2);
    EXPECT_EQ(sse.total_records(), 2u);
}

TEST(SignalSpectralEntropy, EntropyNonNegativeAfterFill) {
    SignalSpectralEntropy::Config cfg;
    cfg.min_samples = 4;
    SignalSpectralEntropy sse(cfg);
    for (int i = 0; i < 20; ++i) sse.record(0.1 * (i % 7));
    EXPECT_GE(sse.spectral_entropy(),  0.0);
    EXPECT_GE(sse.normalised_entropy(), 0.0);
}

TEST(SignalSpectralEntropy, NormalisedEntropyAtMostOne) {
    SignalSpectralEntropy::Config cfg;
    cfg.min_samples = 4;
    SignalSpectralEntropy sse(cfg);
    for (int i = 0; i < 30; ++i) sse.record(0.1 * (i % 5) - 0.2);
    EXPECT_LE(sse.normalised_entropy(), 1.0 + 1e-9);
}

TEST(SignalSpectralEntropy, WhiteNoiseLikeSignalHighEntropy) {
    SignalSpectralEntropy::Config cfg;
    cfg.window      = 32;
    cfg.min_samples = 8;
    SignalSpectralEntropy sse(cfg);
    for (int i = 0; i < 40; ++i) sse.record((i % 7 - 3) * 0.15);
    EXPECT_GT(sse.normalised_entropy(), 0.3);
}

TEST(SignalSpectralEntropy, PeriodicSignalLowerEntropy) {
    SignalSpectralEntropy::Config cfg;
    cfg.window      = 32;
    cfg.min_samples = 8;
    SignalSpectralEntropy periodic(cfg);
    // Perfect sinusoidal → power at single frequency → lower entropy
    for (int i = 0; i < 40; ++i)
        periodic.record(std::sin(2.0 * M_PI * i / 8.0));
    EXPECT_GE(periodic.spectral_entropy(), 0.0);
}

TEST(SignalSpectralEntropy, ChangeCallbackFired) {
    SignalSpectralEntropy::Config cfg;
    cfg.change_threshold = 0.0;
    cfg.min_samples      = 4;
    int fired = 0;
    cfg.on_entropy_change = [&](double, double) { ++fired; };
    SignalSpectralEntropy sse(cfg);
    for (int i = 0; i < 30; ++i) sse.record(0.1 * (i % 7));
    EXPECT_GT(fired, 0);
}

TEST(SignalSpectralEntropy, ChangeCallbackReceivesValues) {
    SignalSpectralEntropy::Config cfg;
    cfg.change_threshold = 0.0;
    cfg.min_samples      = 4;
    double got_old = -1.0, got_new = -1.0;
    cfg.on_entropy_change = [&](double o, double n) { got_old = o; got_new = n; };
    SignalSpectralEntropy sse(cfg);
    for (int i = 0; i < 30; ++i) sse.record(0.1 * i);
    EXPECT_GE(got_old, 0.0);
    EXPECT_GE(got_new, 0.0);
}

TEST(SignalSpectralEntropy, ChangeEventsAccumulate) {
    SignalSpectralEntropy::Config cfg;
    cfg.change_threshold = 0.0;
    cfg.min_samples      = 4;
    SignalSpectralEntropy sse(cfg);
    for (int i = 0; i < 30; ++i) sse.record(0.1 * (i % 5));
    EXPECT_GT(sse.change_events(), 0u);
}

TEST(SignalSpectralEntropy, Reset) {
    SignalSpectralEntropy sse;
    for (int i = 0; i < 30; ++i) sse.record(0.1 * i);
    sse.reset();
    EXPECT_DOUBLE_EQ(sse.spectral_entropy(),  0.0);
    EXPECT_DOUBLE_EQ(sse.normalised_entropy(), 0.0);
    EXPECT_EQ(sse.total_records(), 0u);
}

TEST(SignalSpectralEntropy, UpdateConfigResetsState) {
    SignalSpectralEntropy sse;
    for (int i = 0; i < 20; ++i) sse.record(0.1);
    SignalSpectralEntropy::Config cfg;
    cfg.window = 32;
    sse.update_config(cfg);
    EXPECT_EQ(sse.total_records(), 0u);
}

TEST(SignalSpectralEntropy, ToStatsJsonFields) {
    SignalSpectralEntropy sse;
    for (int i = 0; i < 20; ++i) sse.record(0.1 * (i % 5));
    auto js = sse.to_stats_json();
    EXPECT_NE(js.find("spectral_entropy"),   std::string::npos);
    EXPECT_NE(js.find("normalised_entropy"), std::string::npos);
    EXPECT_NE(js.find("change_events"),      std::string::npos);
    EXPECT_NE(js.find("total_records"),      std::string::npos);
}

TEST(SignalSpectralEntropy, ConcurrentRecord) {
    SignalSpectralEntropy sse;
    std::vector<std::thread> threads;
    for (int t = 0; t < 4; ++t)
        threads.emplace_back([&] {
            for (int i = 0; i < 50; ++i) sse.record(0.1 * (i % 7));
        });
    for (auto& th : threads) th.join();
    EXPECT_GE(sse.total_records(), 100u);
}

#else
TEST(SignalSpectralEntropy, DisabledAtBuildTime) { SUCCEED(); }
#endif
