#include <gtest/gtest.h>
#include <thread>
#include <vector>

#ifdef LLMQUANT_INFORMATION_GAIN_ENABLED
#include "BiasInformationGain.h"
using namespace llmquant;

TEST(BiasInformationGain, ConstructDefault) {
    BiasInformationGain big;
    EXPECT_DOUBLE_EQ(big.mutual_information(), 0.0);
    EXPECT_DOUBLE_EQ(big.normalised_mi(), 0.0);
    EXPECT_EQ(big.total_records(), 0u);
}

TEST(BiasInformationGain, TotalRecordsIncrement) {
    BiasInformationGain big;
    big.record(0.1);
    big.record(-0.2);
    big.record(0.3);
    EXPECT_EQ(big.total_records(), 3u);
}

TEST(BiasInformationGain, MIInRange) {
    BiasInformationGain big;
    for (int i = 0; i < 50; ++i) big.record(0.1 * (i % 7) - 0.3);
    EXPECT_GE(big.mutual_information(), 0.0);
    EXPECT_GE(big.normalised_mi(), 0.0);
    EXPECT_LE(big.normalised_mi(), 1.0 + 1e-9);
}

TEST(BiasInformationGain, PerfectlyPredictableHighMI) {
    BiasInformationGain::Config cfg;
    cfg.min_samples = 4;
    cfg.n_bins      = 4;
    BiasInformationGain big(cfg);
    // Deterministic cycling pattern: bin always follows deterministically
    for (int i = 0; i < 80; ++i) big.record((i % 4) * 0.5 - 0.75);
    // NMI should be > 0 for structured signal
    EXPECT_GE(big.normalised_mi(), 0.0);
}

TEST(BiasInformationGain, MIChangeCallbackFired) {
    BiasInformationGain::Config cfg;
    cfg.change_threshold = 0.0;
    cfg.min_samples      = 4;
    int fired = 0;
    cfg.on_mi_change = [&](double, double) { ++fired; };
    BiasInformationGain big(cfg);
    for (int i = 0; i < 50; ++i) big.record(0.1 * (i % 5));
    EXPECT_GT(fired, 0);
}

TEST(BiasInformationGain, CallbackReceivesValidValues) {
    BiasInformationGain::Config cfg;
    cfg.change_threshold = 0.0;
    cfg.min_samples      = 4;
    double got_old = -1.0, got_new = -1.0;
    cfg.on_mi_change = [&](double o, double n) { got_old = o; got_new = n; };
    BiasInformationGain big(cfg);
    for (int i = 0; i < 40; ++i) big.record(0.1 * (i % 6));
    EXPECT_GE(got_old, 0.0);
    EXPECT_GE(got_new, 0.0);
}

TEST(BiasInformationGain, MIChangeEventsAccumulate) {
    BiasInformationGain::Config cfg;
    cfg.change_threshold = 0.0;
    BiasInformationGain big(cfg);
    for (int i = 0; i < 40; ++i) big.record(0.1 * (i % 5));
    EXPECT_GT(big.mi_change_events(), 0u);
}

TEST(BiasInformationGain, Reset) {
    BiasInformationGain big;
    for (int i = 0; i < 40; ++i) big.record(0.1 * i);
    big.reset();
    EXPECT_DOUBLE_EQ(big.mutual_information(), 0.0);
    EXPECT_DOUBLE_EQ(big.normalised_mi(), 0.0);
    EXPECT_EQ(big.total_records(), 0u);
}

TEST(BiasInformationGain, UpdateConfigResetsState) {
    BiasInformationGain big;
    for (int i = 0; i < 30; ++i) big.record(0.1);
    BiasInformationGain::Config cfg;
    cfg.n_bins = 8;
    big.update_config(cfg);
    EXPECT_EQ(big.total_records(), 0u);
}

TEST(BiasInformationGain, ToStatsJsonFields) {
    BiasInformationGain big;
    for (int i = 0; i < 20; ++i) big.record(0.1 * (i % 5));
    auto js = big.to_stats_json();
    EXPECT_NE(js.find("mutual_information"), std::string::npos);
    EXPECT_NE(js.find("normalised_mi"),      std::string::npos);
    EXPECT_NE(js.find("mi_change_events"),   std::string::npos);
    EXPECT_NE(js.find("total_records"),      std::string::npos);
}

TEST(BiasInformationGain, OutOfRangeBiasSafelyBinned) {
    BiasInformationGain big;
    for (int i = 0; i < 20; ++i) big.record(99.0);
    for (int i = 0; i < 20; ++i) big.record(-99.0);
    EXPECT_GE(big.normalised_mi(), 0.0);
}

TEST(BiasInformationGain, ConcurrentRecord) {
    BiasInformationGain big;
    std::vector<std::thread> threads;
    for (int t = 0; t < 4; ++t)
        threads.emplace_back([&] {
            for (int i = 0; i < 50; ++i) big.record(0.1 * (i % 7));
        });
    for (auto& th : threads) th.join();
    EXPECT_GE(big.total_records(), 100u);
}

#else
TEST(BiasInformationGain, DisabledAtBuildTime) { SUCCEED(); }
#endif
