#include <gtest/gtest.h>

#ifdef LLMQUANT_AUTOREGRESSOR_ENABLED
#include "SignalAutoregressor.h"
#include <atomic>
#include <cmath>
#include <thread>
using namespace llmquant;

TEST(SignalAutoregressor, DefaultState) {
    SignalAutoregressor ar;
    EXPECT_EQ(ar.total_records(), 0u);
    EXPECT_EQ(ar.spike_events(), 0u);
    EXPECT_NEAR(ar.last_prediction_error(), 0.0, 1e-9);
}

TEST(SignalAutoregressor, TotalRecordsIncrements) {
    SignalAutoregressor ar;
    for (int i = 0; i < 15; ++i) ar.record(static_cast<double>(i) * 0.05);
    EXPECT_EQ(ar.total_records(), 15u);
}

TEST(SignalAutoregressor, ConstantSeriesLowError) {
    SignalAutoregressor::Config cfg;
    cfg.order  = 2;
    cfg.lambda = 0.99;
    SignalAutoregressor ar(cfg);
    for (int i = 0; i < 50; ++i) ar.record(0.5);
    EXPECT_LT(std::abs(ar.last_prediction_error()), 0.1);
}

TEST(SignalAutoregressor, SpikeCallbackFires) {
    SignalAutoregressor::Config cfg;
    cfg.order           = 2;
    cfg.error_threshold = 0.1;
    std::atomic<int> fires{0};
    cfg.on_prediction_error_spike = [&](double) { ++fires; };
    SignalAutoregressor ar(cfg);
    for (int i = 0; i < 20; ++i) ar.record(0.0);
    for (int i = 0; i < 5;  ++i) ar.record(1.0);
    EXPECT_GE(fires.load(), 1);
    EXPECT_GE(ar.spike_events(), 1u);
}

TEST(SignalAutoregressor, ResetClearsState) {
    SignalAutoregressor ar;
    for (int i = 0; i < 20; ++i) ar.record(0.5);
    ar.reset();
    EXPECT_EQ(ar.total_records(), 0u);
    EXPECT_EQ(ar.spike_events(), 0u);
    EXPECT_NEAR(ar.last_prediction_error(), 0.0, 1e-9);
}

TEST(SignalAutoregressor, UpdateConfigResetsState) {
    SignalAutoregressor ar;
    for (int i = 0; i < 20; ++i) ar.record(0.5);
    SignalAutoregressor::Config cfg2;
    cfg2.order = 3;
    ar.update_config(cfg2);
    EXPECT_EQ(ar.total_records(), 0u);
}

TEST(SignalAutoregressor, StatsJsonContainsFields) {
    SignalAutoregressor ar;
    for (int i = 0; i < 10; ++i) ar.record(0.1);
    std::string j = ar.to_stats_json();
    EXPECT_NE(j.find("last_prediction"),       std::string::npos);
    EXPECT_NE(j.find("last_prediction_error"), std::string::npos);
    EXPECT_NE(j.find("spike_events"),          std::string::npos);
    EXPECT_NE(j.find("total_records"),         std::string::npos);
}

TEST(SignalAutoregressor, ConcurrentRecordSafe) {
    SignalAutoregressor ar;
    std::atomic<bool> go{false};
    auto w0 = [&] { while (!go) {} for (int i = 0; i < 150; ++i) ar.record(static_cast<double>(i % 10) * 0.1); };
    auto w1 = [&] { while (!go) {} for (int i = 0; i < 150; ++i) ar.record(static_cast<double>(i % 5) * 0.2 - 0.4); };
    std::thread t0(w0), t1(w1);
    go = true;
    t0.join(); t1.join();
    EXPECT_EQ(ar.total_records(), 300u);
}

#else
TEST(SignalAutoregressor, DisabledAtBuildTime) { SUCCEED(); }
#endif
