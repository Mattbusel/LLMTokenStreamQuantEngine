#include <gtest/gtest.h>

#ifdef LLMQUANT_ECHO_SUPPRESSOR_ENABLED

#include "SignalEchoSuppressor.h"

#include <atomic>
#include <thread>
#include <vector>

using namespace llmquant;

namespace {

TEST(SignalEchoSuppressor, DefaultState) {
    SignalEchoSuppressor s;
    EXPECT_NEAR(s.echo_rate(), 0.0, 1e-9);
    EXPECT_FALSE(s.is_echoing());
    EXPECT_EQ(s.total_signals(), 0u);
    EXPECT_EQ(s.echo_count(), 0u);
    EXPECT_EQ(s.echo_events(), 0u);
}

TEST(SignalEchoSuppressor, FirstSignalIsNotEcho) {
    SignalEchoSuppressor s;
    bool echo = s.record(0.5);
    EXPECT_FALSE(echo);
    EXPECT_EQ(s.total_signals(), 1u);
    EXPECT_EQ(s.echo_count(), 0u);
}

TEST(SignalEchoSuppressor, IdenticalSignalsAreEchoes) {
    SignalEchoSuppressor::Config cfg;
    cfg.echo_threshold = 1e-3;
    SignalEchoSuppressor s(cfg);
    s.record(0.5);
    bool echo = s.record(0.5);
    EXPECT_TRUE(echo);
    EXPECT_EQ(s.echo_count(), 1u);
}

TEST(SignalEchoSuppressor, NearIdenticalSignalsAreEchoes) {
    SignalEchoSuppressor::Config cfg;
    cfg.echo_threshold = 1e-3;
    SignalEchoSuppressor s(cfg);
    s.record(0.5);
    bool echo = s.record(0.5 + 5e-4);  // within threshold
    EXPECT_TRUE(echo);
}

TEST(SignalEchoSuppressor, DifferentSignalIsNotEcho) {
    SignalEchoSuppressor::Config cfg;
    cfg.echo_threshold = 1e-3;
    SignalEchoSuppressor s(cfg);
    s.record(0.5);
    bool echo = s.record(0.8);  // large jump
    EXPECT_FALSE(echo);
    EXPECT_EQ(s.echo_count(), 0u);
}

TEST(SignalEchoSuppressor, EchoRateBounded) {
    SignalEchoSuppressor::Config cfg;
    cfg.echo_threshold = 1e-3;
    cfg.window         = 16;
    SignalEchoSuppressor s(cfg);
    for (int i = 0; i < 32; ++i) s.record(0.5);  // all echoes after first
    double r = s.echo_rate();
    EXPECT_GE(r, 0.0);
    EXPECT_LE(r, 1.0 + 1e-9);
}

TEST(SignalEchoSuppressor, HighEchoRateTriggersCallback) {
    SignalEchoSuppressor::Config cfg;
    cfg.echo_threshold     = 1e-3;
    cfg.window             = 8;
    cfg.echo_rate_threshold = 0.5;
    std::atomic<int> fires{0};
    cfg.on_echo_detected = [&](double) { ++fires; };
    SignalEchoSuppressor s(cfg);
    s.record(1.0);  // seed
    for (int i = 0; i < 16; ++i) s.record(1.0);  // all echoes
    EXPECT_GE(fires.load(), 1);
    EXPECT_GT(s.echo_events(), 0u);
    EXPECT_TRUE(s.is_echoing());
}

TEST(SignalEchoSuppressor, EchoClearedCallbackFires) {
    SignalEchoSuppressor::Config cfg;
    cfg.echo_threshold     = 1e-3;
    cfg.window             = 8;
    cfg.echo_rate_threshold = 0.5;
    cfg.clear_hysteresis   = 0.9;
    std::atomic<int> cleared{0};
    cfg.on_echo_cleared = [&](double) { ++cleared; };
    SignalEchoSuppressor s(cfg);
    // Enter echo state.
    s.record(1.0);
    for (int i = 0; i < 8; ++i) s.record(1.0);
    // Leave echo state with distinct signals.
    for (int i = 0; i < 16; ++i) s.record(static_cast<double>(i) * 0.1);
    EXPECT_GE(cleared.load(), 1);
}

TEST(SignalEchoSuppressor, TotalSignalsIncrements) {
    SignalEchoSuppressor s;
    s.record(1.0);
    s.record(2.0);
    s.record(3.0);
    EXPECT_EQ(s.total_signals(), 3u);
}

TEST(SignalEchoSuppressor, ResetClearsState) {
    SignalEchoSuppressor s;
    s.record(1.0);
    s.record(1.0);
    s.reset();
    EXPECT_EQ(s.total_signals(), 0u);
    EXPECT_EQ(s.echo_count(), 0u);
    EXPECT_NEAR(s.echo_rate(), 0.0, 1e-9);
    EXPECT_FALSE(s.is_echoing());
}

TEST(SignalEchoSuppressor, UpdateConfigResetsState) {
    SignalEchoSuppressor s;
    s.record(1.0);
    s.record(1.0);
    SignalEchoSuppressor::Config cfg2;
    cfg2.window = 8;
    s.update_config(cfg2);
    EXPECT_EQ(s.total_signals(), 0u);
}

TEST(SignalEchoSuppressor, StatsJsonContainsFields) {
    SignalEchoSuppressor s;
    s.record(1.0);
    s.record(1.0);
    std::string j = s.to_stats_json();
    EXPECT_NE(j.find("echo_rate"),     std::string::npos);
    EXPECT_NE(j.find("is_echoing"),    std::string::npos);
    EXPECT_NE(j.find("total_signals"), std::string::npos);
    EXPECT_NE(j.find("echo_count"),    std::string::npos);
    EXPECT_NE(j.find("echo_events"),   std::string::npos);
}

TEST(SignalEchoSuppressor, ConcurrentRecordSafe) {
    SignalEchoSuppressor::Config cfg;
    cfg.window = 32;
    SignalEchoSuppressor s(cfg);
    std::atomic<bool> go{false};
    auto worker = [&](int id) {
        while (!go.load()) {}
        for (int i = 0; i < 250; ++i)
            s.record(id % 2 == 0 ? 1.0 : static_cast<double>(i) * 0.01);
    };
    std::vector<std::thread> threads;
    for (int t = 0; t < 4; ++t) threads.emplace_back(worker, t);
    go.store(true);
    for (auto& t : threads) t.join();
    EXPECT_EQ(s.total_signals(), 1000u);
    double r = s.echo_rate();
    EXPECT_GE(r, 0.0);
    EXPECT_LE(r, 1.0 + 1e-9);
}

}  // namespace

#else

TEST(SignalEchoSuppressor, DisabledAtBuildTime) { SUCCEED(); }

#endif  // LLMQUANT_ECHO_SUPPRESSOR_ENABLED
