#include <gtest/gtest.h>

#ifdef LLMQUANT_TOKEN_CLOCK_RECALIBRATOR_ENABLED
#include "TokenClockRecalibrator.h"
#include <atomic>
#include <thread>
using namespace llmquant;

TEST(TokenClockRecalibratorTest, InitialState) {
    TokenClockRecalibrator tcr;
    EXPECT_NEAR(tcr.rate_hz(), 0.0, 1e-9);
    EXPECT_NEAR(tcr.budget_scale(), 1.0, 1e-9);
    EXPECT_EQ(tcr.token_count(), 0u);
}

TEST(TokenClockRecalibratorTest, RateEstimatedAfterTokens) {
    TokenClockRecalibrator::Config cfg;
    cfg.target_rate_hz = 10.0;
    TokenClockRecalibrator tcr(cfg);

    // Feed tokens at 10 Hz (100 ms gaps).
    uint64_t ts = 0;
    for (int i = 0; i < 20; ++i) {
        ts += 100'000'000ULL; // 100 ms in ns
        tcr.record_token(ts);
    }
    // Rate should be near 10 Hz.
    EXPECT_GT(tcr.rate_hz(), 5.0);
    EXPECT_LT(tcr.rate_hz(), 20.0);
}

TEST(TokenClockRecalibratorTest, BudgetScaleScalesWithRate) {
    TokenClockRecalibrator::Config cfg;
    cfg.target_rate_hz = 10.0;
    cfg.min_scale = 0.1;
    cfg.max_scale = 10.0;
    TokenClockRecalibrator tcr(cfg);

    // Feed tokens much faster than target: 100 Hz (10 ms gaps).
    uint64_t ts = 0;
    for (int i = 0; i < 30; ++i) {
        ts += 10'000'000ULL; // 10 ms
        tcr.record_token(ts);
    }
    // Rate ≈ 100 Hz, target = 10, so scale should be < 1 (tight budget).
    EXPECT_LT(tcr.budget_scale(), 1.0);
}

TEST(TokenClockRecalibratorTest, ScaleClampedToMinMax) {
    TokenClockRecalibrator::Config cfg;
    cfg.target_rate_hz = 10.0;
    cfg.min_scale = 0.5;
    cfg.max_scale = 2.0;
    TokenClockRecalibrator tcr(cfg);

    // Extremely fast tokens → would want scale << 0.5.
    uint64_t ts = 0;
    for (int i = 0; i < 30; ++i) {
        ts += 1'000'000ULL; // 1 ms
        tcr.record_token(ts);
    }
    EXPECT_GE(tcr.budget_scale(), cfg.min_scale - 1e-9);

    // Reset and try extremely slow tokens → scale clamped to max_scale.
    tcr.reset();
    ts = 0;
    for (int i = 0; i < 30; ++i) {
        ts += 10'000'000'000ULL; // 10 s
        tcr.record_token(ts);
    }
    EXPECT_LE(tcr.budget_scale(), cfg.max_scale + 1e-9);
}

TEST(TokenClockRecalibratorTest, RateChangeCallbackFires) {
    TokenClockRecalibrator::Config cfg;
    cfg.target_rate_hz = 10.0;
    cfg.rate_change_threshold = 0.01; // very sensitive
    std::atomic<int> fires{0};
    cfg.on_rate_change = [&](double, double) { ++fires; };
    TokenClockRecalibrator tcr(cfg);

    uint64_t ts = 0;
    // First batch: slow.
    for (int i = 0; i < 15; ++i) {
        ts += 100'000'000ULL;
        tcr.record_token(ts);
    }
    // Second batch: fast.
    for (int i = 0; i < 15; ++i) {
        ts += 5'000'000ULL;
        tcr.record_token(ts);
    }
    EXPECT_GE(fires.load(), 1);
}

TEST(TokenClockRecalibratorTest, JitterCvNonNegative) {
    TokenClockRecalibrator tcr;
    uint64_t ts = 0;
    for (int i = 0; i < 20; ++i) {
        ts += static_cast<uint64_t>(10'000'000 + (i % 3) * 5'000'000);
        tcr.record_token(ts);
    }
    EXPECT_GE(tcr.jitter_cv(), 0.0);
}

TEST(TokenClockRecalibratorTest, ResetClearsState) {
    TokenClockRecalibrator tcr;
    uint64_t ts = 0;
    for (int i = 0; i < 10; ++i) { ts += 50'000'000ULL; tcr.record_token(ts); }
    EXPECT_GT(tcr.token_count(), 0u);
    tcr.reset();
    EXPECT_EQ(tcr.token_count(), 0u);
    EXPECT_NEAR(tcr.rate_hz(), 0.0, 1e-9);
}

TEST(TokenClockRecalibratorTest, StatsJsonNonEmpty) {
    TokenClockRecalibrator tcr;
    tcr.record_token(1'000'000'000ULL);
    std::string j = tcr.to_stats_json();
    EXPECT_NE(j.find("rate_hz"), std::string::npos);
    EXPECT_NE(j.find("budget_scale"), std::string::npos);
}

TEST(TokenClockRecalibratorTest, ConcurrentTokensSafe) {
    TokenClockRecalibrator tcr;
    std::atomic<bool> go{false};
    std::atomic<uint64_t> ts_counter{0};
    auto worker = [&] {
        while (!go.load()) {}
        for (int i = 0; i < 200; ++i) {
            uint64_t t = ts_counter.fetch_add(50'000'000ULL);
            tcr.record_token(t);
        }
    };
    std::thread t1(worker), t2(worker);
    go.store(true);
    t1.join(); t2.join();
    EXPECT_EQ(tcr.token_count(), 400u);
}

#else
TEST(TokenClockRecalibratorTest, DisabledAtBuildTime) { SUCCEED(); }
#endif
