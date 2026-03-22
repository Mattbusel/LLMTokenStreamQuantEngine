#include <gtest/gtest.h>

#ifdef LLMQUANT_SIGNAL_DECAY_HALFLIFE_ENABLED

#include "SignalDecayHalfLife.h"

#include <atomic>
#include <cmath>
#include <thread>
#include <vector>

using namespace llmquant;

namespace {

// -----------------------------------------------------------------------
// Default state
// -----------------------------------------------------------------------

TEST(SignalDecayHalfLife, DefaultState) {
    SignalDecayHalfLife d;
    EXPECT_TRUE(std::isnan(d.half_life()));
    EXPECT_TRUE(std::isnan(d.decay_rate()));
    EXPECT_EQ(d.total_records(), 0u);
}

// -----------------------------------------------------------------------
// Total records increments
// -----------------------------------------------------------------------

TEST(SignalDecayHalfLife, TotalRecordsIncrements) {
    SignalDecayHalfLife d;
    d.record(0.5);
    d.record(0.3);
    EXPECT_EQ(d.total_records(), 2u);
}

// -----------------------------------------------------------------------
// NaN before min_samples
// -----------------------------------------------------------------------

TEST(SignalDecayHalfLife, NaNBeforeMinSamples) {
    SignalDecayHalfLife::Config cfg;
    cfg.window      = 16;
    cfg.min_samples = 8;
    SignalDecayHalfLife d(cfg);
    for (int i = 0; i < 5; ++i) d.record(0.5);
    EXPECT_TRUE(std::isnan(d.half_life()));
}

// -----------------------------------------------------------------------
// Exponentially decaying signal → positive decay rate, finite half-life
// -----------------------------------------------------------------------

TEST(SignalDecayHalfLife, ExponentialDecayPositiveRate) {
    SignalDecayHalfLife::Config cfg;
    cfg.window      = 32;
    cfg.min_samples = 8;
    SignalDecayHalfLife d(cfg);
    // Feed x_t = exp(-t * 0.1): clean exponential decay
    for (int i = 0; i < 32; ++i)
        d.record(std::exp(-static_cast<double>(i) * 0.1));
    EXPECT_GT(d.decay_rate(), 0.0);
    EXPECT_TRUE(std::isfinite(d.half_life()));
    EXPECT_GT(d.half_life(), 0.0);
    // True half-life = ln(2)/0.1 ≈ 6.93 samples; should be in ballpark
    EXPECT_NEAR(d.half_life(), std::log(2.0) / 0.1, 3.0);
}

// -----------------------------------------------------------------------
// Constant signal → near-zero decay rate (slow/no decay)
// -----------------------------------------------------------------------

TEST(SignalDecayHalfLife, ConstantSignalSmallDecayRate) {
    SignalDecayHalfLife::Config cfg;
    cfg.window      = 16;
    cfg.min_samples = 8;
    SignalDecayHalfLife d(cfg);
    for (int i = 0; i < 16; ++i) d.record(1.0);
    // Constant → slope ≈ 0, decay rate ≈ 0, half-life NaN or very large
    if (std::isfinite(d.decay_rate())) {
        EXPECT_NEAR(d.decay_rate(), 0.0, 0.1);
    }
}

// -----------------------------------------------------------------------
// Fast decay callback fires
// -----------------------------------------------------------------------

TEST(SignalDecayHalfLife, FastDecayCbFires) {
    SignalDecayHalfLife::Config cfg;
    cfg.window          = 32;
    cfg.min_samples     = 8;
    cfg.fast_threshold  = 20.0;  // half-life < 20 → fast
    std::atomic<int> fires{0};
    cfg.on_fast_decay = [&](double) { ++fires; };
    SignalDecayHalfLife d(cfg);
    // Feed rapid exponential decay: λ ≈ 0.2, t½ ≈ 3.5 < 20
    for (int i = 0; i < 32; ++i)
        d.record(std::exp(-static_cast<double>(i) * 0.2));
    EXPECT_GE(fires.load(), 1);
    EXPECT_GT(d.fast_decay_events(), 0u);
}

// -----------------------------------------------------------------------
// Slow decay callback fires
// -----------------------------------------------------------------------

TEST(SignalDecayHalfLife, SlowDecayCbFires) {
    SignalDecayHalfLife::Config cfg;
    cfg.window          = 32;
    cfg.min_samples     = 8;
    cfg.slow_threshold  = 5.0;   // half-life > 5 → slow
    std::atomic<int> fires{0};
    cfg.on_slow_decay = [&](double) { ++fires; };
    SignalDecayHalfLife d(cfg);
    // Feed very slow decay: λ ≈ 0.01, t½ ≈ 69 > 5
    for (int i = 0; i < 32; ++i)
        d.record(std::exp(-static_cast<double>(i) * 0.01));
    EXPECT_GE(fires.load(), 1);
    EXPECT_GT(d.slow_decay_events(), 0u);
}

// -----------------------------------------------------------------------
// reset() clears state
// -----------------------------------------------------------------------

TEST(SignalDecayHalfLife, ResetClearsState) {
    SignalDecayHalfLife d;
    for (int i = 0; i < 20; ++i) d.record(std::exp(-static_cast<double>(i) * 0.1));
    d.reset();
    EXPECT_EQ(d.total_records(), 0u);
    EXPECT_TRUE(std::isnan(d.half_life()));
    EXPECT_EQ(d.fast_decay_events(), 0u);
}

// -----------------------------------------------------------------------
// update_config resets state
// -----------------------------------------------------------------------

TEST(SignalDecayHalfLife, UpdateConfigResetsState) {
    SignalDecayHalfLife d;
    for (int i = 0; i < 20; ++i) d.record(0.5);
    SignalDecayHalfLife::Config cfg2;
    cfg2.window = 16;
    d.update_config(cfg2);
    EXPECT_EQ(d.total_records(), 0u);
    EXPECT_TRUE(std::isnan(d.half_life()));
}

// -----------------------------------------------------------------------
// to_stats_json contains expected keys
// -----------------------------------------------------------------------

TEST(SignalDecayHalfLife, ToStatsJsonContainsKeys) {
    SignalDecayHalfLife d;
    for (int i = 0; i < 20; ++i) d.record(std::exp(-static_cast<double>(i) * 0.1));
    std::string j = d.to_stats_json();
    EXPECT_NE(j.find("half_life"),         std::string::npos);
    EXPECT_NE(j.find("decay_rate"),        std::string::npos);
    EXPECT_NE(j.find("r_squared"),         std::string::npos);
    EXPECT_NE(j.find("total_records"),     std::string::npos);
    EXPECT_NE(j.find("fast_decay_events"), std::string::npos);
}

// -----------------------------------------------------------------------
// Concurrent record() is thread-safe
// -----------------------------------------------------------------------

TEST(SignalDecayHalfLife, ConcurrentRecordSafe) {
    SignalDecayHalfLife d;
    std::vector<std::thread> threads;
    for (int t = 0; t < 4; ++t) {
        threads.emplace_back([&, t] {
            for (int i = 0; i < 250; ++i)
                d.record(std::exp(-static_cast<double>(i + t * 100) * 0.01));
        });
    }
    for (auto& th : threads) th.join();
    EXPECT_EQ(d.total_records(), 1000u);
}

} // namespace

#else

TEST(SignalDecayHalfLife, DisabledAtBuildTime) { SUCCEED(); }

#endif  // LLMQUANT_SIGNAL_DECAY_HALFLIFE_ENABLED
