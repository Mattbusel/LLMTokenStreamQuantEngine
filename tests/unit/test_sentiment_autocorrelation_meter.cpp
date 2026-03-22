#include <gtest/gtest.h>

#ifdef LLMQUANT_AUTOCORR_METER_ENABLED
#include "SentimentAutocorrelationMeter.h"
#include <atomic>
#include <cmath>
#include <thread>
using namespace llmquant;

TEST(SentimentAutocorrelationMeterTest, InitialState) {
    SentimentAutocorrelationMeter m;
    EXPECT_NEAR(m.autocorrelation(1), 0.0, 1e-9);
    EXPECT_FALSE(m.is_trending());
    EXPECT_EQ(m.observation_count(), 0u);
}

TEST(SentimentAutocorrelationMeterTest, ConstantSeriesZeroAC) {
    SentimentAutocorrelationMeter::Config cfg;
    cfg.window = 20;
    cfg.max_lag = 3;
    SentimentAutocorrelationMeter m(cfg);
    for (int i = 0; i < 20; ++i) m.record(0.5);  // constant → zero variance → AC=0
    EXPECT_NEAR(m.autocorrelation(1), 0.0, 1e-9);
}

TEST(SentimentAutocorrelationMeterTest, TrendingSeriesHighLag1) {
    SentimentAutocorrelationMeter::Config cfg;
    cfg.window   = 30;
    cfg.max_lag  = 3;
    cfg.trending_threshold = 0.5;
    SentimentAutocorrelationMeter m(cfg);
    // Strictly monotone series → high lag-1 AC.
    for (int i = 0; i < 30; ++i) m.record(static_cast<double>(i) * 0.1);
    EXPECT_GT(m.autocorrelation(1), 0.5);
    EXPECT_TRUE(m.is_trending());
}

TEST(SentimentAutocorrelationMeterTest, AlternatingSeriesNegativeLag1) {
    SentimentAutocorrelationMeter::Config cfg;
    cfg.window  = 30;
    cfg.max_lag = 3;
    SentimentAutocorrelationMeter m(cfg);
    // Alternating signs → lag-1 AC ≈ -1.
    for (int i = 0; i < 30; ++i)
        m.record((i % 2 == 0) ? 1.0 : -1.0);
    EXPECT_LT(m.autocorrelation(1), -0.5);
}

TEST(SentimentAutocorrelationMeterTest, ACBoundedMinusOneToOne) {
    SentimentAutocorrelationMeter::Config cfg;
    cfg.window  = 30;
    cfg.max_lag = 5;
    SentimentAutocorrelationMeter m(cfg);
    for (int i = 0; i < 40; ++i)
        m.record(std::sin(i * 0.5));
    for (int k = 1; k <= 5; ++k) {
        EXPECT_LE(m.autocorrelation(k),  1.0 + 1e-6);
        EXPECT_GE(m.autocorrelation(k), -1.0 - 1e-6);
    }
}

TEST(SentimentAutocorrelationMeterTest, TrendingCallbackFires) {
    SentimentAutocorrelationMeter::Config cfg;
    cfg.window   = 20;
    cfg.max_lag  = 2;
    cfg.trending_threshold = 0.5;
    std::atomic<int> fires{0};
    cfg.on_regime_change = [&](double, bool) { ++fires; };
    SentimentAutocorrelationMeter m(cfg);
    // Start with alternating (not trending), then switch to monotone (trending).
    for (int i = 0; i < 20; ++i) m.record((i % 2 == 0) ? 1.0 : -1.0);
    for (int i = 0; i < 20; ++i) m.record(static_cast<double>(i) * 0.5);
    EXPECT_GE(fires.load(), 1);
}

TEST(SentimentAutocorrelationMeterTest, OutOfRangeLagReturnsZero) {
    SentimentAutocorrelationMeter m;
    EXPECT_NEAR(m.autocorrelation(0), 0.0, 1e-9);
    EXPECT_NEAR(m.autocorrelation(SentimentAutocorrelationMeter::kMaxLag + 1), 0.0, 1e-9);
}

TEST(SentimentAutocorrelationMeterTest, ObservationCountIncrements) {
    SentimentAutocorrelationMeter m;
    m.record(0.1);
    m.record(0.2);
    EXPECT_EQ(m.observation_count(), 2u);
}

TEST(SentimentAutocorrelationMeterTest, ResetClearsAll) {
    SentimentAutocorrelationMeter m;
    for (int i = 0; i < 20; ++i) m.record(i * 0.1);
    m.reset();
    EXPECT_EQ(m.observation_count(), 0u);
    EXPECT_NEAR(m.autocorrelation(1), 0.0, 1e-9);
}

TEST(SentimentAutocorrelationMeterTest, StatsJsonNonEmpty) {
    SentimentAutocorrelationMeter m;
    for (int i = 0; i < 10; ++i) m.record(i * 0.1);
    std::string j = m.to_stats_json();
    EXPECT_NE(j.find("is_trending"),       std::string::npos);
    EXPECT_NE(j.find("autocorrelations"),  std::string::npos);
}

TEST(SentimentAutocorrelationMeterTest, ConcurrentSafe) {
    SentimentAutocorrelationMeter m;
    std::atomic<bool> go{false};
    auto worker = [&] {
        while (!go.load()) {}
        for (int i = 0; i < 200; ++i) m.record(i * 0.01);
    };
    std::thread t1(worker), t2(worker);
    go.store(true);
    t1.join(); t2.join();
    EXPECT_EQ(m.observation_count(), 400u);
}

#else
TEST(SentimentAutocorrelationMeterTest, DisabledAtBuildTime) { SUCCEED(); }
#endif
