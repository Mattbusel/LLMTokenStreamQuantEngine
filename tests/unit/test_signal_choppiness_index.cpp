#include <gtest/gtest.h>
#include "SignalChoppinessIndex.h"
#include <cmath>

namespace llmquant::tests {

TEST(SignalChoppinessIndex, DefaultConstruction) {
    SignalChoppinessIndex ci;
    EXPECT_EQ(ci.total_records(), 0u);
    EXPECT_DOUBLE_EQ(ci.choppiness(), 61.8);
}

TEST(SignalChoppinessIndex, HighCIForAlternatingSignal) {
    SignalChoppinessIndex::Config cfg;
    cfg.min_samples       = 5;
    cfg.choppy_threshold  = 61.8;
    cfg.trending_threshold = 38.2;
    SignalChoppinessIndex ci(cfg);
    // Alternating signal → many direction changes → high CI
    for (int i = 0; i < 30; ++i) ci.record(i % 2 == 0 ? 1.0 : -1.0);
    EXPECT_GT(ci.choppiness(), 50.0);
}

TEST(SignalChoppinessIndex, LowCIForTrendingSignal) {
    SignalChoppinessIndex::Config cfg;
    cfg.min_samples        = 5;
    cfg.trending_threshold = 38.2;
    cfg.choppy_threshold   = 61.8;
    SignalChoppinessIndex ci(cfg);
    // Monotone uptrend → one big move, small ATR-sum/range ratio → low CI
    for (int i = 0; i < 30; ++i) ci.record(static_cast<double>(i));
    EXPECT_LT(ci.choppiness(), 50.0);
}

TEST(SignalChoppinessIndex, OnTrendingCallback) {
    SignalChoppinessIndex::Config cfg;
    cfg.min_samples        = 5;
    cfg.trending_threshold = 80.0; // very high → always "trending"
    int trending = 0;
    cfg.on_trending = [&](double) { ++trending; };
    SignalChoppinessIndex ci(cfg);
    for (int i = 0; i < 30; ++i) ci.record(static_cast<double>(i));
    EXPECT_GT(trending, 0);
}

TEST(SignalChoppinessIndex, OnChoppyCallback) {
    SignalChoppinessIndex::Config cfg;
    cfg.min_samples      = 5;
    cfg.choppy_threshold = 0.0; // very low → always "choppy"
    int choppy = 0;
    cfg.on_choppy = [&](double) { ++choppy; };
    SignalChoppinessIndex ci(cfg);
    for (int i = 0; i < 30; ++i) ci.record(i % 2 == 0 ? 1.0 : -1.0);
    EXPECT_GT(choppy, 0);
}

TEST(SignalChoppinessIndex, IsTrendingAndIsChoppy) {
    SignalChoppinessIndex ci;
    for (int i = 0; i < 20; ++i) ci.record(0.1);
    // Just confirm accessors don't crash and return bool
    bool t = ci.is_trending();
    bool c = ci.is_choppy();
    EXPECT_FALSE(t && c); // cannot be both simultaneously
}

TEST(SignalChoppinessIndex, Reset) {
    SignalChoppinessIndex ci;
    for (int i = 0; i < 25; ++i) ci.record(0.1);
    ci.reset();
    EXPECT_EQ(ci.total_records(), 0u);
    EXPECT_DOUBLE_EQ(ci.choppiness(), 61.8);
    EXPECT_EQ(ci.trending_events(), 0u);
    EXPECT_EQ(ci.choppy_events(), 0u);
}

TEST(SignalChoppinessIndex, ToStatsJson) {
    SignalChoppinessIndex ci;
    for (int i = 0; i < 20; ++i) ci.record(0.05);
    auto json = ci.to_stats_json();
    EXPECT_NE(json.find("choppiness"), std::string::npos);
    EXPECT_NE(json.find("trending_events"), std::string::npos);
}

} // namespace llmquant::tests
