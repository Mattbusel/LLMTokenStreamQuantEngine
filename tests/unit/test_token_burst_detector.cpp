#include <gtest/gtest.h>

#ifdef LLMQUANT_BURST_DETECTOR_ENABLED
#include "TokenBurstDetector.h"
#include <chrono>
#include <thread>

using namespace llmquant;
using namespace std::chrono;

TEST(TokenBurstDetectorTest, ConstructsOk) {
    TokenBurstDetector det;
    EXPECT_EQ(det.total_tokens(), 0u);
    EXPECT_FALSE(det.is_burst());
}

TEST(TokenBurstDetectorTest, SingleTokenNotBurst) {
    TokenBurstDetector::Config cfg;
    cfg.burst_threshold = 100.0;
    cfg.window_ms       = 500;
    TokenBurstDetector det(cfg);
    det.record_at(steady_clock::now());
    EXPECT_FALSE(det.is_burst());
    EXPECT_EQ(det.total_tokens(), 1u);
}

TEST(TokenBurstDetectorTest, BurstThresholdTriggered) {
    // Window = 1000 ms, threshold = 50 tok/s → need > 50 tokens in 1 s.
    TokenBurstDetector::Config cfg;
    cfg.window_ms       = 1000;
    cfg.burst_threshold = 50.0;
    cfg.hysteresis      = 0.0;
    int fires = 0;
    cfg.on_burst_start = [&](double) { ++fires; };
    TokenBurstDetector det(cfg);

    auto t0 = steady_clock::now();
    for (int i = 0; i < 60; ++i)
        det.record_at(t0 + milliseconds(i * 10)); // 60 tokens in 600 ms → 100 tok/s

    EXPECT_GE(fires, 1);
    EXPECT_TRUE(det.is_burst());
    EXPECT_GE(det.burst_events(), 1u);
}

TEST(TokenBurstDetectorTest, BurstEndCallbackFires) {
    TokenBurstDetector::Config cfg;
    cfg.window_ms       = 1000;
    cfg.burst_threshold = 50.0;
    cfg.hysteresis      = 0.0;
    int end_fires = 0;
    cfg.on_burst_end = [&](double) { ++end_fires; };
    TokenBurstDetector det(cfg);

    // Create burst.
    auto t0 = steady_clock::now();
    for (int i = 0; i < 60; ++i)
        det.record_at(t0 + milliseconds(i * 10));

    // Now add a token 2 seconds later (all previous tokens expired → rate near 0).
    det.record_at(t0 + milliseconds(3000));
    EXPECT_GE(end_fires, 1);
    EXPECT_FALSE(det.is_burst());
}

TEST(TokenBurstDetectorTest, CurrentRateZeroWithNoTokens) {
    TokenBurstDetector det;
    EXPECT_DOUBLE_EQ(det.current_rate(), 0.0);
}

TEST(TokenBurstDetectorTest, ResetClearsState) {
    TokenBurstDetector::Config cfg;
    cfg.window_ms       = 1000;
    cfg.burst_threshold = 10.0;
    cfg.hysteresis      = 0.0;
    TokenBurstDetector det(cfg);

    auto t0 = steady_clock::now();
    for (int i = 0; i < 20; ++i)
        det.record_at(t0 + milliseconds(i * 10));

    det.reset();
    EXPECT_EQ(det.total_tokens(), 20u); // total_tokens is cumulative — not reset
    EXPECT_FALSE(det.is_burst());
    EXPECT_DOUBLE_EQ(det.current_rate(), 0.0);
}

TEST(TokenBurstDetectorTest, HysteresisPreventsFalseEnd) {
    // With hysteresis = 0.2, exit threshold = 50 * 0.8 = 40 tok/s.
    // If rate drops to 45 tok/s, still in burst.
    TokenBurstDetector::Config cfg;
    cfg.window_ms       = 1000;
    cfg.burst_threshold = 50.0;
    cfg.hysteresis      = 0.2;
    int end_fires = 0;
    cfg.on_burst_end = [&](double) { ++end_fires; };
    TokenBurstDetector det(cfg);

    auto t0 = steady_clock::now();
    // 60 tokens → 60 tok/s → burst.
    for (int i = 0; i < 60; ++i)
        det.record_at(t0 + milliseconds(i * 10));
    EXPECT_TRUE(det.is_burst());

    // Add 1 token 300 ms after last → evict some, but rate should still be ~40 tok/s.
    // (45 tokens in 1 s is above exit threshold of 40)
    det.record_at(t0 + milliseconds(600 + 300));
    // We can't guarantee exact rate without knowing eviction, but we can
    // at least verify no end callback has fired (rate is still likely above 40).
    EXPECT_EQ(end_fires, 0);
}

TEST(TokenBurstDetectorTest, UpdateConfigResetsWindow) {
    TokenBurstDetector det;
    auto t0 = steady_clock::now();
    for (int i = 0; i < 5; ++i)
        det.record_at(t0 + milliseconds(i * 10));
    det.update_config(TokenBurstDetector::Config{});
    EXPECT_DOUBLE_EQ(det.current_rate(), 0.0);
    EXPECT_FALSE(det.is_burst());
}

TEST(TokenBurstDetectorTest, StatsJsonContainsRequiredKeys) {
    TokenBurstDetector det;
    std::string json = det.to_stats_json();
    EXPECT_NE(json.find("current_rate"),  std::string::npos);
    EXPECT_NE(json.find("is_burst"),      std::string::npos);
    EXPECT_NE(json.find("total_tokens"),  std::string::npos);
    EXPECT_NE(json.find("burst_events"),  std::string::npos);
}

TEST(TokenBurstDetectorTest, TotalTokensCountsAllCalls) {
    TokenBurstDetector det;
    auto t0 = steady_clock::now();
    for (int i = 0; i < 7; ++i)
        det.record_at(t0 + milliseconds(i));
    EXPECT_EQ(det.total_tokens(), 7u);
}

#else
TEST(TokenBurstDetectorTest, FeatureFlagDisabled) { SUCCEED(); }
#endif
