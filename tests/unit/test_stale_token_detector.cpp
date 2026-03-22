#include <gtest/gtest.h>

#ifdef LLMQUANT_STALE_DETECTOR_ENABLED

#include "StaleTokenDetector.h"

#include <atomic>
#include <chrono>
#include <thread>

using namespace llmquant;

namespace {

TEST(StaleTokenDetector, DefaultConstructsNotStale) {
    StaleTokenDetector det;
    EXPECT_FALSE(det.is_stale());
    EXPECT_EQ(det.stale_events(), 0u);
    EXPECT_EQ(det.recovery_events(), 0u);
}

TEST(StaleTokenDetector, ConfigConstructsWithCustomThreshold) {
    StaleTokenDetector::Config cfg;
    cfg.stale_threshold_ms = 100;
    StaleTokenDetector det(cfg);
    EXPECT_FALSE(det.is_stale());
}

TEST(StaleTokenDetector, CheckReturnsFalseWhenFresh) {
    StaleTokenDetector::Config cfg;
    cfg.stale_threshold_ms = 5000;
    StaleTokenDetector det(cfg);
    det.record_token();
    EXPECT_FALSE(det.check());
}

TEST(StaleTokenDetector, CheckReturnsTrueAfterThreshold) {
    StaleTokenDetector::Config cfg;
    cfg.stale_threshold_ms = 50;
    StaleTokenDetector det(cfg);
    // Do not record any token; wait past threshold.
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
    bool result = det.check();
    EXPECT_TRUE(result);
    EXPECT_TRUE(det.is_stale());
}

TEST(StaleTokenDetector, StaleCallbackFiredOnce) {
    StaleTokenDetector::Config cfg;
    cfg.stale_threshold_ms = 50;
    StaleTokenDetector det(cfg);

    std::atomic<int> fired{0};
    det.set_stale_callback([&](int64_t gap_ms) {
        EXPECT_GT(gap_ms, 0);
        fired.fetch_add(1);
    });

    std::this_thread::sleep_for(std::chrono::milliseconds(100));
    det.check();
    det.check();  // second check must NOT re-fire
    EXPECT_EQ(fired.load(), 1);
    EXPECT_EQ(det.stale_events(), 1u);
}

TEST(StaleTokenDetector, RecoveryCallbackFiredAfterToken) {
    StaleTokenDetector::Config cfg;
    cfg.stale_threshold_ms = 50;
    StaleTokenDetector det(cfg);

    std::atomic<int> recovered{0};
    det.set_recovery_callback([&]() { recovered.fetch_add(1); });

    std::this_thread::sleep_for(std::chrono::milliseconds(100));
    det.check();                   // → stale
    EXPECT_TRUE(det.is_stale());

    det.record_token();            // token arrives
    det.check();                   // → recovered
    EXPECT_FALSE(det.is_stale());
    EXPECT_EQ(recovered.load(), 1);
    EXPECT_EQ(det.recovery_events(), 1u);
}

TEST(StaleTokenDetector, RecordTokenResetsGap) {
    StaleTokenDetector::Config cfg;
    cfg.stale_threshold_ms = 5000;
    StaleTokenDetector det(cfg);
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
    det.record_token();
    // Gap after record_token should be very small.
    EXPECT_LT(det.ms_since_last_token(), 200LL);
}

TEST(StaleTokenDetector, ResetClearsStaleFlagAndTimestamp) {
    StaleTokenDetector::Config cfg;
    cfg.stale_threshold_ms = 50;
    StaleTokenDetector det(cfg);

    std::this_thread::sleep_for(std::chrono::milliseconds(100));
    det.check();
    EXPECT_TRUE(det.is_stale());

    det.reset();
    EXPECT_FALSE(det.is_stale());
    EXPECT_LT(det.ms_since_last_token(), 200LL);
}

TEST(StaleTokenDetector, MsSinceLastTokenIncreasesOverTime) {
    StaleTokenDetector::Config cfg;
    cfg.stale_threshold_ms = 5000;
    StaleTokenDetector det(cfg);
    det.reset();
    int64_t t0 = det.ms_since_last_token();
    std::this_thread::sleep_for(std::chrono::milliseconds(50));
    int64_t t1 = det.ms_since_last_token();
    EXPECT_GE(t1, t0);
}

TEST(StaleTokenDetector, ToStatsJsonContainsKeys) {
    StaleTokenDetector det;
    std::string json = det.to_stats_json();
    EXPECT_NE(json.find("stale"), std::string::npos);
    EXPECT_NE(json.find("ms_since_last_token"), std::string::npos);
    EXPECT_NE(json.find("stale_events"), std::string::npos);
    EXPECT_NE(json.find("stale_threshold_ms"), std::string::npos);
}

TEST(StaleTokenDetector, UpdateConfigChangesThreshold) {
    StaleTokenDetector det;
    StaleTokenDetector::Config cfg;
    cfg.stale_threshold_ms = 99;
    det.update_config(cfg);
    // Not stale immediately after update.
    det.reset();
    EXPECT_FALSE(det.check());
}

TEST(StaleTokenDetector, NoCallbackNoCrash) {
    StaleTokenDetector::Config cfg;
    cfg.stale_threshold_ms = 30;
    StaleTokenDetector det(cfg);
    std::this_thread::sleep_for(std::chrono::milliseconds(60));
    // No callbacks registered — must not crash.
    EXPECT_NO_FATAL_FAILURE(det.check());
}

TEST(StaleTokenDetector, MultipleRecordTokensKeepFresh) {
    StaleTokenDetector::Config cfg;
    cfg.stale_threshold_ms = 100;
    StaleTokenDetector det(cfg);
    for (int i = 0; i < 5; ++i) {
        det.record_token();
        std::this_thread::sleep_for(std::chrono::milliseconds(20));
        EXPECT_FALSE(det.check()) << "Should still be fresh after token " << i;
    }
}

} // namespace

#else  // LLMQUANT_STALE_DETECTOR_ENABLED not defined

TEST(StaleTokenDetector, DisabledAtBuildTime) {
    SUCCEED();
}

#endif  // LLMQUANT_STALE_DETECTOR_ENABLED
