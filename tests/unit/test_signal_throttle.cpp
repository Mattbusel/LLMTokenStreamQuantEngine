#include <gtest/gtest.h>

#ifdef LLMQUANT_SIGNAL_THROTTLE_ENABLED

#include "SignalThrottle.h"

#include <atomic>
#include <chrono>
#include <thread>

using namespace llmquant;

namespace {

// ============================================================
// Construction
// ============================================================

TEST(SignalThrottle, DefaultConstructsWithFullBucket) {
    SignalThrottle throttle;
    EXPECT_GT(throttle.current_tokens(), 0.0);
    EXPECT_EQ(throttle.total_attempts(), 0u);
    EXPECT_EQ(throttle.dropped_count(),  0u);
}

// ============================================================
// try_acquire with full bucket passes
// ============================================================

TEST(SignalThrottle, FullBucketPassesAcquire) {
    SignalThrottle::Config cfg;
    cfg.capacity    = 5.0;
    cfg.refill_rate = 0.0;  // no refill
    SignalThrottle throttle(cfg);

    EXPECT_TRUE(throttle.try_acquire());
    EXPECT_EQ(throttle.passed_count(),  1u);
    EXPECT_EQ(throttle.dropped_count(), 0u);
}

// ============================================================
// Empty bucket drops signals
// ============================================================

TEST(SignalThrottle, EmptyBucketDropsAcquire) {
    SignalThrottle::Config cfg;
    cfg.capacity      = 2.0;
    cfg.refill_rate   = 0.0;
    cfg.initial_tokens = 0.0;  // start empty
    SignalThrottle throttle(cfg);

    EXPECT_FALSE(throttle.try_acquire());
    EXPECT_EQ(throttle.dropped_count(), 1u);
    EXPECT_EQ(throttle.passed_count(),  0u);
}

// ============================================================
// Bucket drains to zero
// ============================================================

TEST(SignalThrottle, BucketDrainsToEmpty) {
    SignalThrottle::Config cfg;
    cfg.capacity    = 3.0;
    cfg.refill_rate = 0.0;
    SignalThrottle throttle(cfg);

    EXPECT_TRUE(throttle.try_acquire());   // 2 left
    EXPECT_TRUE(throttle.try_acquire());   // 1 left
    EXPECT_TRUE(throttle.try_acquire());   // 0 left
    EXPECT_FALSE(throttle.try_acquire());  // empty
    EXPECT_EQ(throttle.dropped_count(), 1u);
    EXPECT_EQ(throttle.passed_count(),  3u);
}

// ============================================================
// Refill over time
// ============================================================

TEST(SignalThrottle, RefillReplenishesTokens) {
    SignalThrottle::Config cfg;
    cfg.capacity       = 5.0;
    cfg.refill_rate    = 100.0;  // very fast refill
    cfg.initial_tokens = 0.0;   // start empty
    SignalThrottle throttle(cfg);

    std::this_thread::sleep_for(std::chrono::milliseconds(50));
    // After 50ms at 100 tokens/s, should have ~5 tokens.
    EXPECT_TRUE(throttle.try_acquire());
}

// ============================================================
// Drop callback
// ============================================================

TEST(SignalThrottle, DropCallbackFiredOnDrop) {
    SignalThrottle::Config cfg;
    cfg.capacity       = 1.0;
    cfg.refill_rate    = 0.0;
    cfg.initial_tokens = 0.0;
    SignalThrottle throttle(cfg);

    std::atomic<int> drops{0};
    throttle.set_drop_callback([&] { drops.fetch_add(1); });

    (void)throttle.try_acquire();  // bucket empty → drop
    EXPECT_EQ(drops.load(), 1);
}

// ============================================================
// drop_rate
// ============================================================

TEST(SignalThrottle, DropRateComputedCorrectly) {
    SignalThrottle::Config cfg;
    cfg.capacity    = 2.0;
    cfg.refill_rate = 0.0;
    SignalThrottle throttle(cfg);

    (void)throttle.try_acquire();  // pass
    (void)throttle.try_acquire();  // pass
    (void)throttle.try_acquire();  // drop
    (void)throttle.try_acquire();  // drop

    EXPECT_NEAR(throttle.drop_rate(), 0.5, 1e-9);
}

// ============================================================
// reset
// ============================================================

TEST(SignalThrottle, ResetRestoresFullBucket) {
    SignalThrottle::Config cfg;
    cfg.capacity    = 3.0;
    cfg.refill_rate = 0.0;
    SignalThrottle throttle(cfg);

    (void)throttle.try_acquire();
    (void)throttle.try_acquire();
    (void)throttle.try_acquire();
    EXPECT_FALSE(throttle.try_acquire());  // empty

    throttle.reset();
    EXPECT_TRUE(throttle.try_acquire());   // full again
    EXPECT_EQ(throttle.dropped_count(), 0u);  // counters cleared
}

// ============================================================
// update_config
// ============================================================

TEST(SignalThrottle, UpdateConfigChangesCapacity) {
    SignalThrottle throttle;
    SignalThrottle::Config cfg;
    cfg.capacity       = 1.0;
    cfg.refill_rate    = 0.0;
    cfg.initial_tokens = 1.0;
    throttle.update_config(cfg);

    EXPECT_TRUE(throttle.try_acquire());
    EXPECT_FALSE(throttle.try_acquire());
}

// ============================================================
// to_stats_json
// ============================================================

TEST(SignalThrottle, ToStatsJsonContainsKeys) {
    SignalThrottle throttle;
    (void)throttle.try_acquire();
    std::string json = throttle.to_stats_json();
    EXPECT_NE(json.find("tokens"),       std::string::npos);
    EXPECT_NE(json.find("capacity"),     std::string::npos);
    EXPECT_NE(json.find("refill_rate"),  std::string::npos);
    EXPECT_NE(json.find("dropped"),      std::string::npos);
    EXPECT_NE(json.find("drop_rate"),    std::string::npos);
}

} // namespace

#else  // LLMQUANT_SIGNAL_THROTTLE_ENABLED not defined

TEST(SignalThrottle, DisabledAtBuildTime) {
    SUCCEED();
}

#endif  // LLMQUANT_SIGNAL_THROTTLE_ENABLED
