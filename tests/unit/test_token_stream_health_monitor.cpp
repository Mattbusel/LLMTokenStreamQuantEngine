#include <gtest/gtest.h>

#ifdef LLMQUANT_STREAM_HEALTH_ENABLED

#include "TokenStreamHealthMonitor.h"

#include <atomic>
#include <chrono>
#include <thread>

using namespace llmquant;

namespace {

// ============================================================
// Default state
// ============================================================

TEST(TokenStreamHealthMonitor, DefaultStateHealthy) {
    TokenStreamHealthMonitor mon;
    EXPECT_TRUE(mon.is_healthy());
    EXPECT_EQ(mon.status(), TokenStreamHealthMonitor::Status::Healthy);
    EXPECT_EQ(mon.total_tokens(), 0u);
    EXPECT_EQ(mon.stall_count(), 0u);
    EXPECT_EQ(mon.flood_count(), 0u);
}

// ============================================================
// ms_since_last_token returns INT64_MAX before first ping
// ============================================================

TEST(TokenStreamHealthMonitor, MsSinceLastTokenBeforePingMax) {
    TokenStreamHealthMonitor mon;
    EXPECT_EQ(mon.ms_since_last_token(), INT64_MAX);
}

// ============================================================
// ping increments total_tokens
// ============================================================

TEST(TokenStreamHealthMonitor, PingIncrementsTotalTokens) {
    TokenStreamHealthMonitor mon;
    mon.ping();
    mon.ping();
    mon.ping();
    EXPECT_EQ(mon.total_tokens(), 3u);
}

// ============================================================
// ms_since_last_token is small after recent ping
// ============================================================

TEST(TokenStreamHealthMonitor, MsSinceLastTokenSmallAfterPing) {
    TokenStreamHealthMonitor mon;
    mon.ping();
    EXPECT_LT(mon.ms_since_last_token(), 100);
}

// ============================================================
// Flood detection: rapid pings exceed max_tokens_per_sec
// ============================================================

TEST(TokenStreamHealthMonitor, FloodDetectedOnHighRate) {
    TokenStreamHealthMonitor::Config cfg;
    cfg.max_tokens_per_sec = 5.0;   // very low threshold
    cfg.rate_window_ms     = 1000;
    cfg.stall_timeout_ms   = 60000; // don't stall during this test
    std::atomic<int> floods{0};
    cfg.on_flood = [&](double) { ++floods; };
    TokenStreamHealthMonitor mon(cfg);

    // Send many tokens very quickly.
    for (int i = 0; i < 50; ++i) mon.ping();

    EXPECT_EQ(mon.status(), TokenStreamHealthMonitor::Status::Flooded);
    EXPECT_GT(mon.flood_count(), 0u);
    EXPECT_GT(floods.load(), 0);
}

// ============================================================
// Stall detection via poll() after a sleep
// ============================================================

TEST(TokenStreamHealthMonitor, StallDetectedAfterTimeout) {
    TokenStreamHealthMonitor::Config cfg;
    cfg.stall_timeout_ms   = 50;  // 50ms stall threshold
    cfg.max_tokens_per_sec = 0;   // disable flood
    std::atomic<int> stalls{0};
    cfg.on_stall = [&](int64_t) { ++stalls; };
    TokenStreamHealthMonitor mon(cfg);

    mon.ping();  // establish first token
    std::this_thread::sleep_for(std::chrono::milliseconds(80));
    mon.poll();  // trigger stall detection

    EXPECT_EQ(mon.status(), TokenStreamHealthMonitor::Status::Stalled);
    EXPECT_GT(mon.stall_count(), 0u);
    EXPECT_GT(stalls.load(), 0);
}

// ============================================================
// Recovery from stall when ping arrives
// ============================================================

TEST(TokenStreamHealthMonitor, RecoveryAfterStall) {
    TokenStreamHealthMonitor::Config cfg;
    cfg.stall_timeout_ms   = 30;
    cfg.max_tokens_per_sec = 0;
    std::atomic<int> recoveries{0};
    cfg.on_recovery = [&] { ++recoveries; };
    TokenStreamHealthMonitor mon(cfg);

    mon.ping();
    std::this_thread::sleep_for(std::chrono::milliseconds(50));
    mon.poll();  // trigger stall
    EXPECT_EQ(mon.status(), TokenStreamHealthMonitor::Status::Stalled);

    // New token should trigger recovery.
    mon.ping();
    // Immediately after ping the stall gap is 0; re-evaluate.
    mon.poll();
    EXPECT_EQ(mon.status(), TokenStreamHealthMonitor::Status::Healthy);
    EXPECT_GT(recoveries.load(), 0);
}

// ============================================================
// reset() returns to initial state
// ============================================================

TEST(TokenStreamHealthMonitor, ResetRestoresState) {
    TokenStreamHealthMonitor::Config cfg;
    cfg.max_tokens_per_sec = 2.0;
    TokenStreamHealthMonitor mon(cfg);

    for (int i = 0; i < 20; ++i) mon.ping();
    EXPECT_GT(mon.total_tokens(), 0u);

    mon.reset();
    EXPECT_EQ(mon.total_tokens(), 0u);
    EXPECT_TRUE(mon.is_healthy());
    EXPECT_EQ(mon.ms_since_last_token(), INT64_MAX);
}

// ============================================================
// to_stats_json contains required keys
// ============================================================

TEST(TokenStreamHealthMonitor, ToStatsJsonContainsKeys) {
    TokenStreamHealthMonitor mon;
    mon.ping();
    std::string json = mon.to_stats_json();
    EXPECT_NE(json.find("status"),        std::string::npos);
    EXPECT_NE(json.find("rate_tps"),      std::string::npos);
    EXPECT_NE(json.find("total_tokens"),  std::string::npos);
    EXPECT_NE(json.find("stall_count"),   std::string::npos);
    EXPECT_NE(json.find("flood_count"),   std::string::npos);
}

// ============================================================
// Concurrent ping() is thread-safe
// ============================================================

TEST(TokenStreamHealthMonitor, ConcurrentPingSafe) {
    TokenStreamHealthMonitor::Config cfg;
    cfg.max_tokens_per_sec = 0; // disable flood for this test
    cfg.stall_timeout_ms   = 60000;
    TokenStreamHealthMonitor mon(cfg);

    std::vector<std::thread> threads;
    for (int t = 0; t < 4; ++t) {
        threads.emplace_back([&] {
            for (int i = 0; i < 250; ++i) mon.ping();
        });
    }
    for (auto& th : threads) th.join();

    EXPECT_EQ(mon.total_tokens(), 1000u);
}

} // namespace

#else  // LLMQUANT_STREAM_HEALTH_ENABLED not defined

TEST(TokenStreamHealthMonitor, DisabledAtBuildTime) {
    SUCCEED();
}

#endif  // LLMQUANT_STREAM_HEALTH_ENABLED
