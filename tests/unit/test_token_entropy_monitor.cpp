#include <gtest/gtest.h>

#ifdef LLMQUANT_ENTROPY_MONITOR_ENABLED

#include "TokenEntropyMonitor.h"

#include <cmath>
#include <functional>
#include <string>
#include <thread>
#include <vector>

using namespace llmquant;

namespace {

static uint64_t H(const std::string& s) {
    return std::hash<std::string>{}(s);
}

// ---------------------------------------------------------------------------
// Construction and defaults
// ---------------------------------------------------------------------------

TEST(TokenEntropyMonitor, DefaultConstructs) {
    TokenEntropyMonitor mon;
    EXPECT_EQ(mon.total_recorded(), 0u);
    EXPECT_EQ(mon.unique_tokens(), 0);
    EXPECT_NEAR(mon.entropy_ema(), 0.5, 0.001);
}

TEST(TokenEntropyMonitor, ConfigConstructs) {
    TokenEntropyMonitor::Config cfg;
    cfg.window_size = 64;
    cfg.focused_threshold = 0.3;
    TokenEntropyMonitor mon(cfg);
    auto c = mon.get_config();
    EXPECT_EQ(c.window_size, 64);
    EXPECT_NEAR(c.focused_threshold, 0.3, 1e-9);
}

// ---------------------------------------------------------------------------
// Entropy of perfectly uniform distribution = 1.0 (max)
// ---------------------------------------------------------------------------

TEST(TokenEntropyMonitor, UniformDistributionMaxEntropy) {
    TokenEntropyMonitor::Config cfg;
    cfg.window_size = 64;
    cfg.ema_alpha = 1.0;  // instant update
    TokenEntropyMonitor mon(cfg);

    // Fill with 64 distinct tokens.
    for (int i = 0; i < 64; ++i) {
        mon.record(static_cast<uint64_t>(i));
    }
    double h = mon.entropy();
    EXPECT_NEAR(h, 1.0, 0.02);
}

// ---------------------------------------------------------------------------
// Entropy of single repeated token ≈ 0 (minimum)
// ---------------------------------------------------------------------------

TEST(TokenEntropyMonitor, SingleTokenMinEntropy) {
    TokenEntropyMonitor::Config cfg;
    cfg.window_size = 64;
    cfg.ema_alpha = 1.0;
    TokenEntropyMonitor mon(cfg);

    for (int i = 0; i < 64; ++i) mon.record(42ULL);
    double h = mon.entropy();
    EXPECT_NEAR(h, 0.0, 0.01);
}

// ---------------------------------------------------------------------------
// is_focused after low-entropy stream
// ---------------------------------------------------------------------------

TEST(TokenEntropyMonitor, IsFocusedWhenEntropyLow) {
    TokenEntropyMonitor::Config cfg;
    cfg.window_size = 64;
    cfg.focused_threshold = 0.4;
    cfg.ema_alpha = 1.0;
    TokenEntropyMonitor mon(cfg);

    // Only two distinct tokens — low entropy.
    for (int i = 0; i < 64; ++i) mon.record(i % 2);
    (void)mon.entropy();  // update EMA
    EXPECT_TRUE(mon.is_focused());
}

// ---------------------------------------------------------------------------
// is_focused returns false for high-entropy stream
// ---------------------------------------------------------------------------

TEST(TokenEntropyMonitor, NotFocusedWhenEntropyHigh) {
    TokenEntropyMonitor::Config cfg;
    cfg.window_size = 64;
    cfg.focused_threshold = 0.4;
    cfg.ema_alpha = 1.0;
    TokenEntropyMonitor mon(cfg);

    for (int i = 0; i < 64; ++i) mon.record(static_cast<uint64_t>(i));
    (void)mon.entropy();
    EXPECT_FALSE(mon.is_focused());
}

// ---------------------------------------------------------------------------
// Rolling window: old tokens expire
// ---------------------------------------------------------------------------

TEST(TokenEntropyMonitor, RollingWindowExpires) {
    TokenEntropyMonitor::Config cfg;
    cfg.window_size = 4;
    cfg.ema_alpha = 1.0;
    TokenEntropyMonitor mon(cfg);

    // Fill with 4 distinct tokens.
    for (int i = 0; i < 4; ++i) mon.record(static_cast<uint64_t>(i));
    double h1 = mon.entropy();
    EXPECT_NEAR(h1, 1.0, 0.01);

    // Overwrite with a single repeated token.
    for (int i = 0; i < 4; ++i) mon.record(999ULL);
    double h2 = mon.entropy();
    EXPECT_LT(h2, 0.01);  // near zero
}

// ---------------------------------------------------------------------------
// unique_tokens
// ---------------------------------------------------------------------------

TEST(TokenEntropyMonitor, UniqueTokensCountsDistinct) {
    TokenEntropyMonitor::Config cfg;
    cfg.window_size = 32;
    TokenEntropyMonitor mon(cfg);

    for (int i = 0; i < 5; ++i) mon.record(static_cast<uint64_t>(i));
    EXPECT_EQ(mon.unique_tokens(), 5);

    // Adding the same tokens again does not increase count beyond window.
    for (int i = 0; i < 5; ++i) mon.record(static_cast<uint64_t>(i));
    EXPECT_EQ(mon.unique_tokens(), 5);
}

// ---------------------------------------------------------------------------
// total_recorded increments monotonically
// ---------------------------------------------------------------------------

TEST(TokenEntropyMonitor, TotalRecordedIncrements) {
    TokenEntropyMonitor mon;
    mon.record(1);
    mon.record(2);
    mon.record(3);
    EXPECT_EQ(mon.total_recorded(), 3u);
}

// ---------------------------------------------------------------------------
// EMA smoothing
// ---------------------------------------------------------------------------

TEST(TokenEntropyMonitor, EmaSmoothing) {
    TokenEntropyMonitor::Config cfg;
    cfg.window_size = 32;
    cfg.ema_alpha = 0.1;
    TokenEntropyMonitor mon(cfg);

    // Record uniform tokens → raw entropy ≈ 1.0.
    for (int i = 0; i < 32; ++i) mon.record(static_cast<uint64_t>(i));

    double prev_ema = mon.entropy_ema();
    for (int iter = 0; iter < 10; ++iter) (void)mon.entropy();
    double next_ema = mon.entropy_ema();

    // EMA should be moving toward 1.0.
    EXPECT_GT(next_ema, prev_ema);
}

// ---------------------------------------------------------------------------
// reset() clears state
// ---------------------------------------------------------------------------

TEST(TokenEntropyMonitor, ResetClearsState) {
    TokenEntropyMonitor mon;
    for (int i = 0; i < 10; ++i) mon.record(static_cast<uint64_t>(i));
    EXPECT_GT(mon.total_recorded(), 0u);

    mon.reset();
    EXPECT_EQ(mon.total_recorded(), 0u);
    EXPECT_EQ(mon.unique_tokens(), 0);
    EXPECT_NEAR(mon.entropy_ema(), 0.5, 0.001);
}

// ---------------------------------------------------------------------------
// update_config() resizes window and clears data
// ---------------------------------------------------------------------------

TEST(TokenEntropyMonitor, UpdateConfigResizes) {
    TokenEntropyMonitor mon;
    for (int i = 0; i < 10; ++i) mon.record(static_cast<uint64_t>(i));

    TokenEntropyMonitor::Config cfg2;
    cfg2.window_size = 32;
    mon.update_config(cfg2);
    EXPECT_EQ(mon.get_config().window_size, 32);
    EXPECT_EQ(mon.unique_tokens(), 0);
}

// ---------------------------------------------------------------------------
// to_stats_json() returns valid JSON-like string
// ---------------------------------------------------------------------------

TEST(TokenEntropyMonitor, ToStatsJsonContainsFields) {
    TokenEntropyMonitor mon;
    mon.record(1);
    mon.record(2);
    (void)mon.entropy();
    std::string j = mon.to_stats_json();
    EXPECT_NE(j.find("entropy_ema"), std::string::npos);
    EXPECT_NE(j.find("is_focused"), std::string::npos);
    EXPECT_NE(j.find("unique_tokens"), std::string::npos);
    EXPECT_NE(j.find("total_recorded"), std::string::npos);
}

// ---------------------------------------------------------------------------
// Window size clamped to kMaxWindowSize
// ---------------------------------------------------------------------------

TEST(TokenEntropyMonitor, WindowSizeClamped) {
    TokenEntropyMonitor::Config cfg;
    cfg.window_size = 99999;  // over max
    TokenEntropyMonitor mon(cfg);
    EXPECT_LE(mon.get_config().window_size, 4096);
}

// ---------------------------------------------------------------------------
// Hash-based record via real token strings
// ---------------------------------------------------------------------------

TEST(TokenEntropyMonitor, RealTokenStrings) {
    TokenEntropyMonitor::Config cfg;
    cfg.window_size = 128;
    cfg.ema_alpha = 1.0;
    TokenEntropyMonitor mon(cfg);

    std::vector<std::string> tokens = {"bullish", "rally", "crash", "bearish",
                                       "bullish", "bullish", "rally", "positive"};
    for (auto& t : tokens) mon.record(H(t));

    double h = mon.entropy();
    // 4 distinct tokens, not uniform → entropy between 0 and 1
    EXPECT_GT(h, 0.0);
    EXPECT_LT(h, 1.0);
}

// ---------------------------------------------------------------------------
// Concurrent record() calls are safe
// ---------------------------------------------------------------------------

TEST(TokenEntropyMonitor, ConcurrentRecordSafe) {
    TokenEntropyMonitor mon;
    std::atomic<bool> go{false};

    auto worker = [&] {
        while (!go.load(std::memory_order_acquire)) {}
        for (int i = 0; i < 500; ++i)
            mon.record(static_cast<uint64_t>(i % 8));
    };

    std::vector<std::thread> threads;
    for (int t = 0; t < 4; ++t) threads.emplace_back(worker);
    go.store(true, std::memory_order_release);
    for (auto& t : threads) t.join();

    EXPECT_EQ(mon.total_recorded(), 2000u);
    double h = mon.entropy();
    EXPECT_GE(h, 0.0);
    EXPECT_LE(h, 1.0);
}

// ---------------------------------------------------------------------------
// Entropy with two equally probable tokens ≈ 1/log2(W) normalised
// ---------------------------------------------------------------------------

TEST(TokenEntropyMonitor, TwoTokenEntropyKnownValue) {
    TokenEntropyMonitor::Config cfg;
    cfg.window_size = 4;
    cfg.ema_alpha = 1.0;
    TokenEntropyMonitor mon(cfg);

    // Exactly 2 tokens, each appearing twice in window of 4.
    mon.record(0); mon.record(1); mon.record(0); mon.record(1);
    double h = mon.entropy();
    // H = -2*(0.5*log2(0.5)) / log2(4) = 1.0 / 2.0 = 0.5
    EXPECT_NEAR(h, 0.5, 0.02);
}

} // namespace

#else

TEST(TokenEntropyMonitor, DisabledAtBuildTime) { SUCCEED(); }

#endif  // LLMQUANT_ENTROPY_MONITOR_ENABLED
