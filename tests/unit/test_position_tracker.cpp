#include <gtest/gtest.h>

#if defined(LLMQUANT_POSITION_TRACKER_ENABLED) && defined(LLMQUANT_KELLY_SIZER_ENABLED)

#include "PositionTracker.h"
#include "KellyPositionSizer.h"

#include <atomic>
#include <thread>
#include <vector>

using namespace llmquant;

namespace {

static TradeSignal make_signal(double bias, double confidence = 0.8) {
    TradeSignal s{};
    s.delta_bias_shift      = bias;
    s.confidence            = confidence;
    s.volatility_adjustment = 0.01;
    s.spread_modifier       = 0.001;
    s.signal_quality        = 0.9;
    return s;
}

// ============================================================
// Construction
// ============================================================

TEST(PositionTracker, DefaultConstructsWithZeroStats) {
    KellyPositionSizer kelly;
    PositionTracker tracker(kelly);
    auto s = tracker.get_stats();
    EXPECT_EQ(s.total_trades, 0u);
    EXPECT_EQ(s.open_trades, 0u);
    EXPECT_NEAR(s.cumulative_pnl, 0.0, 1e-12);
}

TEST(PositionTracker, ConfigConstructs) {
    KellyPositionSizer kelly;
    PositionTracker::Config cfg;
    cfg.max_open_trades = 4;
    PositionTracker tracker(kelly, cfg);
    EXPECT_EQ(tracker.get_config().max_open_trades, 4);
}

// ============================================================
// open_trade / close_trade lifecycle
// ============================================================

TEST(PositionTracker, OpenTradeReturnsNonZeroId) {
    KellyPositionSizer kelly;
    PositionTracker tracker(kelly);
    uint64_t id = tracker.open_trade(make_signal(1.0), 100.0);
    EXPECT_GT(id, 0u);
    EXPECT_EQ(tracker.open_trade_count(), 1u);
}

TEST(PositionTracker, CloseTradeReturnsTrueForKnownId) {
    KellyPositionSizer kelly;
    PositionTracker tracker(kelly);
    uint64_t id = tracker.open_trade(make_signal(1.0), 100.0);
    EXPECT_TRUE(tracker.close_trade(id, 102.0));
    EXPECT_EQ(tracker.open_trade_count(), 0u);
}

TEST(PositionTracker, CloseTradeReturnsFalseForUnknownId) {
    KellyPositionSizer kelly;
    PositionTracker tracker(kelly);
    EXPECT_FALSE(tracker.close_trade(9999u, 100.0));
}

// ============================================================
// P&L calculation
// ============================================================

TEST(PositionTracker, WinningTradeIncreasesWinCount) {
    KellyPositionSizer kelly;
    PositionTracker tracker(kelly);
    uint64_t id = tracker.open_trade(make_signal(1.0), 100.0);
    tracker.close_trade(id, 105.0);  // +5%
    auto s = tracker.get_stats();
    EXPECT_EQ(s.total_trades, 1u);
    EXPECT_EQ(s.total_wins, 1u);
    EXPECT_EQ(s.total_losses, 0u);
    EXPECT_NEAR(s.cumulative_pnl, 0.05, 1e-9);
}

TEST(PositionTracker, LosingTradeIncreasesLossCount) {
    KellyPositionSizer kelly;
    PositionTracker tracker(kelly);
    uint64_t id = tracker.open_trade(make_signal(1.0), 100.0);
    tracker.close_trade(id, 95.0);  // -5%
    auto s = tracker.get_stats();
    EXPECT_EQ(s.total_wins, 0u);
    EXPECT_EQ(s.total_losses, 1u);
    EXPECT_NEAR(s.cumulative_pnl, -0.05, 1e-9);
}

TEST(PositionTracker, BearishSignalFlipsReturnSign) {
    KellyPositionSizer kelly;
    PositionTracker tracker(kelly);
    // Negative bias = short.  If price falls we profit.
    uint64_t id = tracker.open_trade(make_signal(-1.0), 100.0);
    tracker.close_trade(id, 95.0);  // price fell — short wins
    auto s = tracker.get_stats();
    EXPECT_EQ(s.total_wins, 1u);
    EXPECT_NEAR(s.cumulative_pnl, 0.05, 1e-9);
}

TEST(PositionTracker, WinRateComputedCorrectly) {
    KellyPositionSizer kelly;
    PositionTracker tracker(kelly);
    uint64_t a = tracker.open_trade(make_signal(1.0), 100.0);
    uint64_t b = tracker.open_trade(make_signal(1.0), 100.0);
    uint64_t c = tracker.open_trade(make_signal(1.0), 100.0);
    tracker.close_trade(a, 102.0);  // win
    tracker.close_trade(b, 102.0);  // win
    tracker.close_trade(c,  98.0);  // loss
    auto s = tracker.get_stats();
    EXPECT_NEAR(s.win_rate, 2.0 / 3.0, 1e-9);
}

// ============================================================
// max_open_trades limit
// ============================================================

TEST(PositionTracker, RejectsWhenMaxOpenTradesReached) {
    KellyPositionSizer kelly;
    PositionTracker::Config cfg;
    cfg.max_open_trades = 2;
    PositionTracker tracker(kelly, cfg);
    EXPECT_GT(tracker.open_trade(make_signal(1.0), 100.0), 0u);
    EXPECT_GT(tracker.open_trade(make_signal(1.0), 100.0), 0u);
    EXPECT_EQ(tracker.open_trade(make_signal(1.0), 100.0), 0u);  // rejected
    EXPECT_EQ(tracker.get_stats().rejected_opens, 1u);
}

// ============================================================
// Automatic Kelly feedback
// ============================================================

TEST(PositionTracker, AutoKellyFeedbackUpdatesKellyOutcomes) {
    KellyPositionSizer kelly;
    PositionTracker tracker(kelly);
    EXPECT_EQ(kelly.total_outcomes(), 0u);
    uint64_t id = tracker.open_trade(make_signal(1.0), 100.0);
    tracker.close_trade(id, 103.0);
    EXPECT_EQ(kelly.total_outcomes(), 1u);
}

TEST(PositionTracker, DisabledAutoKellyFeedbackDoesNotUpdateKelly) {
    KellyPositionSizer kelly;
    PositionTracker::Config cfg;
    cfg.auto_kelly_feedback = false;
    PositionTracker tracker(kelly, cfg);
    uint64_t id = tracker.open_trade(make_signal(1.0), 100.0);
    tracker.close_trade(id, 103.0);
    EXPECT_EQ(kelly.total_outcomes(), 0u);
}

TEST(PositionTracker, MinReturnFilterSkipsSmallTrades) {
    KellyPositionSizer kelly;
    PositionTracker::Config cfg;
    cfg.min_return_for_kelly = 0.05;
    PositionTracker tracker(kelly, cfg);
    uint64_t id = tracker.open_trade(make_signal(1.0), 100.0);
    tracker.close_trade(id, 101.0);  // +1%, below threshold
    EXPECT_EQ(kelly.total_outcomes(), 0u);
}

// ============================================================
// DrawDown and best/worst
// ============================================================

TEST(PositionTracker, MaxDrawdownTracked) {
    KellyPositionSizer kelly;
    PositionTracker tracker(kelly);
    uint64_t a = tracker.open_trade(make_signal(1.0), 100.0);
    tracker.close_trade(a, 110.0);  // +10%, peak pnl = 0.1
    uint64_t b = tracker.open_trade(make_signal(1.0), 100.0);
    tracker.close_trade(b, 90.0);   // -10%, pnl = 0.0, drawdown = 0.1
    auto s = tracker.get_stats();
    EXPECT_NEAR(s.max_drawdown, 0.10, 1e-9);
}

TEST(PositionTracker, BestAndWorstTradeTracked) {
    KellyPositionSizer kelly;
    PositionTracker tracker(kelly);
    uint64_t a = tracker.open_trade(make_signal(1.0), 100.0);
    tracker.close_trade(a, 115.0);  // +15%
    uint64_t b = tracker.open_trade(make_signal(1.0), 100.0);
    tracker.close_trade(b, 92.0);   // -8%
    auto s = tracker.get_stats();
    EXPECT_NEAR(s.best_trade,  0.15, 1e-9);
    EXPECT_NEAR(s.worst_trade, -0.08, 1e-9);
}

// ============================================================
// reset
// ============================================================

TEST(PositionTracker, ResetClearsAllStats) {
    KellyPositionSizer kelly;
    PositionTracker tracker(kelly);
    uint64_t id = tracker.open_trade(make_signal(1.0), 100.0);
    tracker.close_trade(id, 105.0);
    tracker.reset();
    auto s = tracker.get_stats();
    EXPECT_EQ(s.total_trades, 0u);
    EXPECT_NEAR(s.cumulative_pnl, 0.0, 1e-12);
}

// ============================================================
// Callback
// ============================================================

TEST(PositionTracker, TradeCloseCallbackFired) {
    KellyPositionSizer kelly;
    PositionTracker tracker(kelly);
    std::atomic<int> called{0};
    double captured_return = 0.0;
    bool   captured_win    = false;

    tracker.set_trade_close_callback([&](uint64_t /*id*/, double r, bool w) {
        captured_return = r;
        captured_win    = w;
        called.fetch_add(1);
    });

    uint64_t id = tracker.open_trade(make_signal(1.0), 100.0);
    tracker.close_trade(id, 104.0);
    EXPECT_EQ(called.load(), 1);
    EXPECT_NEAR(captured_return, 0.04, 1e-9);
    EXPECT_TRUE(captured_win);
}

// ============================================================
// to_stats_json
// ============================================================

TEST(PositionTracker, ToStatsJsonContainsKeys) {
    KellyPositionSizer kelly;
    PositionTracker tracker(kelly);
    std::string json = tracker.to_stats_json();
    EXPECT_NE(json.find("total_trades"),   std::string::npos);
    EXPECT_NE(json.find("win_rate"),       std::string::npos);
    EXPECT_NE(json.find("cumulative_pnl"), std::string::npos);
    EXPECT_NE(json.find("max_drawdown"),   std::string::npos);
}

// ============================================================
// Thread safety
// ============================================================

TEST(PositionTracker, ConcurrentOpenCloseIsThreadSafe) {
    KellyPositionSizer kelly;
    PositionTracker tracker(kelly);

    std::atomic<bool> start{false};
    auto worker = [&]() {
        while (!start.load()) {}
        for (int i = 0; i < 50; ++i) {
            uint64_t id = tracker.open_trade(make_signal(1.0), 100.0);
            if (id != 0) {
                (void)tracker.close_trade(id, 100.0 + static_cast<double>(i % 10));
            }
        }
    };

    std::vector<std::thread> threads;
    for (int i = 0; i < 4; ++i) threads.emplace_back(worker);
    start.store(true);
    for (auto& t : threads) t.join();
    SUCCEED();
}

} // namespace

#else  // LLMQUANT_POSITION_TRACKER_ENABLED && LLMQUANT_KELLY_SIZER_ENABLED not defined

TEST(PositionTracker, DisabledAtBuildTime) {
    SUCCEED();
}

#endif  // LLMQUANT_POSITION_TRACKER_ENABLED && LLMQUANT_KELLY_SIZER_ENABLED
