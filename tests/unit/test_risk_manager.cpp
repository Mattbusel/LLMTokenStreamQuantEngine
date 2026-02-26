#include "gtest/gtest.h"
#include "RiskManager.h"
#include <thread>
#include <chrono>
#include <string>

using namespace llmquant;

// Helper: build a neutral passing signal.
static TradeSignal make_signal(double bias = 0.1, double vol = 0.1,
                               double spread = 0.05, double conf = 0.8) {
    TradeSignal s;
    s.delta_bias_shift      = bias;
    s.volatility_adjustment = vol;
    s.spread_modifier       = spread;
    s.confidence            = conf;
    s.timestamp_ns          = 1;
    return s;
}

// Default permissive config used by most tests.
static RiskManager::Config default_config() {
    RiskManager::Config cfg;
    cfg.max_bias_magnitude       = 1.0;
    cfg.max_volatility_magnitude = 1.0;
    cfg.max_spread_magnitude     = 0.5;
    cfg.min_confidence           = 0.1;
    cfg.max_signals_per_second   = 100;
    cfg.max_drawdown             = 5.0;
    cfg.drawdown_window          = std::chrono::seconds{60};
    return cfg;
}

// ============================================================
// Test 1: a well-formed signal within all limits passes.
// ============================================================
TEST(RiskManagerTest, test_risk_manager_passing_signal_increments_passed_count) {
    RiskManager rm(default_config());
    auto sig = make_signal(0.1, 0.1, 0.05, 0.8);

    bool result = rm.evaluate(sig);

    EXPECT_TRUE(result);
    EXPECT_EQ(rm.get_stats().signals_passed.load(), 1u);
    EXPECT_EQ(rm.get_stats().signals_blocked_magnitude.load(), 0u);
    EXPECT_EQ(rm.get_stats().signals_blocked_confidence.load(), 0u);
    EXPECT_EQ(rm.get_stats().signals_blocked_rate.load(), 0u);
    EXPECT_EQ(rm.get_stats().signals_blocked_drawdown.load(), 0u);
}

// ============================================================
// Test 2: delta_bias_shift exceeding max_bias_magnitude blocks.
// ============================================================
TEST(RiskManagerTest, test_risk_manager_magnitude_bias_too_large_blocks_signal) {
    RiskManager rm(default_config());
    // Exactly at limit is allowed; one ULP over the limit is blocked.
    auto sig = make_signal(1.5, 0.1, 0.05, 0.8);  // bias 1.5 > 1.0

    bool result = rm.evaluate(sig);

    EXPECT_FALSE(result);
    EXPECT_EQ(rm.get_stats().signals_blocked_magnitude.load(), 1u);
    EXPECT_EQ(rm.get_stats().signals_passed.load(), 0u);
}

// ============================================================
// Test 3: volatility_adjustment exceeding limit blocks.
// ============================================================
TEST(RiskManagerTest, test_risk_manager_magnitude_volatility_too_large_blocks) {
    RiskManager rm(default_config());
    auto sig = make_signal(0.1, 2.0, 0.05, 0.8);  // vol 2.0 > 1.0

    bool result = rm.evaluate(sig);

    EXPECT_FALSE(result);
    EXPECT_EQ(rm.get_stats().signals_blocked_magnitude.load(), 1u);
}

// ============================================================
// Test 4: spread_modifier exceeding limit blocks.
// ============================================================
TEST(RiskManagerTest, test_risk_manager_magnitude_spread_too_large_blocks) {
    RiskManager rm(default_config());
    auto sig = make_signal(0.1, 0.1, 0.9, 0.8);  // spread 0.9 > 0.5

    bool result = rm.evaluate(sig);

    EXPECT_FALSE(result);
    EXPECT_EQ(rm.get_stats().signals_blocked_magnitude.load(), 1u);
}

// ============================================================
// Test 5: confidence below minimum blocks.
// ============================================================
TEST(RiskManagerTest, test_risk_manager_low_confidence_blocks_signal) {
    RiskManager rm(default_config());
    auto sig = make_signal(0.1, 0.1, 0.05, 0.05);  // conf 0.05 < 0.1

    bool result = rm.evaluate(sig);

    EXPECT_FALSE(result);
    EXPECT_EQ(rm.get_stats().signals_blocked_confidence.load(), 1u);
    EXPECT_EQ(rm.get_stats().signals_blocked_magnitude.load(), 0u);
}

// ============================================================
// Test 6: rate limit blocks signals beyond max_signals_per_second.
// ============================================================
TEST(RiskManagerTest, test_risk_manager_rate_limit_blocks_after_threshold) {
    RiskManager::Config cfg = default_config();
    cfg.max_signals_per_second = 3;
    RiskManager rm(cfg);

    auto sig = make_signal(0.1, 0.1, 0.05, 0.8);

    // First 3 should pass.
    EXPECT_TRUE(rm.evaluate(sig));
    EXPECT_TRUE(rm.evaluate(sig));
    EXPECT_TRUE(rm.evaluate(sig));

    // 4th within the same second must be blocked.
    bool result = rm.evaluate(sig);
    EXPECT_FALSE(result);
    EXPECT_GT(rm.get_stats().signals_blocked_rate.load(), 0u);
    EXPECT_EQ(rm.get_stats().signals_passed.load(), 3u);
}

// ============================================================
// Test 7: cumulative drawdown blocks when limit exceeded.
// ============================================================
TEST(RiskManagerTest, test_risk_manager_drawdown_blocks_after_cumulative_bias) {
    RiskManager::Config cfg = default_config();
    cfg.max_drawdown             = 1.0;
    cfg.max_signals_per_second   = 1000;  // remove rate limit interference
    RiskManager rm(cfg);

    // Each signal adds 0.4 bias; after 3 cumulative = 1.2 > 1.0.
    auto sig = make_signal(0.4, 0.1, 0.05, 0.8);

    EXPECT_TRUE(rm.evaluate(sig));   // cumulative = 0.4
    EXPECT_TRUE(rm.evaluate(sig));   // cumulative = 0.8
    bool blocked = rm.evaluate(sig); // would push to 1.2 > 1.0
    EXPECT_FALSE(blocked);
    EXPECT_GT(rm.get_stats().signals_blocked_drawdown.load(), 0u);
}

// ============================================================
// Test 8: alert callback is invoked with the correct reason.
// ============================================================
TEST(RiskManagerTest, test_risk_manager_alert_callback_fired_on_block) {
    RiskManager rm(default_config());

    std::string captured_reason;
    rm.set_alert_callback([&](const std::string& reason, const TradeSignal&) {
        captured_reason = reason;
    });

    // Trigger a magnitude block.
    auto sig = make_signal(5.0, 0.1, 0.05, 0.8);
    rm.evaluate(sig);

    EXPECT_EQ(captured_reason, "magnitude_exceeded");
}

// ============================================================
// Test 9: reset() clears drawdown accumulator and rate window.
// ============================================================
TEST(RiskManagerTest, test_risk_manager_reset_clears_drawdown_and_rate) {
    RiskManager::Config cfg = default_config();
    cfg.max_drawdown           = 0.3;
    cfg.max_signals_per_second = 1;
    RiskManager rm(cfg);

    auto sig = make_signal(0.3, 0.1, 0.05, 0.8);

    // Exhaust drawdown and rate limit.
    rm.evaluate(sig);  // passes, cumulative = 0.3, rate used = 1

    // Both drawdown (0.3 + 0.3 = 0.6 > 0.3) and rate (already 1 in window)
    // would block, but after reset both should clear.
    rm.reset();

    bool result = rm.evaluate(sig);
    EXPECT_TRUE(result);
    EXPECT_EQ(rm.get_stats().signals_passed.load(), 2u);
}

// ============================================================
// Test 10: drawdown accumulator resets after the window elapses.
// ============================================================
TEST(RiskManagerTest, test_risk_manager_drawdown_resets_after_window) {
    RiskManager::Config cfg = default_config();
    cfg.max_drawdown           = 0.5;
    cfg.drawdown_window        = std::chrono::seconds{1};
    cfg.max_signals_per_second = 1000;
    RiskManager rm(cfg);

    auto sig = make_signal(0.4, 0.1, 0.05, 0.8);

    // First signal passes, cumulative = 0.4.
    EXPECT_TRUE(rm.evaluate(sig));

    // Second would push cumulative to 0.8 > 0.5 — blocked.
    EXPECT_FALSE(rm.evaluate(sig));
    EXPECT_EQ(rm.get_stats().signals_blocked_drawdown.load(), 1u);

    // Wait for the drawdown window to expire (1 second + small buffer).
    std::this_thread::sleep_for(std::chrono::milliseconds(1100));

    // After window reset the next signal should pass again.
    bool result = rm.evaluate(sig);
    EXPECT_TRUE(result);
}

// ============================================================
// Test 11: exact boundary values are accepted (not over-rejected).
// ============================================================
TEST(RiskManagerTest, test_risk_manager_exact_boundary_values_pass) {
    RiskManager rm(default_config());
    // Exactly at the limit — should pass.
    auto sig = make_signal(1.0, 1.0, 0.5, 0.1);

    bool result = rm.evaluate(sig);

    EXPECT_TRUE(result);
    EXPECT_EQ(rm.get_stats().signals_passed.load(), 1u);
    EXPECT_EQ(rm.get_stats().signals_blocked_magnitude.load(), 0u);
}

// ============================================================
// Test 12: negative bias and spread magnitudes are checked via abs().
// ============================================================
TEST(RiskManagerTest, test_risk_manager_negative_magnitude_checked_via_abs) {
    RiskManager rm(default_config());

    // Negative values outside range should also block.
    auto sig_bias   = make_signal(-1.5, 0.1, 0.05, 0.8);
    auto sig_spread = make_signal(0.1,  0.1, -0.9, 0.8);

    EXPECT_FALSE(rm.evaluate(sig_bias));
    EXPECT_FALSE(rm.evaluate(sig_spread));
    EXPECT_EQ(rm.get_stats().signals_blocked_magnitude.load(), 2u);
}

// ============================================================
// Test 13 (OMS): hard position breach blocks signal.
// ============================================================
TEST(RiskManagerTest, test_risk_manager_oms_hard_position_breach_blocks_signal) {
    RiskManager rm(default_config());

    // Position already at 0.9 with limit 1.0; a signal of +0.2 would push to
    // 1.1 which exceeds the hard limit.
    RiskManager::PositionState pos;
    pos.net_position   = 0.9;
    pos.position_limit = 1.0;
    pos.pnl            = 0.0;
    pos.pnl_limit      = -10.0;
    rm.update_position(pos);

    auto sig = make_signal(0.2, 0.1, 0.05, 0.8);
    bool result = rm.evaluate(sig);

    EXPECT_FALSE(result);
    EXPECT_EQ(rm.get_stats().signals_blocked_position.load(), 1u);
    EXPECT_EQ(rm.get_stats().signals_passed.load(), 0u);
}

// ============================================================
// Test 14 (OMS): soft position warn fires callback but allows signal through.
// ============================================================
TEST(RiskManagerTest, test_risk_manager_oms_soft_position_warn_allows_signal_but_fires_callback) {
    RiskManager rm(default_config());

    // Position at 0.5; limit 1.0; warn fraction 0.8; signal +0.4 pushes to
    // 0.9 which is > 0.8*1.0 (soft threshold) but <= 1.0 (hard limit).
    RiskManager::PositionState pos;
    pos.net_position   = 0.5;
    pos.position_limit = 1.0;
    pos.pnl            = 0.0;
    pos.pnl_limit      = -10.0;
    rm.update_position(pos);

    std::string captured_event;
    rm.set_oms_callback([&](const std::string& event,
                             const RiskManager::PositionState&,
                             const TradeSignal&) {
        captured_event = event;
    });

    auto sig = make_signal(0.4, 0.1, 0.05, 0.8);
    bool result = rm.evaluate(sig);

    EXPECT_TRUE(result)
        << "Signal within hard limit must be allowed through despite soft warn";
    EXPECT_EQ(captured_event, "position_limit_approaching")
        << "OMS callback must receive the soft-warn event string";
    EXPECT_EQ(rm.get_stats().signals_blocked_position.load(), 0u);
    EXPECT_EQ(rm.get_stats().signals_passed.load(), 1u);
}

// ============================================================
// Test 15 (OMS): PnL breach blocks signal.
// ============================================================
TEST(RiskManagerTest, test_risk_manager_oms_pnl_breach_blocks_signal) {
    RiskManager rm(default_config());

    // PnL is -15 which is below the pnl_limit of -10; any signal must be blocked.
    RiskManager::PositionState pos;
    pos.net_position   = 0.0;
    pos.position_limit = 1.0;
    pos.pnl            = -15.0;
    pos.pnl_limit      = -10.0;
    rm.update_position(pos);

    auto sig = make_signal(0.1, 0.1, 0.05, 0.8);
    bool result = rm.evaluate(sig);

    EXPECT_FALSE(result);
    EXPECT_EQ(rm.get_stats().signals_blocked_position.load(), 1u);
}

// ============================================================
// Test 16 (OMS): OMS callback receives correct event strings.
// ============================================================
TEST(RiskManagerTest, test_risk_manager_oms_callback_receives_correct_event_string) {
    // Verify that each code path sends the expected event label to the OMS cb.
    // Hard breach path.
    {
        RiskManager rm(default_config());
        RiskManager::PositionState pos;
        pos.net_position   = 0.95;
        pos.position_limit = 1.0;
        pos.pnl            = 0.0;
        pos.pnl_limit      = -10.0;
        rm.update_position(pos);

        std::string ev;
        rm.set_oms_callback([&](const std::string& event,
                                 const RiskManager::PositionState&,
                                 const TradeSignal&) { ev = event; });

        rm.evaluate(make_signal(0.1, 0.1, 0.05, 0.8));  // 0.95 + 0.1 > 1.0
        EXPECT_EQ(ev, "position_limit_breached");
    }

    // PnL breach path.
    {
        RiskManager rm(default_config());
        RiskManager::PositionState pos;
        pos.net_position   = 0.0;
        pos.position_limit = 1.0;
        pos.pnl            = -20.0;
        pos.pnl_limit      = -10.0;
        rm.update_position(pos);

        std::string ev;
        rm.set_oms_callback([&](const std::string& event,
                                 const RiskManager::PositionState&,
                                 const TradeSignal&) { ev = event; });

        rm.evaluate(make_signal(0.1, 0.1, 0.05, 0.8));
        EXPECT_EQ(ev, "pnl_limit_breached");
    }
}
