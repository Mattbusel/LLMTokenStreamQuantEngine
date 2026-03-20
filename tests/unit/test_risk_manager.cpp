#include "gtest/gtest.h"
#include "MetricsLogger.h"
#include "RiskManager.h"
#include <cstdio>
#include <stdexcept>
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

// ============================================================
// Test 17: update_config() changes thresholds at runtime.
// ============================================================
TEST(RiskManagerTest, test_risk_manager_update_config_tightens_and_relaxes_limits) {
    RiskManager rm(default_config());
    // Signal within default limits — passes.
    EXPECT_TRUE(rm.evaluate(make_signal(0.9, 0.1, 0.05, 0.8)));

    // Tighten bias limit to 0.5 — same signal should now be blocked.
    RiskManager::Config tight = default_config();
    tight.max_bias_magnitude = 0.5;
    rm.update_config(tight);
    EXPECT_FALSE(rm.evaluate(make_signal(0.9, 0.1, 0.05, 0.8)));
    EXPECT_GE(rm.get_stats().signals_blocked_magnitude.load(), 1u);

    // Relax back to original limits — signal should pass again.
    rm.update_config(default_config());
    EXPECT_TRUE(rm.evaluate(make_signal(0.9, 0.1, 0.05, 0.8)));
}

// ============================================================
// Test 18: get_config() reflects the active configuration.
// ============================================================
TEST(RiskManagerTest, test_risk_manager_get_config_reflects_update) {
    RiskManager rm(default_config());
    EXPECT_DOUBLE_EQ(rm.get_config().max_bias_magnitude, default_config().max_bias_magnitude);

    RiskManager::Config custom = default_config();
    custom.max_bias_magnitude = 0.25;
    custom.min_confidence     = 0.5;
    rm.update_config(custom);

    auto retrieved = rm.get_config();
    EXPECT_DOUBLE_EQ(retrieved.max_bias_magnitude, 0.25);
    EXPECT_DOUBLE_EQ(retrieved.min_confidence,     0.5);
}

// ============================================================
// Test 19: concurrent evaluate() from multiple threads — no crash / no UB.
// All gates disabled except magnitude; all signals pass.
// Total passed must equal N_THREADS * N_CALLS exactly.
// ============================================================
TEST(RiskManagerTest, test_risk_manager_concurrent_evaluate_no_crash) {
    RiskManager::Config cfg = default_config();
    cfg.max_signals_per_second = 100000;  // remove rate interference
    cfg.disable_drawdown_gate  = true;    // remove drawdown interference
    cfg.disable_position_gate  = true;    // remove position interference
    RiskManager rm(cfg);

    constexpr int N_THREADS = 8;
    constexpr int N_CALLS   = 100;
    std::vector<std::thread> threads;
    threads.reserve(N_THREADS);

    for (int t = 0; t < N_THREADS; ++t) {
        threads.emplace_back([&]() {
            for (int i = 0; i < N_CALLS; ++i) {
                rm.evaluate(make_signal(0.1, 0.1, 0.05, 0.8));
            }
        });
    }
    for (auto& th : threads) th.join();

    // All signals are within limits so every call must have passed.
    EXPECT_EQ(rm.get_stats().signals_passed.load(),
              static_cast<uint64_t>(N_THREADS * N_CALLS));
    EXPECT_EQ(rm.get_stats().signals_blocked_magnitude.load(), 0u);
    EXPECT_EQ(rm.get_stats().signals_blocked_confidence.load(), 0u);
}

// ============================================================
// Test 20: reset_stats() clears all signal counters.
// ============================================================
TEST(RiskManagerTest, test_risk_manager_reset_stats_clears_counters) {
    RiskManager rm(default_config());

    // Block some signals to populate the counters.
    rm.evaluate(make_signal(0.5, 0.1, 0.05, 0.8));  // passes
    rm.evaluate(make_signal(9.9, 0.1, 0.05, 0.8));  // blocked magnitude
    ASSERT_GT(rm.get_stats().signals_passed.load(),             0u);
    ASSERT_GT(rm.get_stats().signals_blocked_magnitude.load(),  0u);

    rm.reset_stats();

    // All stat counters must be zero after reset.
    EXPECT_EQ(rm.get_stats().signals_passed.load(),             0u);
    EXPECT_EQ(rm.get_stats().signals_blocked_magnitude.load(),  0u);
    EXPECT_EQ(rm.get_stats().signals_blocked_confidence.load(), 0u);
    EXPECT_EQ(rm.get_stats().signals_blocked_rate.load(),       0u);
    EXPECT_EQ(rm.get_stats().signals_blocked_drawdown.load(),   0u);
    EXPECT_EQ(rm.get_stats().signals_blocked_position.load(),   0u);
    EXPECT_EQ(rm.get_stats().signals_blocked_pnl.load(),        0u);

    // Engine must still work normally after reset_stats().
    rm.evaluate(make_signal(0.1, 0.1, 0.05, 0.8));
    EXPECT_EQ(rm.get_stats().signals_passed.load(), 1u);
}

// ============================================================
// Test 21: get_cumulative_bias() tracks accumulation and resets.
// ============================================================
TEST(RiskManagerTest, test_risk_manager_get_cumulative_bias_updates_and_resets) {
    RiskManager rm(default_config());
    EXPECT_DOUBLE_EQ(rm.get_cumulative_bias(), 0.0);

    // Process signals with a known bias shift to accumulate drawdown.
    rm.evaluate(make_signal(0.4, 0.1, 0.05, 0.8));  // contributes 0.4 to drawdown
    EXPECT_NE(rm.get_cumulative_bias(), 0.0)
        << "Cumulative bias must be non-zero after evaluating a signal";

    // After full reset, accumulator must return to zero.
    rm.reset();
    EXPECT_DOUBLE_EQ(rm.get_cumulative_bias(), 0.0);
}

// ============================================================
// Test 22: alert callback fires once per blocked signal.
// ============================================================
TEST(RiskManagerTest, test_risk_manager_alert_callback_fires_for_each_blocked_signal) {
    RiskManager rm(default_config());

    std::atomic<int> alert_count{0};
    rm.set_alert_callback([&](const std::string&, const TradeSignal&) {
        alert_count.fetch_add(1, std::memory_order_relaxed);
    });

    // 3 signals that exceed the max_bias_magnitude (1.0).
    for (int i = 0; i < 3; ++i) {
        rm.evaluate(make_signal(9.0, 0.1, 0.05, 0.8));
    }

    EXPECT_EQ(alert_count.load(), 3)
        << "Alert callback must fire exactly once per blocked signal";
    EXPECT_EQ(rm.get_stats().signals_blocked_magnitude.load(), 3u);
}

// ============================================================
// Test 23: short position blocks additional negative-bias signals.
// ============================================================
TEST(RiskManagerTest, test_risk_manager_short_position_blocks_negative_bias) {
    // Position already at -0.95 (short) with limit 1.0.
    // A signal with delta_bias_shift = -0.2 projects to -1.15 which exceeds
    // the absolute limit of 1.0.
    RiskManager::Config cfg = default_config();
    cfg.disable_magnitude_gate  = true;   // allow large magnitude so position gate fires
    cfg.disable_drawdown_gate   = true;
    cfg.disable_rate_gate       = true;
    RiskManager rm(cfg);

    RiskManager::PositionState pos;
    pos.net_position   = -0.95;
    pos.position_limit =  1.0;
    pos.pnl            =  0.0;
    pos.pnl_limit      = -10.0;
    rm.update_position(pos);

    // Negative bias projects position further negative — must be blocked.
    auto sig = make_signal(-0.2, 0.1, 0.05, 0.8);
    bool passed = rm.evaluate(sig);

    EXPECT_FALSE(passed) << "Negative bias on short position exceeding limit must be blocked";
    EXPECT_GT(rm.get_stats().signals_blocked_position.load(), 0u);
}

// ============================================================
// Test 24: update_config() throws on negative/invalid limits.
// ============================================================

TEST(RiskManagerTest, test_risk_manager_update_config_negative_bias_magnitude_throws) {
    RiskManager rm(default_config());
    RiskManager::Config bad = default_config();
    bad.max_bias_magnitude = -1.0;
    EXPECT_THROW(rm.update_config(bad), std::invalid_argument);
}

TEST(RiskManagerTest, test_risk_manager_update_config_negative_volatility_magnitude_throws) {
    RiskManager rm(default_config());
    RiskManager::Config bad = default_config();
    bad.max_volatility_magnitude = -0.5;
    EXPECT_THROW(rm.update_config(bad), std::invalid_argument);
}

TEST(RiskManagerTest, test_risk_manager_update_config_negative_spread_magnitude_throws) {
    RiskManager rm(default_config());
    RiskManager::Config bad = default_config();
    bad.max_spread_magnitude = -0.1;
    EXPECT_THROW(rm.update_config(bad), std::invalid_argument);
}

TEST(RiskManagerTest, test_risk_manager_update_config_confidence_above_one_throws) {
    RiskManager rm(default_config());
    RiskManager::Config bad = default_config();
    bad.min_confidence = 1.5;
    EXPECT_THROW(rm.update_config(bad), std::invalid_argument);
}

TEST(RiskManagerTest, test_risk_manager_update_config_negative_confidence_throws) {
    RiskManager rm(default_config());
    RiskManager::Config bad = default_config();
    bad.min_confidence = -0.1;
    EXPECT_THROW(rm.update_config(bad), std::invalid_argument);
}

TEST(RiskManagerTest, test_risk_manager_update_config_negative_drawdown_throws) {
    RiskManager rm(default_config());
    RiskManager::Config bad = default_config();
    bad.max_drawdown = -5.0;
    EXPECT_THROW(rm.update_config(bad), std::invalid_argument);
}

TEST(RiskManagerTest, test_risk_manager_update_config_valid_succeeds) {
    RiskManager rm(default_config());
    RiskManager::Config good = default_config();
    good.max_bias_magnitude = 2.0;
    good.min_confidence     = 0.05;
    EXPECT_NO_THROW(rm.update_config(good));
    EXPECT_DOUBLE_EQ(rm.get_config().max_bias_magnitude, 2.0);
}

// ============================================================
// Test: evaluate_with_reason() — passing signal returns empty reason.
// ============================================================
TEST(RiskManagerTest, test_risk_manager_evaluate_with_reason_pass_returns_empty_reason) {
    RiskManager rm(default_config());
    std::string reason;
    bool result = rm.evaluate_with_reason(make_signal(0.1, 0.1, 0.05, 0.8), reason);
    EXPECT_TRUE(result);
    EXPECT_TRUE(reason.empty()) << "Reason must be empty on pass";
}

// ============================================================
// Test: evaluate_with_reason() — blocked signal returns non-empty reason.
// ============================================================
TEST(RiskManagerTest, test_risk_manager_evaluate_with_reason_block_returns_reason_string) {
    RiskManager rm(default_config());
    std::string reason;
    // Magnitude too large — should be blocked with reason "magnitude_exceeded".
    bool result = rm.evaluate_with_reason(make_signal(5.0, 0.1, 0.05, 0.8), reason);
    EXPECT_FALSE(result);
    EXPECT_EQ(reason, "magnitude_exceeded");
}

// ============================================================
// Test: evaluate_with_reason() — each rejection reason maps correctly.
// ============================================================
// ============================================================
// Test: get_drawdown_budget_remaining() returns correct headroom.
// ============================================================
// ============================================================
// Test: disable_all_gates() lets every signal through.
// ============================================================
TEST(RiskManagerTest, test_risk_manager_disable_all_gates_lets_every_signal_through) {
    RiskManager::Config cfg = default_config();
    cfg.max_bias_magnitude       = 0.01;  // would normally block most signals
    cfg.max_signals_per_second   = 1;     // rate limit
    cfg.max_drawdown             = 0.01;  // tiny drawdown
    cfg.min_confidence           = 0.99;  // very high confidence required
    RiskManager rm(cfg);

    rm.disable_all_gates();

    // A signal that would normally fail every check must now pass.
    auto sig = make_signal(5.0, 5.0, 5.0, 0.01);
    for (int i = 0; i < 5; ++i) {
        EXPECT_TRUE(rm.evaluate(sig))
            << "All signals must pass when all gates are disabled";
    }
    EXPECT_EQ(rm.get_stats().signals_passed.load(), 5u);
    EXPECT_EQ(rm.get_stats().signals_blocked_magnitude.load(), 0u);
}

// ============================================================
// Test: enable_all_gates() re-activates all checks.
// ============================================================
TEST(RiskManagerTest, test_risk_manager_enable_all_gates_blocks_signals_again) {
    RiskManager rm(default_config());
    rm.disable_all_gates();

    // Large magnitude signal passes when gates disabled.
    EXPECT_TRUE(rm.evaluate(make_signal(5.0, 0.1, 0.05, 0.8)));

    // Re-enable — now the same signal must be blocked by magnitude gate.
    rm.enable_all_gates();
    EXPECT_FALSE(rm.evaluate(make_signal(5.0, 0.1, 0.05, 0.8)));
    EXPECT_GT(rm.get_stats().signals_blocked_magnitude.load(), 0u);
}

TEST(RiskManagerTest, test_risk_manager_get_drawdown_budget_remaining_full_at_start) {
    RiskManager::Config cfg = default_config();
    cfg.max_drawdown = 5.0;
    RiskManager rm(cfg);

    // No signals yet — full budget available.
    EXPECT_DOUBLE_EQ(rm.get_drawdown_budget_remaining(), 5.0);
}

TEST(RiskManagerTest, test_risk_manager_get_drawdown_budget_remaining_decreases_on_signal) {
    RiskManager::Config cfg = default_config();
    cfg.max_drawdown           = 5.0;
    cfg.max_signals_per_second = 1000;
    RiskManager rm(cfg);

    auto sig = make_signal(1.0, 0.1, 0.05, 0.8);
    rm.evaluate(sig);  // cumulative_bias = 1.0

    double remaining = rm.get_drawdown_budget_remaining();
    // max_drawdown(5.0) - |cumulative_bias(1.0)| = 4.0
    EXPECT_NEAR(remaining, 4.0, 1e-9);
}

TEST(RiskManagerTest, test_risk_manager_get_drawdown_budget_remaining_clamps_at_zero) {
    RiskManager::Config cfg = default_config();
    cfg.max_drawdown           = 0.1;
    cfg.max_signals_per_second = 1000;
    cfg.disable_drawdown_gate  = true;  // bypass gate so we can accumulate beyond limit
    RiskManager rm(cfg);

    // Accumulate 3 * 0.4 = 1.2 bias >> 0.1 max_drawdown.
    for (int i = 0; i < 3; ++i) rm.evaluate(make_signal(0.4, 0.1, 0.05, 0.8));

    // Budget must clamp at 0.0, not go negative.
    EXPECT_DOUBLE_EQ(rm.get_drawdown_budget_remaining(), 0.0);
}

TEST(RiskManagerTest, test_risk_manager_evaluate_with_reason_confidence_reason) {
    RiskManager rm(default_config());
    std::string reason;
    rm.evaluate_with_reason(make_signal(0.1, 0.1, 0.05, 0.01 /*below 0.1 min*/), reason);
    EXPECT_EQ(reason, "confidence_below_minimum");
}

// ============================================================
// Test: reset_drawdown() clears the accumulator without touching the rate window.
// ============================================================
TEST(RiskManagerTest, test_risk_manager_reset_drawdown_clears_accumulator) {
    RiskManager::Config cfg = default_config();
    cfg.max_drawdown           = 0.5;
    cfg.max_signals_per_second = 1000;
    RiskManager rm(cfg);

    auto sig = make_signal(0.4, 0.1, 0.05, 0.8);
    EXPECT_TRUE(rm.evaluate(sig));  // cumulative = 0.4

    // Second signal would push cumulative to 0.8 > 0.5 — blocked.
    EXPECT_FALSE(rm.evaluate(sig));
    EXPECT_EQ(rm.get_stats().signals_blocked_drawdown.load(), 1u);

    // reset_drawdown() must clear accumulator so the next signal passes.
    rm.reset_drawdown();
    EXPECT_DOUBLE_EQ(rm.get_cumulative_bias(), 0.0);
    EXPECT_TRUE(rm.evaluate(sig));
}

// ============================================================
// Test: reset_drawdown() does NOT reset rate-limit window.
// ============================================================
TEST(RiskManagerTest, test_risk_manager_reset_drawdown_does_not_reset_rate_limit) {
    RiskManager::Config cfg = default_config();
    cfg.max_signals_per_second = 2;
    cfg.max_drawdown           = 100.0;  // no drawdown interference
    RiskManager rm(cfg);

    auto sig = make_signal(0.1, 0.1, 0.05, 0.8);
    EXPECT_TRUE(rm.evaluate(sig));   // rate count = 1
    EXPECT_TRUE(rm.evaluate(sig));   // rate count = 2
    EXPECT_FALSE(rm.evaluate(sig));  // rate count = 3 > 2, blocked

    // reset_drawdown() must NOT clear the rate window — signal must still be blocked.
    rm.reset_drawdown();
    EXPECT_FALSE(rm.evaluate(sig))
        << "reset_drawdown() must not reset the rate-limit window";
    EXPECT_GE(rm.get_stats().signals_blocked_rate.load(), 2u);
}

// ============================================================
// Cycle 22: evaluate_with_reason for rate-limit and drawdown gates
// ============================================================

TEST(RiskManagerTest, test_risk_manager_evaluate_with_reason_rate_limit_reason) {
    RiskManager::Config cfg = default_config();
    cfg.max_signals_per_second = 1;
    RiskManager rm(cfg);

    std::string reason;
    // First signal passes.
    rm.evaluate_with_reason(make_signal(), reason);
    EXPECT_TRUE(reason.empty());

    // Immediately fire a second — rate limit fires.
    bool result = rm.evaluate_with_reason(make_signal(), reason);
    EXPECT_FALSE(result);
    EXPECT_EQ(reason, "rate_limit_exceeded");
}

TEST(RiskManagerTest, test_risk_manager_evaluate_with_reason_drawdown_reason) {
    RiskManager::Config cfg = default_config();
    cfg.max_drawdown = 0.01;  // tiny window
    RiskManager rm(cfg);

    std::string reason;
    // Fire crash signals to bust drawdown; unique-enough so rate limit won't fire.
    for (int i = 0; i < 5; ++i) {
        TradeSignal s;
        s.delta_bias_shift      = 1.0;
        s.volatility_adjustment = 0.1;
        s.spread_modifier       = 0.05;
        s.confidence            = 0.8;
        s.timestamp_ns          = static_cast<uint64_t>(i + 1);
        rm.evaluate_with_reason(s, reason);
    }
    EXPECT_EQ(reason, "drawdown_limit_exceeded");
}

TEST(RiskManagerTest, test_risk_manager_evaluate_with_reason_does_not_disturb_existing_callback) {
    // Verify the internal callback swap doesn't clobber a caller-registered callback.
    RiskManager rm(default_config());
    std::atomic<int> alert_count{0};
    rm.set_alert_callback([&alert_count](const std::string&, const TradeSignal&) {
        ++alert_count;
    });

    std::string reason;
    rm.evaluate_with_reason(make_signal(5.0, 0.1, 0.05, 0.8), reason);  // magnitude block
    EXPECT_FALSE(reason.empty());
    // The original callback must still have fired.
    EXPECT_EQ(alert_count.load(), 1);
}

// ============================================================
// Test: get_rate_limit_utilization() reports window usage.
// ============================================================
TEST(RiskManagerTest, test_risk_manager_rate_limit_utilization_zero_at_start) {
    RiskManager rm(default_config());
    EXPECT_DOUBLE_EQ(rm.get_rate_limit_utilization(), 0.0);
}

TEST(RiskManagerTest, test_risk_manager_rate_limit_utilization_increases_with_signals) {
    RiskManager::Config cfg = default_config();
    cfg.max_signals_per_second = 10;
    RiskManager rm(cfg);
    auto sig = make_signal(0.1, 0.1, 0.05, 0.8);
    rm.evaluate(sig);
    rm.evaluate(sig);
    rm.evaluate(sig);
    EXPECT_DOUBLE_EQ(rm.get_rate_limit_utilization(), 0.3);
}

TEST(RiskManagerTest, test_risk_manager_rate_limit_utilization_caps_at_one) {
    RiskManager::Config cfg = default_config();
    cfg.max_signals_per_second = 2;
    RiskManager rm(cfg);
    auto sig = make_signal(0.1, 0.1, 0.05, 0.8);
    for (int i = 0; i < 5; ++i) rm.evaluate(sig);
    EXPECT_LE(rm.get_rate_limit_utilization(), 1.0);
}

// ============================================================
// Cycle 25: evaluate_with_reason — position and PnL gates
// ============================================================

TEST(RiskManagerTest, test_risk_manager_evaluate_with_reason_position_limit_reason) {
    RiskManager rm(default_config());

    // Set a tight position limit so the projected position exceeds it.
    RiskManager::PositionState pos;
    pos.net_position    = 0.0;
    pos.position_limit  = 0.05;  // tiny limit
    pos.pnl             = 100.0; // healthy PnL so PnL gate doesn't fire first
    pos.pnl_limit       = -100.0;
    rm.update_position(pos);

    std::string reason;
    // Signal with delta_bias_shift=0.1 projects net_position to 0.1 > 0.05 limit.
    TradeSignal s;
    s.delta_bias_shift      = 0.1;
    s.volatility_adjustment = 0.1;
    s.spread_modifier       = 0.05;
    s.confidence            = 0.8;
    s.timestamp_ns          = 1;
    bool result = rm.evaluate_with_reason(s, reason);
    EXPECT_FALSE(result);
    EXPECT_EQ(reason, "position_limit");
}

TEST(RiskManagerTest, test_risk_manager_evaluate_with_reason_pnl_limit_reason) {
    RiskManager rm(default_config());

    RiskManager::PositionState pos;
    pos.net_position    = 0.0;
    pos.position_limit  = 100.0;  // generous limit
    pos.pnl             = -999.0; // well below PnL limit
    pos.pnl_limit       = -1.0;   // PnL limit
    rm.update_position(pos);

    std::string reason;
    bool result = rm.evaluate_with_reason(make_signal(), reason);
    EXPECT_FALSE(result);
    EXPECT_EQ(reason, "pnl_limit");
}

// ============================================================
// Cycle 25: MetricsLogger + RiskManager integration
// ============================================================

TEST(RiskManagerTest, test_risk_manager_evaluate_with_reason_integrates_with_metrics_logger) {
    // When a signal is blocked, pass the reason to MetricsLogger::log_risk_rejection.
    // Verify both components work together without throwing.
    RiskManager rm(default_config());

    MetricsLogger::Config mlcfg;
    mlcfg.log_file_path         = "/tmp/test_rm_metrics_integration.log";
    mlcfg.format                = MetricsLogger::OutputFormat::JSON;
    mlcfg.enable_console_output = false;
    mlcfg.flush_interval        = std::chrono::milliseconds{10};
    MetricsLogger logger(mlcfg);

    std::string reason;
    bool passed = rm.evaluate_with_reason(make_signal(5.0, 0.1, 0.05, 0.8), reason);
    ASSERT_FALSE(passed);
    ASSERT_FALSE(reason.empty());

    // Log the rejection — must not throw.
    EXPECT_NO_THROW(logger.log_risk_rejection(reason, 5.0, 0.8));
    EXPECT_EQ(logger.get_log_entry_count(), uint64_t{1});

    std::remove("/tmp/test_rm_metrics_integration.log");
}

// ============================================================
// Test: Stats::blocked_total() sums gate counters correctly.
// ============================================================
TEST(RiskManagerTest, test_risk_manager_stats_blocked_total_sums_all_gates) {
    RiskManager::Config cfg = default_config();
    cfg.max_signals_per_second = 1000;
    RiskManager rm(cfg);

    rm.evaluate(make_signal(5.0, 0.1, 0.05, 0.8));  // magnitude block
    rm.evaluate(make_signal(0.1, 0.1, 0.05, 0.01)); // confidence block

    uint64_t total = rm.get_stats().blocked_total();
    EXPECT_EQ(total, 2u) << "blocked_total must equal magnitude + confidence blocks";
}

TEST(RiskManagerTest, test_risk_manager_blocked_rate_zero_when_all_pass) {
    RiskManager rm(default_config());
    rm.evaluate(make_signal(0.1, 0.1, 0.05, 0.8));
    EXPECT_DOUBLE_EQ(rm.get_blocked_rate(), 0.0)
        << "Blocked rate must be 0 when all signals pass";
}

TEST(RiskManagerTest, test_risk_manager_blocked_rate_correct_fraction) {
    RiskManager rm(default_config());
    rm.evaluate(make_signal(0.1, 0.1, 0.05, 0.8));  // passes
    rm.evaluate(make_signal(5.0, 0.1, 0.05, 0.8));  // blocked magnitude

    // 1 blocked out of 2 evaluated -> 0.5
    EXPECT_NEAR(rm.get_blocked_rate(), 0.5, 1e-9);
}

// ============================================================
// Test: try_evaluate() — nullopt on pass.
// ============================================================
TEST(RiskManagerTest, test_try_evaluate_returns_nullopt_on_pass) {
    RiskManager rm(default_config());
    auto result = rm.try_evaluate(make_signal(0.1, 0.1, 0.05, 0.8));
    EXPECT_FALSE(result.has_value()) << "try_evaluate must return nullopt when signal passes";
}

// ============================================================
// Test: try_evaluate() — returns reason string on block.
// ============================================================
TEST(RiskManagerTest, test_try_evaluate_returns_reason_on_block) {
    RiskManager rm(default_config());
    auto result = rm.try_evaluate(make_signal(5.0, 0.1, 0.05, 0.8));
    ASSERT_TRUE(result.has_value()) << "try_evaluate must return a reason when signal is blocked";
    EXPECT_EQ(*result, "magnitude_exceeded");
}

// ============================================================
// Test: try_evaluate() — confidence block returns correct reason.
// ============================================================
TEST(RiskManagerTest, test_try_evaluate_confidence_reason) {
    RiskManager rm(default_config());
    auto result = rm.try_evaluate(make_signal(0.1, 0.1, 0.05, 0.01));
    ASSERT_TRUE(result.has_value());
    EXPECT_EQ(*result, "confidence_below_minimum");
}

// ============================================================
// Cycle 28: try_evaluate — remaining gates + consistency with evaluate_with_reason
// ============================================================

TEST(RiskManagerTest, test_try_evaluate_rate_limit_returns_reason) {
    RiskManager::Config cfg = default_config();
    cfg.max_signals_per_second = 1;
    RiskManager rm(cfg);

    // First signal passes.
    EXPECT_FALSE(rm.try_evaluate(make_signal()).has_value());

    // Immediately fire another — rate limit.
    auto result = rm.try_evaluate(make_signal());
    ASSERT_TRUE(result.has_value());
    EXPECT_EQ(*result, "rate_limit_exceeded");
}

TEST(RiskManagerTest, test_try_evaluate_position_limit_returns_reason) {
    RiskManager rm(default_config());
    RiskManager::PositionState pos;
    pos.net_position   = 0.0;
    pos.position_limit = 0.05;
    pos.pnl            = 100.0;
    pos.pnl_limit      = -100.0;
    rm.update_position(pos);

    TradeSignal s;
    s.delta_bias_shift      = 0.1;
    s.volatility_adjustment = 0.1;
    s.spread_modifier       = 0.05;
    s.confidence            = 0.8;
    s.timestamp_ns          = 1;
    auto result = rm.try_evaluate(s);
    ASSERT_TRUE(result.has_value());
    EXPECT_EQ(*result, "position_limit");
}

TEST(RiskManagerTest, test_try_evaluate_agrees_with_evaluate_with_reason) {
    // try_evaluate and evaluate_with_reason must agree on blocked/passed and reason.
    RiskManager rm(default_config());
    TradeSignal blocked_sig = make_signal(5.0, 0.1, 0.05, 0.8);

    std::string reason_wr;
    bool passed_wr = rm.evaluate_with_reason(blocked_sig, reason_wr);

    // Reset rate-limit window before second call.
    rm.reset();
    auto result_te = rm.try_evaluate(blocked_sig);

    EXPECT_FALSE(passed_wr);
    EXPECT_TRUE(result_te.has_value());
    EXPECT_EQ(reason_wr, *result_te)
        << "try_evaluate and evaluate_with_reason must return the same reason";
}

// ============================================================
// Test: snapshot() captures consistent state.
// ============================================================
TEST(RiskManagerTest, test_risk_manager_snapshot_reflects_current_state) {
    RiskManager::Config cfg = default_config();
    cfg.max_drawdown           = 5.0;
    cfg.max_signals_per_second = 1000;
    RiskManager rm(cfg);

    // Process a signal to accumulate some drawdown.
    rm.evaluate(make_signal(1.0, 0.1, 0.05, 0.8));  // cumulative_bias = 1.0

    auto snap = rm.snapshot();

    EXPECT_DOUBLE_EQ(snap.config.max_drawdown, 5.0);
    EXPECT_DOUBLE_EQ(snap.cumulative_bias, 1.0);
    EXPECT_NEAR(snap.drawdown_budget_remaining, 4.0, 1e-9)
        << "Drawdown budget must be max_drawdown - |cumulative_bias|";
    EXPECT_EQ(snap.signals_passed, 1u);
    EXPECT_EQ(snap.signals_blocked_total, 0u);
}

TEST(RiskManagerTest, test_risk_manager_snapshot_blocked_total_matches_stats) {
    RiskManager rm(default_config());
    rm.evaluate(make_signal(5.0, 0.1, 0.05, 0.8));  // magnitude block

    auto snap = rm.snapshot();
    EXPECT_EQ(snap.signals_blocked_total, rm.get_stats().blocked_total())
        << "snapshot().signals_blocked_total must equal get_stats().blocked_total()";
}

// ============================================================
// Test: get_total_evaluated() = passed + blocked.
// ============================================================
TEST(RiskManagerTest, test_risk_manager_get_total_evaluated_zero_at_start) {
    RiskManager rm(default_config());
    EXPECT_EQ(rm.get_total_evaluated(), 0u);
}

TEST(RiskManagerTest, test_risk_manager_get_total_evaluated_counts_all_signals) {
    RiskManager rm(default_config());
    rm.evaluate(make_signal(0.1, 0.1, 0.05, 0.8));  // passes
    rm.evaluate(make_signal(5.0, 0.1, 0.05, 0.8));  // blocked magnitude
    rm.evaluate(make_signal(0.1, 0.1, 0.05, 0.8));  // passes
    EXPECT_EQ(rm.get_total_evaluated(), 3u);
}

// ---------------------------------------------------------------------------
// Cycle 33: set_position_limit()
// ---------------------------------------------------------------------------

TEST(RiskManagerTest, test_set_position_limit_updates_get_position) {
    RiskManager rm(default_config());
    rm.set_position_limit(2.0, -20.0);
    auto pos = rm.get_position();
    EXPECT_DOUBLE_EQ(pos.position_limit, 2.0);
    EXPECT_DOUBLE_EQ(pos.pnl_limit,      -20.0);
}

TEST(RiskManagerTest, test_set_position_limit_preserves_net_position) {
    RiskManager rm(default_config());
    // Set a net_position via update_position first.
    RiskManager::PositionState st;
    st.net_position   = 0.5;
    st.position_limit = 1.0;
    st.pnl            = 5.0;
    st.pnl_limit      = -10.0;
    rm.update_position(st);

    // Only update limits — net_position and pnl must remain unchanged.
    rm.set_position_limit(3.0, -30.0);
    auto pos = rm.get_position();
    EXPECT_DOUBLE_EQ(pos.net_position,   0.5) << "net_position must not change";
    EXPECT_DOUBLE_EQ(pos.pnl,            5.0) << "pnl must not change";
    EXPECT_DOUBLE_EQ(pos.position_limit, 3.0) << "position_limit must be updated";
    EXPECT_DOUBLE_EQ(pos.pnl_limit,    -30.0) << "pnl_limit must be updated";
}

TEST(RiskManagerTest, test_set_position_limit_default_pnl_limit) {
    RiskManager rm(default_config());
    rm.set_position_limit(5.0);  // pnl_limit defaults to -10.0
    auto pos = rm.get_position();
    EXPECT_DOUBLE_EQ(pos.position_limit, 5.0);
    EXPECT_DOUBLE_EQ(pos.pnl_limit,    -10.0);
}

// ---------------------------------------------------------------------------
// Cycle 33: evaluate_batch()
// ---------------------------------------------------------------------------

TEST(RiskManagerTest, test_evaluate_batch_empty_returns_empty) {
    RiskManager rm(default_config());
    auto results = rm.evaluate_batch({});
    EXPECT_TRUE(results.empty());
}

TEST(RiskManagerTest, test_evaluate_batch_all_pass) {
    RiskManager rm(default_config());
    std::vector<TradeSignal> signals = {
        make_signal(0.1, 0.1, 0.05, 0.8),
        make_signal(0.2, 0.1, 0.05, 0.9),
    };
    auto results = rm.evaluate_batch(signals);
    ASSERT_EQ(results.size(), 2u);
    EXPECT_TRUE(results[0]);
    EXPECT_TRUE(results[1]);
}

TEST(RiskManagerTest, test_evaluate_batch_mixed_pass_block) {
    RiskManager rm(default_config());
    std::vector<TradeSignal> signals = {
        make_signal(0.1, 0.1, 0.05, 0.8),   // passes
        make_signal(5.0, 0.1, 0.05, 0.8),   // blocked (magnitude)
    };
    auto results = rm.evaluate_batch(signals);
    ASSERT_EQ(results.size(), 2u);
    EXPECT_TRUE(results[0])  << "First signal should pass";
    EXPECT_FALSE(results[1]) << "Second signal should be blocked";
}

// ---------------------------------------------------------------------------
// Cycle 33: set_position_limit() enforcement
// ---------------------------------------------------------------------------

TEST(RiskManagerTest, test_set_position_limit_blocks_signal_exceeding_limit) {
    RiskManager rm(default_config());
    // Tighten position limit to 0.05 so a delta_bias_shift of 0.1 exceeds it.
    rm.set_position_limit(0.05, -100.0);
    // net_position = 0, projected = 0 + 0.1 = 0.1 > 0.05 → blocked.
    TradeSignal sig = make_signal(0.1, 0.1, 0.05, 0.8);
    EXPECT_FALSE(rm.evaluate(sig))
        << "signal exceeding position_limit must be blocked";
    EXPECT_GT(rm.get_stats().signals_blocked_position.load(), 0u);
}

TEST(RiskManagerTest, test_set_position_limit_allows_signal_within_limit) {
    RiskManager rm(default_config());
    // Generous limit: projected = 0 + 0.1 = 0.1 < 10.0 → passes.
    rm.set_position_limit(10.0, -100.0);
    TradeSignal sig = make_signal(0.1, 0.1, 0.05, 0.8);
    EXPECT_TRUE(rm.evaluate(sig))
        << "signal within position_limit must pass";
}

TEST(RiskManagerTest, test_set_position_limit_pnl_blocks_on_breach) {
    RiskManager rm(default_config());
    // Set pnl_limit to -1.0, then put PnL at -5.0 (breach).
    rm.set_position_limit(100.0, -1.0);
    RiskManager::PositionState pos;
    pos.net_position   = 0.0;
    pos.position_limit = 100.0;
    pos.pnl            = -5.0;
    pos.pnl_limit      = -1.0;
    rm.update_position(pos);
    TradeSignal sig = make_signal(0.1, 0.1, 0.05, 0.8);
    EXPECT_FALSE(rm.evaluate(sig))
        << "signal must be blocked when pnl < pnl_limit";
}
