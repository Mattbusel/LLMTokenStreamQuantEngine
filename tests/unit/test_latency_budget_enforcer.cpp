#include <gtest/gtest.h>

#ifdef LLMQUANT_LATENCY_ENFORCER_ENABLED

#include "LatencyBudgetEnforcer.h"

#include <atomic>

using namespace llmquant;

namespace {

// ============================================================
// Construction
// ============================================================

TEST(LatencyBudgetEnforcer, DefaultConstructsAsNormal) {
    LatencyBudgetEnforcer lbe;
    EXPECT_EQ(lbe.current_action(), LatencyBudgetEnforcer::Action::Normal);
    EXPECT_EQ(lbe.escalation_count(), 0u);
    EXPECT_EQ(lbe.recovery_count(),   0u);
}

// ============================================================
// action_name
// ============================================================

TEST(LatencyBudgetEnforcer, ActionNamesAreCorrect) {
    using A = LatencyBudgetEnforcer::Action;
    EXPECT_STREQ(LatencyBudgetEnforcer::action_name(A::Normal),   "Normal");
    EXPECT_STREQ(LatencyBudgetEnforcer::action_name(A::Warn),     "Warn");
    EXPECT_STREQ(LatencyBudgetEnforcer::action_name(A::Throttle), "Throttle");
    EXPECT_STREQ(LatencyBudgetEnforcer::action_name(A::Drop),     "Drop");
    EXPECT_STREQ(LatencyBudgetEnforcer::action_name(A::Breaker),  "Breaker");
}

// ============================================================
// Tier escalation
// ============================================================

TEST(LatencyBudgetEnforcer, CheckBelowWarnStaysNormal) {
    LatencyBudgetEnforcer::Config cfg;
    cfg.warn_us = 10;
    LatencyBudgetEnforcer lbe(cfg);
    EXPECT_EQ(lbe.check(9), LatencyBudgetEnforcer::Action::Normal);
}

TEST(LatencyBudgetEnforcer, CheckAtWarnReturnsWarn) {
    LatencyBudgetEnforcer::Config cfg;
    cfg.warn_us     = 5;
    cfg.throttle_us = 8;
    cfg.drop_us     = 12;
    cfg.breaker_us  = 20;
    LatencyBudgetEnforcer lbe(cfg);
    EXPECT_EQ(lbe.check(5), LatencyBudgetEnforcer::Action::Warn);
}

TEST(LatencyBudgetEnforcer, CheckAtThrottleReturnsThrottle) {
    LatencyBudgetEnforcer::Config cfg;
    cfg.warn_us = 5; cfg.throttle_us = 8; cfg.drop_us = 12; cfg.breaker_us = 20;
    LatencyBudgetEnforcer lbe(cfg);
    EXPECT_EQ(lbe.check(8), LatencyBudgetEnforcer::Action::Throttle);
}

TEST(LatencyBudgetEnforcer, CheckAtDropReturnsDrop) {
    LatencyBudgetEnforcer::Config cfg;
    cfg.warn_us = 5; cfg.throttle_us = 8; cfg.drop_us = 12; cfg.breaker_us = 20;
    LatencyBudgetEnforcer lbe(cfg);
    EXPECT_EQ(lbe.check(12), LatencyBudgetEnforcer::Action::Drop);
}

TEST(LatencyBudgetEnforcer, CheckAtBreakerReturnsBreaker) {
    LatencyBudgetEnforcer::Config cfg;
    cfg.warn_us = 5; cfg.throttle_us = 8; cfg.drop_us = 12; cfg.breaker_us = 20;
    LatencyBudgetEnforcer lbe(cfg);
    EXPECT_EQ(lbe.check(25), LatencyBudgetEnforcer::Action::Breaker);
}

TEST(LatencyBudgetEnforcer, EscalationCountIncrements) {
    LatencyBudgetEnforcer::Config cfg;
    cfg.warn_us = 5; cfg.throttle_us = 10; cfg.drop_us = 15; cfg.breaker_us = 20;
    LatencyBudgetEnforcer lbe(cfg);
    lbe.check(6);   // → Warn
    lbe.check(11);  // → Throttle
    EXPECT_EQ(lbe.escalation_count(), 2u);
}

// ============================================================
// Callbacks fire on tier change
// ============================================================

TEST(LatencyBudgetEnforcer, WarnCallbackFiredOnce) {
    LatencyBudgetEnforcer::Config cfg;
    cfg.warn_us = 5; cfg.throttle_us = 10; cfg.drop_us = 15; cfg.breaker_us = 20;
    cfg.recovery_factor = 0.5;
    LatencyBudgetEnforcer lbe(cfg);

    std::atomic<int> fired{0};
    lbe.set_warn_callback([&](int64_t) { fired.fetch_add(1); });

    lbe.check(6);  // → Warn (fires)
    lbe.check(7);  // stays Warn (no re-fire)
    EXPECT_EQ(fired.load(), 1);
}

TEST(LatencyBudgetEnforcer, RecoveryCallbackFiredOnReturn) {
    LatencyBudgetEnforcer::Config cfg;
    cfg.warn_us = 5; cfg.throttle_us = 10; cfg.drop_us = 15; cfg.breaker_us = 20;
    cfg.recovery_factor = 0.5;
    LatencyBudgetEnforcer lbe(cfg);

    std::atomic<int> recovered{0};
    lbe.set_recovery_callback([&](int64_t) { recovered.fetch_add(1); });

    lbe.check(6);  // → Warn
    lbe.check(2);  // below 5*0.5=2.5 → recover
    EXPECT_EQ(recovered.load(), 1);
    EXPECT_EQ(lbe.recovery_count(), 1u);
    EXPECT_EQ(lbe.current_action(), LatencyBudgetEnforcer::Action::Normal);
}

// ============================================================
// Hysteresis prevents flapping
// ============================================================

TEST(LatencyBudgetEnforcer, HysteresisPreventsPrematureRecovery) {
    LatencyBudgetEnforcer::Config cfg;
    cfg.warn_us = 10; cfg.throttle_us = 20; cfg.drop_us = 30; cfg.breaker_us = 40;
    cfg.recovery_factor = 0.8;
    LatencyBudgetEnforcer lbe(cfg);

    lbe.check(10);  // → Warn
    // Latency drops to 9, but recovery band is 10*0.8=8 → still Warn.
    EXPECT_EQ(lbe.check(9), LatencyBudgetEnforcer::Action::Warn);
    // Latency drops to 7, below recovery band → Normal.
    EXPECT_EQ(lbe.check(7), LatencyBudgetEnforcer::Action::Normal);
}

// ============================================================
// reset
// ============================================================

TEST(LatencyBudgetEnforcer, ResetRestoresNormal) {
    LatencyBudgetEnforcer::Config cfg;
    cfg.warn_us = 5; cfg.throttle_us = 8; cfg.drop_us = 12; cfg.breaker_us = 20;
    LatencyBudgetEnforcer lbe(cfg);
    lbe.check(25);  // → Breaker
    EXPECT_EQ(lbe.current_action(), LatencyBudgetEnforcer::Action::Breaker);
    lbe.reset();
    EXPECT_EQ(lbe.current_action(), LatencyBudgetEnforcer::Action::Normal);
}

// ============================================================
// update_config
// ============================================================

TEST(LatencyBudgetEnforcer, UpdateConfigChangesThresholds) {
    LatencyBudgetEnforcer lbe;
    LatencyBudgetEnforcer::Config cfg;
    cfg.warn_us = 1; cfg.throttle_us = 2; cfg.drop_us = 3; cfg.breaker_us = 4;
    lbe.update_config(cfg);
    EXPECT_EQ(lbe.check(5), LatencyBudgetEnforcer::Action::Breaker);
}

// ============================================================
// No callback — no crash
// ============================================================

TEST(LatencyBudgetEnforcer, NoCallbackNoCrash) {
    LatencyBudgetEnforcer::Config cfg;
    cfg.warn_us = 1; cfg.throttle_us = 2; cfg.drop_us = 3; cfg.breaker_us = 4;
    LatencyBudgetEnforcer lbe(cfg);
    EXPECT_NO_FATAL_FAILURE(lbe.check(100));
}

// ============================================================
// to_stats_json
// ============================================================

TEST(LatencyBudgetEnforcer, ToStatsJsonContainsKeys) {
    LatencyBudgetEnforcer lbe;
    lbe.check(7);
    std::string json = lbe.to_stats_json();
    EXPECT_NE(json.find("action"),            std::string::npos);
    EXPECT_NE(json.find("escalation_count"),  std::string::npos);
    EXPECT_NE(json.find("recovery_count"),    std::string::npos);
    EXPECT_NE(json.find("warn_us"),           std::string::npos);
    EXPECT_NE(json.find("breaker_us"),        std::string::npos);
}

// ============================================================
// Skip a tier: Normal → Drop directly
// ============================================================

TEST(LatencyBudgetEnforcer, SkipsTierWhenLatencyJumpsHigh) {
    LatencyBudgetEnforcer::Config cfg;
    cfg.warn_us = 5; cfg.throttle_us = 8; cfg.drop_us = 12; cfg.breaker_us = 20;
    LatencyBudgetEnforcer lbe(cfg);
    // Jump directly from Normal past Warn and Throttle to Drop.
    EXPECT_EQ(lbe.check(15), LatencyBudgetEnforcer::Action::Drop);
    EXPECT_EQ(lbe.escalation_count(), 1u);
}

} // namespace

#else  // LLMQUANT_LATENCY_ENFORCER_ENABLED not defined

TEST(LatencyBudgetEnforcer, DisabledAtBuildTime) {
    SUCCEED();
}

#endif  // LLMQUANT_LATENCY_ENFORCER_ENABLED
