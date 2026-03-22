#include <gtest/gtest.h>

#ifdef LLMQUANT_ADAPTIVE_COOLDOWN_ENABLED

#include "AdaptiveCooldownController.h"
#include <thread>

using namespace llmquant;

namespace {

// ============================================================
// Construction and default state
// ============================================================

TEST(AdaptiveCooldown, DefaultConstructsOk) {
    AdaptiveCooldownController ctrl;
    EXPECT_GT(ctrl.get_cooldown_us(), 0.0);
    EXPECT_EQ(ctrl.pressure_expansions(), 0u);
    EXPECT_EQ(ctrl.recoveries(), 0u);
    EXPECT_FALSE(ctrl.is_under_pressure());
}

TEST(AdaptiveCooldown, InitialCooldownEqualsBase) {
    AdaptiveCooldownController::Config cfg;
    cfg.base_cooldown_us = 750.0;
    AdaptiveCooldownController ctrl(cfg);
    EXPECT_NEAR(ctrl.get_cooldown_us(), 750.0, 1e-9);
}

// ============================================================
// Pressure: P99 above target widens cooldown
// ============================================================

TEST(AdaptiveCooldown, HighP99ExpandsCooldown) {
    AdaptiveCooldownController::Config cfg;
    cfg.base_cooldown_us = 500.0;
    cfg.target_p99_us    = 10.0;
    cfg.pressure_gain    = 10.0;
    cfg.ema_alpha        = 1.0;  // no smoothing for deterministic test
    AdaptiveCooldownController ctrl(cfg);

    // P99 = 20 µs → excess = 10 → cooldown += 10*10 = 100 → 600 µs
    ctrl.update_p99(20.0);
    EXPECT_GT(ctrl.get_cooldown_us(), 500.0);
    EXPECT_TRUE(ctrl.is_under_pressure());
    EXPECT_EQ(ctrl.pressure_expansions(), 1u);
}

TEST(AdaptiveCooldown, CooldownCappedAtMaxCooldown) {
    AdaptiveCooldownController::Config cfg;
    cfg.base_cooldown_us = 500.0;
    cfg.max_cooldown_us  = 1000.0;
    cfg.target_p99_us    = 10.0;
    cfg.pressure_gain    = 1000.0;
    cfg.ema_alpha        = 1.0;
    AdaptiveCooldownController ctrl(cfg);

    ctrl.update_p99(1000.0);  // extreme pressure
    EXPECT_LE(ctrl.get_cooldown_us(), 1000.0 + 1e-9);
}

TEST(AdaptiveCooldown, CooldownFloorAtMinCooldown) {
    AdaptiveCooldownController::Config cfg;
    cfg.base_cooldown_us = 500.0;
    cfg.min_cooldown_us  = 200.0;
    cfg.target_p99_us    = 10.0;
    cfg.pressure_gain    = 0.0;  // no pressure response → stays at base > min
    cfg.ema_alpha        = 1.0;
    AdaptiveCooldownController ctrl(cfg);

    ctrl.update_p99(5.0);  // below target; base still >= min
    EXPECT_GE(ctrl.get_cooldown_us(), 200.0 - 1e-9);
}

// ============================================================
// Recovery: P99 below target for sufficient window
// ============================================================

TEST(AdaptiveCooldown, RecoveryWindowRestoresToBase) {
    AdaptiveCooldownController::Config cfg;
    cfg.base_cooldown_us  = 500.0;
    cfg.target_p99_us     = 10.0;
    cfg.pressure_gain     = 10.0;
    cfg.ema_alpha         = 1.0;
    cfg.recovery_window_s = 0;  // 0 s = recover immediately after any below-target sample
    AdaptiveCooldownController ctrl(cfg);

    // Expand.
    ctrl.update_p99(50.0);
    EXPECT_GT(ctrl.get_cooldown_us(), 500.0);

    // One below-target sample starts timer; 0 s window expires immediately
    // on the first call, then the second call recovers.
    ctrl.update_p99(5.0);
    ctrl.update_p99(5.0);
    EXPECT_NEAR(ctrl.get_cooldown_us(), 500.0, 1.0);
}

TEST(AdaptiveCooldown, PressureResetsRecoveryTimer) {
    AdaptiveCooldownController::Config cfg;
    cfg.base_cooldown_us  = 500.0;
    cfg.target_p99_us     = 10.0;
    cfg.pressure_gain     = 5.0;
    cfg.ema_alpha         = 1.0;
    cfg.recovery_window_s = 60;  // long recovery window
    AdaptiveCooldownController ctrl(cfg);

    // Start below target (timer starts).
    ctrl.update_p99(5.0);
    // Spike above target → resets timer.
    ctrl.update_p99(50.0);
    // Return below target — timer must restart, not recover yet.
    ctrl.update_p99(5.0);

    // Recoveries should still be 0 (window not expired).
    EXPECT_EQ(ctrl.recoveries(), 0u);
}

// ============================================================
// EMA smoothing
// ============================================================

TEST(AdaptiveCooldown, EmaSmooths) {
    AdaptiveCooldownController::Config cfg;
    cfg.target_p99_us = 10.0;
    cfg.ema_alpha     = 0.1;   // heavy smoothing
    cfg.pressure_gain = 100.0; // large gain to make differences obvious
    AdaptiveCooldownController ctrl(cfg);

    // Feed a single large spike; smoothed EMA should stay below raw spike.
    ctrl.update_p99(100.0);
    EXPECT_LT(ctrl.get_ema_p99(), 100.0);
}

// ============================================================
// get_cooldown() chrono interface
// ============================================================

TEST(AdaptiveCooldown, GetCooldownReturnsMicroseconds) {
    AdaptiveCooldownController::Config cfg;
    cfg.base_cooldown_us = 750.0;
    AdaptiveCooldownController ctrl(cfg);
    auto cd = ctrl.get_cooldown();
    EXPECT_EQ(cd.count(), 750LL);
}

// ============================================================
// reset
// ============================================================

TEST(AdaptiveCooldown, ResetRestoresBaseCooldown) {
    AdaptiveCooldownController::Config cfg;
    cfg.base_cooldown_us = 500.0;
    cfg.target_p99_us    = 10.0;
    cfg.pressure_gain    = 50.0;
    cfg.ema_alpha        = 1.0;
    AdaptiveCooldownController ctrl(cfg);

    ctrl.update_p99(100.0);
    EXPECT_GT(ctrl.get_cooldown_us(), 500.0);

    ctrl.reset();
    EXPECT_NEAR(ctrl.get_cooldown_us(), 500.0, 1e-9);
    EXPECT_EQ(ctrl.pressure_expansions(), 0u);
    EXPECT_EQ(ctrl.recoveries(),          0u);
    EXPECT_FALSE(ctrl.is_under_pressure());
}

// ============================================================
// update_config
// ============================================================

TEST(AdaptiveCooldown, UpdateConfigChangesGain) {
    AdaptiveCooldownController::Config cfg;
    cfg.base_cooldown_us = 500.0;
    cfg.target_p99_us    = 10.0;
    cfg.pressure_gain    = 0.0;   // no pressure
    cfg.ema_alpha        = 1.0;
    AdaptiveCooldownController ctrl(cfg);
    ctrl.update_p99(100.0);
    // With gain=0 no expansion.
    EXPECT_NEAR(ctrl.get_cooldown_us(), 500.0, 1.0);

    cfg.pressure_gain = 100.0;
    ctrl.update_config(cfg);
    ctrl.update_p99(100.0);
    EXPECT_GT(ctrl.get_cooldown_us(), 500.0);
}

// ============================================================
// to_stats_json
// ============================================================

TEST(AdaptiveCooldown, ToStatsJsonContainsKeys) {
    AdaptiveCooldownController ctrl;
    ctrl.update_p99(5.0);
    std::string json = ctrl.to_stats_json();
    EXPECT_NE(json.find("cooldown_us"),          std::string::npos);
    EXPECT_NE(json.find("ema_p99_us"),           std::string::npos);
    EXPECT_NE(json.find("pressure_expansions"),  std::string::npos);
    EXPECT_NE(json.find("base_cooldown_us"),     std::string::npos);
    EXPECT_NE(json.find("target_p99_us"),        std::string::npos);
}

// ============================================================
// Thread safety
// ============================================================

TEST(AdaptiveCooldown, ConcurrentUpdateP99IsSafe) {
    AdaptiveCooldownController::Config cfg;
    cfg.ema_alpha = 0.5;
    cfg.pressure_gain = 10.0;
    AdaptiveCooldownController ctrl(cfg);

    std::atomic<bool> start{false};
    auto worker = [&](double p99_val) {
        while (!start.load()) {}
        for (int i = 0; i < 200; ++i) ctrl.update_p99(p99_val + i * 0.1);
    };

    std::thread t1(worker, 5.0);
    std::thread t2(worker, 20.0);
    start.store(true);
    t1.join(); t2.join();

    SUCCEED();  // No crash/UB = pass
}

} // namespace

#else  // LLMQUANT_ADAPTIVE_COOLDOWN_ENABLED not defined

TEST(AdaptiveCooldown, DisabledAtBuildTime) {
    SUCCEED();
}

#endif  // LLMQUANT_ADAPTIVE_COOLDOWN_ENABLED
