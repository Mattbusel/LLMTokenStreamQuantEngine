#include <gtest/gtest.h>

#ifdef LLMQUANT_CONTEXT_BUDGET_ENABLED

#include "ContextWindowBudget.h"

using namespace llmquant;

namespace {

// ============================================================
// Construction
// ============================================================

TEST(ContextWindowBudget, DefaultConstructsNormal) {
    ContextWindowBudget bud;
    EXPECT_EQ(bud.current_tier(), ContextWindowBudget::Tier::Normal);
    EXPECT_EQ(bud.tokens_used(), 0u);
    EXPECT_NEAR(bud.fill_fraction(), 0.0, 1e-12);
}

// ============================================================
// tier_name
// ============================================================

TEST(ContextWindowBudget, TierNames) {
    EXPECT_STREQ(ContextWindowBudget::tier_name(ContextWindowBudget::Tier::Normal),   "Normal");
    EXPECT_STREQ(ContextWindowBudget::tier_name(ContextWindowBudget::Tier::Warn),     "Warn");
    EXPECT_STREQ(ContextWindowBudget::tier_name(ContextWindowBudget::Tier::Critical), "Critical");
    EXPECT_STREQ(ContextWindowBudget::tier_name(ContextWindowBudget::Tier::Overflow), "Overflow");
}

// ============================================================
// Warn tier
// ============================================================

TEST(ContextWindowBudget, WarnTierTriggeredAtSoftFraction) {
    ContextWindowBudget::Config cfg;
    cfg.capacity      = 1000;
    cfg.soft_fraction = 0.5;
    cfg.hard_fraction = 0.9;
    ContextWindowBudget bud(cfg);

    auto tier = bud.consume(499);
    EXPECT_EQ(tier, ContextWindowBudget::Tier::Normal);

    tier = bud.consume(1);  // now at 500 = 50%
    EXPECT_EQ(tier, ContextWindowBudget::Tier::Warn);
}

// ============================================================
// Critical tier
// ============================================================

TEST(ContextWindowBudget, CriticalTierTriggered) {
    ContextWindowBudget::Config cfg;
    cfg.capacity      = 100;
    cfg.soft_fraction = 0.5;
    cfg.hard_fraction = 0.9;
    ContextWindowBudget bud(cfg);

    bud.consume(89);
    auto tier = bud.consume(1);  // 90 = 90%
    EXPECT_EQ(tier, ContextWindowBudget::Tier::Critical);
}

// ============================================================
// Overflow tier
// ============================================================

TEST(ContextWindowBudget, OverflowTierAtCapacity) {
    ContextWindowBudget::Config cfg;
    cfg.capacity      = 100;
    cfg.soft_fraction = 0.5;
    cfg.hard_fraction = 0.9;
    ContextWindowBudget bud(cfg);

    bud.consume(99);
    auto tier = bud.consume(1);  // 100 = 100%
    EXPECT_EQ(tier, ContextWindowBudget::Tier::Overflow);
}

// ============================================================
// Warn callback fires once
// ============================================================

TEST(ContextWindowBudget, WarnCallbackFiresOnce) {
    ContextWindowBudget::Config cfg;
    cfg.capacity      = 100;
    cfg.soft_fraction = 0.5;
    cfg.hard_fraction = 0.9;
    int fires = 0;
    cfg.on_warn = [&](uint64_t, uint64_t) { ++fires; };
    ContextWindowBudget bud(cfg);

    bud.consume(50);  // warn
    bud.consume(10);  // still warn — no second fire
    bud.consume(10);  // still warn
    EXPECT_EQ(fires, 1);
    EXPECT_EQ(bud.warn_events(), 1u);
}

// ============================================================
// Critical callback fires once
// ============================================================

TEST(ContextWindowBudget, CriticalCallbackFiresOnce) {
    ContextWindowBudget::Config cfg;
    cfg.capacity      = 100;
    cfg.soft_fraction = 0.5;
    cfg.hard_fraction = 0.9;
    int fires = 0;
    cfg.on_critical = [&](uint64_t, uint64_t) { ++fires; };
    ContextWindowBudget bud(cfg);

    bud.consume(90);  // warn + critical
    bud.consume(5);   // still critical
    EXPECT_EQ(fires, 1);
    EXPECT_EQ(bud.critical_events(), 1u);
}

// ============================================================
// Overflow callback fires once
// ============================================================

TEST(ContextWindowBudget, OverflowCallbackFiresOnce) {
    ContextWindowBudget::Config cfg;
    cfg.capacity      = 50;
    cfg.soft_fraction = 0.5;
    cfg.hard_fraction = 0.9;
    int fires = 0;
    cfg.on_overflow = [&](uint64_t, uint64_t) { ++fires; };
    ContextWindowBudget bud(cfg);

    bud.consume(50);   // overflow
    bud.consume(100);  // already overflowed
    EXPECT_EQ(fires, 1);
    EXPECT_EQ(bud.overflow_events(), 1u);
}

// ============================================================
// fill_fraction
// ============================================================

TEST(ContextWindowBudget, FillFractionCorrect) {
    ContextWindowBudget::Config cfg;
    cfg.capacity = 200;
    ContextWindowBudget bud(cfg);
    bud.consume(100);
    EXPECT_NEAR(bud.fill_fraction(), 0.5, 1e-9);
}

// ============================================================
// tokens_remaining
// ============================================================

TEST(ContextWindowBudget, TokensRemainingDecreases) {
    ContextWindowBudget::Config cfg;
    cfg.capacity = 100;
    ContextWindowBudget bud(cfg);
    EXPECT_EQ(bud.tokens_remaining(), 100u);
    bud.consume(40);
    EXPECT_EQ(bud.tokens_remaining(), 60u);
}

// ============================================================
// reset restores state and re-fires callbacks
// ============================================================

TEST(ContextWindowBudget, ResetAllowsCallbacksToFireAgain) {
    ContextWindowBudget::Config cfg;
    cfg.capacity      = 100;
    cfg.soft_fraction = 0.5;
    cfg.hard_fraction = 0.9;
    int fires = 0;
    cfg.on_warn = [&](uint64_t, uint64_t) { ++fires; };
    ContextWindowBudget bud(cfg);

    bud.consume(60);   // fires warn (fires == 1)
    bud.reset();
    bud.consume(60);   // fires warn again (fires == 2)
    EXPECT_EQ(fires, 2);
    EXPECT_EQ(bud.total_resets(), 2u);  // update_config calls reset internally
}

// ============================================================
// update_config resets state
// ============================================================

TEST(ContextWindowBudget, UpdateConfigResetsState) {
    ContextWindowBudget::Config cfg;
    cfg.capacity = 100;
    ContextWindowBudget bud(cfg);
    bud.consume(90);
    EXPECT_GT(bud.tokens_used(), 0u);

    ContextWindowBudget::Config cfg2;
    cfg2.capacity = 200000;
    bud.update_config(cfg2);
    EXPECT_EQ(bud.tokens_used(), 0u);
    EXPECT_EQ(bud.current_tier(), ContextWindowBudget::Tier::Normal);
}

// ============================================================
// to_stats_json contains required keys
// ============================================================

TEST(ContextWindowBudget, ToStatsJsonContainsKeys) {
    ContextWindowBudget bud;
    bud.consume(100);
    std::string json = bud.to_stats_json();
    EXPECT_NE(json.find("tokens_used"),      std::string::npos);
    EXPECT_NE(json.find("capacity"),         std::string::npos);
    EXPECT_NE(json.find("fill_fraction"),    std::string::npos);
    EXPECT_NE(json.find("tier"),             std::string::npos);
    EXPECT_NE(json.find("total_resets"),     std::string::npos);
    EXPECT_NE(json.find("warn_events"),      std::string::npos);
    EXPECT_NE(json.find("overflow_events"),  std::string::npos);
}

// ============================================================
// No callback — no crash
// ============================================================

TEST(ContextWindowBudget, NoCallbackNoCrash) {
    ContextWindowBudget::Config cfg;
    cfg.capacity      = 10;
    cfg.soft_fraction = 0.5;
    cfg.hard_fraction = 0.9;
    ContextWindowBudget bud(cfg);
    bud.consume(5);   // warn
    bud.consume(4);   // critical
    bud.consume(1);   // overflow
    SUCCEED();
}

} // namespace

#else  // LLMQUANT_CONTEXT_BUDGET_ENABLED not defined

TEST(ContextWindowBudget, DisabledAtBuildTime) {
    SUCCEED();
}

#endif  // LLMQUANT_CONTEXT_BUDGET_ENABLED
