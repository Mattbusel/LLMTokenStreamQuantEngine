#include <gtest/gtest.h>

#ifdef LLMQUANT_TOKEN_COST_BUDGET_ENABLED

#include "TokenCostBudget.h"

#include <atomic>

using namespace llmquant;

namespace {

// ============================================================
// Construction
// ============================================================

TEST(TokenCostBudget, DefaultConstructsAsNormal) {
    TokenCostBudget budget;
    EXPECT_EQ(budget.current_action(), TokenCostBudget::Action::Normal);
    EXPECT_EQ(budget.total_tokens(),   0u);
    EXPECT_NEAR(budget.total_cost_usd(), 0.0, 1e-12);
}

// ============================================================
// action_name
// ============================================================

TEST(TokenCostBudget, ActionNamesAreCorrect) {
    using A = TokenCostBudget::Action;
    EXPECT_STREQ(TokenCostBudget::action_name(A::Normal), "Normal");
    EXPECT_STREQ(TokenCostBudget::action_name(A::Warn),   "Warn");
    EXPECT_STREQ(TokenCostBudget::action_name(A::Halt),   "Halt");
}

// ============================================================
// consume: cost accumulation
// ============================================================

TEST(TokenCostBudget, SingleConsumeAccumulatesTokens) {
    TokenCostBudget budget;
    budget.consume(10);
    EXPECT_EQ(budget.total_tokens(), 10u);
}

TEST(TokenCostBudget, CostCalculatedCorrectly) {
    TokenCostBudget::Config cfg;
    cfg.cost_per_1k_tokens = 1.0;  // $1 per 1k tokens
    cfg.soft_budget_usd    = 100.0;
    cfg.hard_budget_usd    = 200.0;
    TokenCostBudget budget(cfg);

    budget.consume(500);
    EXPECT_NEAR(budget.total_cost_usd(), 0.5, 1e-9);
}

// ============================================================
// Warn tier
// ============================================================

TEST(TokenCostBudget, ConsumeReturnsWarnWhenSoftExceeded) {
    TokenCostBudget::Config cfg;
    cfg.cost_per_1k_tokens = 1.0;
    cfg.soft_budget_usd    = 0.5;
    cfg.hard_budget_usd    = 2.0;
    TokenCostBudget budget(cfg);

    budget.consume(400);  // $0.40 — Normal
    EXPECT_EQ(budget.consume(200), TokenCostBudget::Action::Warn);  // $0.60 > soft
}

TEST(TokenCostBudget, WarnCallbackFiredOnce) {
    TokenCostBudget::Config cfg;
    cfg.cost_per_1k_tokens = 1.0;
    cfg.soft_budget_usd    = 0.5;
    cfg.hard_budget_usd    = 2.0;
    TokenCostBudget budget(cfg);

    std::atomic<int> fired{0};
    budget.set_warning_callback([&](double) { fired.fetch_add(1); });

    budget.consume(600);  // crosses soft
    budget.consume(100);  // already in Warn, no re-fire
    EXPECT_EQ(fired.load(), 1);
}

// ============================================================
// Halt tier
// ============================================================

TEST(TokenCostBudget, ConsumeReturnsHaltWhenHardExceeded) {
    TokenCostBudget::Config cfg;
    cfg.cost_per_1k_tokens = 1.0;
    cfg.soft_budget_usd    = 0.5;
    cfg.hard_budget_usd    = 1.0;
    TokenCostBudget budget(cfg);

    EXPECT_EQ(budget.consume(1100), TokenCostBudget::Action::Halt);
}

TEST(TokenCostBudget, HaltCallbackFiredOnce) {
    TokenCostBudget::Config cfg;
    cfg.cost_per_1k_tokens = 1.0;
    cfg.soft_budget_usd    = 0.5;
    cfg.hard_budget_usd    = 1.0;
    TokenCostBudget budget(cfg);

    std::atomic<int> fired{0};
    budget.set_halt_callback([&](double) { fired.fetch_add(1); });

    budget.consume(2000);  // crosses hard
    budget.consume(100);   // stays in Halt, no re-fire
    EXPECT_EQ(fired.load(), 1);
}

// ============================================================
// budget_fraction
// ============================================================

TEST(TokenCostBudget, BudgetFractionScalesCorrectly) {
    TokenCostBudget::Config cfg;
    cfg.cost_per_1k_tokens = 1.0;
    cfg.soft_budget_usd    = 0.5;
    cfg.hard_budget_usd    = 2.0;
    TokenCostBudget budget(cfg);

    budget.consume(1000);  // $1.00 of $2.00 hard = 0.5
    EXPECT_NEAR(budget.budget_fraction(), 0.5, 1e-9);
}

// ============================================================
// tokens_remaining
// ============================================================

TEST(TokenCostBudget, TokensRemainingDecrementsWithConsumption) {
    TokenCostBudget::Config cfg;
    cfg.cost_per_1k_tokens = 1.0;
    cfg.soft_budget_usd    = 0.5;
    cfg.hard_budget_usd    = 2.0;
    TokenCostBudget budget(cfg);

    uint64_t before = budget.tokens_remaining();
    budget.consume(500);
    uint64_t after  = budget.tokens_remaining();
    EXPECT_LT(after, before);
}

TEST(TokenCostBudget, TokensRemainingIsZeroWhenExceeded) {
    TokenCostBudget::Config cfg;
    cfg.cost_per_1k_tokens = 1.0;
    cfg.soft_budget_usd    = 0.5;
    cfg.hard_budget_usd    = 1.0;
    TokenCostBudget budget(cfg);

    budget.consume(1500);
    EXPECT_EQ(budget.tokens_remaining(), 0u);
}

// ============================================================
// reset
// ============================================================

TEST(TokenCostBudget, ResetRestoresNormalFromHalt) {
    TokenCostBudget::Config cfg;
    cfg.cost_per_1k_tokens = 1.0;
    cfg.soft_budget_usd    = 0.1;
    cfg.hard_budget_usd    = 0.2;
    TokenCostBudget budget(cfg);

    budget.consume(500);  // past hard
    EXPECT_EQ(budget.current_action(), TokenCostBudget::Action::Halt);

    budget.reset();
    EXPECT_EQ(budget.current_action(), TokenCostBudget::Action::Normal);
    EXPECT_EQ(budget.total_tokens(),   0u);
}

// ============================================================
// update_config
// ============================================================

TEST(TokenCostBudget, UpdateConfigChangesThresholds) {
    TokenCostBudget budget;
    TokenCostBudget::Config cfg;
    cfg.cost_per_1k_tokens = 0.001;
    cfg.soft_budget_usd    = 100.0;
    cfg.hard_budget_usd    = 200.0;
    budget.update_config(cfg);
    budget.consume(1000);
    EXPECT_EQ(budget.current_action(), TokenCostBudget::Action::Normal);
}

// ============================================================
// to_stats_json
// ============================================================

TEST(TokenCostBudget, ToStatsJsonContainsKeys) {
    TokenCostBudget budget;
    budget.consume(100);
    std::string json = budget.to_stats_json();
    EXPECT_NE(json.find("action"),          std::string::npos);
    EXPECT_NE(json.find("total_tokens"),    std::string::npos);
    EXPECT_NE(json.find("total_cost_usd"),  std::string::npos);
    EXPECT_NE(json.find("budget_fraction"), std::string::npos);
    EXPECT_NE(json.find("hard_budget_usd"), std::string::npos);
}

// ============================================================
// No callbacks — no crash
// ============================================================

TEST(TokenCostBudget, NoCallbackNoCrash) {
    TokenCostBudget::Config cfg;
    cfg.cost_per_1k_tokens = 1.0;
    cfg.soft_budget_usd    = 0.01;
    cfg.hard_budget_usd    = 0.02;
    TokenCostBudget budget(cfg);
    EXPECT_NO_FATAL_FAILURE(budget.consume(1000));
}

} // namespace

#else  // LLMQUANT_TOKEN_COST_BUDGET_ENABLED not defined

TEST(TokenCostBudget, DisabledAtBuildTime) {
    SUCCEED();
}

#endif  // LLMQUANT_TOKEN_COST_BUDGET_ENABLED
