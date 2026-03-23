#include <gtest/gtest.h>
#include "TokenBudgetManager.h"

using namespace llmquant;

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

static TokenBudget make_budget(
    std::size_t max_in  = 1024,
    std::size_t max_out = 512,
    std::size_t max_tot = 1536,
    double      cost    = 0.000002)
{
    TokenBudget b;
    b.max_input_tokens  = max_in;
    b.max_output_tokens = max_out;
    b.max_total_tokens  = max_tot;
    b.cost_per_token_usd = cost;
    return b;
}

// ---------------------------------------------------------------------------
// Construction / initial state
// ---------------------------------------------------------------------------

TEST(TokenBudgetManager, InitialUsageIsZero) {
    TokenBudgetManager mgr(make_budget());
    EXPECT_EQ(mgr.usage().input_tokens,  0u);
    EXPECT_EQ(mgr.usage().output_tokens, 0u);
    EXPECT_NEAR(mgr.usage().total_cost_usd, 0.0, 1e-12);
}

TEST(TokenBudgetManager, InitialUtilizationIsZero) {
    TokenBudgetManager mgr(make_budget());
    EXPECT_NEAR(mgr.utilization(), 0.0, 1e-12);
}

// ---------------------------------------------------------------------------
// check_input
// ---------------------------------------------------------------------------

TEST(TokenBudgetManager, CheckInputWithinBudget) {
    TokenBudgetManager mgr(make_budget());
    EXPECT_TRUE(mgr.check_input(100));
}

TEST(TokenBudgetManager, CheckInputExceedsInputCap) {
    TokenBudgetManager mgr(make_budget(/*max_in=*/100));
    EXPECT_FALSE(mgr.check_input(101));
}

TEST(TokenBudgetManager, CheckInputExceedsTotalCap) {
    // max_in=2000 but max_tot=500
    TokenBudgetManager mgr(make_budget(/*max_in=*/2000, /*max_out=*/2000, /*max_tot=*/500));
    EXPECT_FALSE(mgr.check_input(501));
}

TEST(TokenBudgetManager, CheckInputExactlyAtCap) {
    TokenBudgetManager mgr(make_budget(/*max_in=*/100));
    EXPECT_TRUE(mgr.check_input(100));
}

// ---------------------------------------------------------------------------
// consume_input
// ---------------------------------------------------------------------------

TEST(TokenBudgetManager, ConsumeInputUpdatesUsage) {
    TokenBudgetManager mgr(make_budget());
    mgr.consume_input(100);
    EXPECT_EQ(mgr.usage().input_tokens, 100u);
    EXPECT_EQ(mgr.usage().total_tokens(), 100u);
}

TEST(TokenBudgetManager, ConsumeInputAccumulatesCost) {
    TokenBudgetManager mgr(make_budget(1024, 512, 1536, 0.001));
    mgr.consume_input(10);
    EXPECT_NEAR(mgr.usage().total_cost_usd, 0.01, 1e-9);
}

TEST(TokenBudgetManager, ConsumeInputThrowsWhenExceeded) {
    TokenBudgetManager mgr(make_budget(/*max_in=*/50));
    EXPECT_THROW(mgr.consume_input(51), std::runtime_error);
}

TEST(TokenBudgetManager, ConsumeInputThrowsOnTotalExceeded) {
    TokenBudgetManager mgr(make_budget(2000, 2000, /*max_tot=*/100));
    mgr.consume_input(80);
    // Next call would push total to 130 > 100
    EXPECT_THROW(mgr.consume_input(30), std::runtime_error);
}

// ---------------------------------------------------------------------------
// consume_output
// ---------------------------------------------------------------------------

TEST(TokenBudgetManager, ConsumeOutputUpdatesUsage) {
    TokenBudgetManager mgr(make_budget());
    mgr.consume_output(64);
    EXPECT_EQ(mgr.usage().output_tokens, 64u);
}

TEST(TokenBudgetManager, ConsumeOutputThrowsWhenExceeded) {
    TokenBudgetManager mgr(make_budget(1024, /*max_out=*/100));
    EXPECT_THROW(mgr.consume_output(101), std::runtime_error);
}

TEST(TokenBudgetManager, MultipleConsumptions) {
    TokenBudgetManager mgr(make_budget());
    mgr.consume_input(100);
    mgr.consume_output(50);
    mgr.consume_input(200);
    EXPECT_EQ(mgr.usage().input_tokens,  300u);
    EXPECT_EQ(mgr.usage().output_tokens, 50u);
    EXPECT_EQ(mgr.usage().total_tokens(), 350u);
}

// ---------------------------------------------------------------------------
// remaining_output_budget
// ---------------------------------------------------------------------------

TEST(TokenBudgetManager, RemainingOutputBudgetInitial) {
    auto b = make_budget(1024, 512, 1536);
    TokenBudgetManager mgr(b);
    // min(512, 1536) = 512
    EXPECT_EQ(mgr.remaining_output_budget(), 512u);
}

TEST(TokenBudgetManager, RemainingOutputBudgetAfterInput) {
    auto b = make_budget(1024, 512, 600);
    TokenBudgetManager mgr(b);
    mgr.consume_input(200);
    // total used = 200, total remaining = 400
    // output cap remaining = 512, total cap remaining = 400
    // min = 400
    EXPECT_EQ(mgr.remaining_output_budget(), 400u);
}

TEST(TokenBudgetManager, RemainingInputBudgetAfterConsumption) {
    auto b = make_budget(1024, 512, 1536);
    TokenBudgetManager mgr(b);
    mgr.consume_input(100);
    EXPECT_EQ(mgr.remaining_input_budget(), 924u);
}

// ---------------------------------------------------------------------------
// utilization
// ---------------------------------------------------------------------------

TEST(TokenBudgetManager, UtilizationAfterConsumption) {
    auto b = make_budget(1024, 512, 1000);
    TokenBudgetManager mgr(b);
    mgr.consume_input(250);
    mgr.consume_output(250);
    EXPECT_NEAR(mgr.utilization(), 0.5, 1e-9);
}

TEST(TokenBudgetManager, UtilizationCappedAtOne) {
    auto b = make_budget(100, 100, 100);
    TokenBudgetManager mgr(b);
    mgr.consume_input(100);
    EXPECT_NEAR(mgr.utilization(), 1.0, 1e-9);
}

// ---------------------------------------------------------------------------
// reset
// ---------------------------------------------------------------------------

TEST(TokenBudgetManager, ResetClearsUsage) {
    TokenBudgetManager mgr(make_budget());
    mgr.consume_input(200);
    mgr.consume_output(100);
    mgr.reset();
    EXPECT_EQ(mgr.usage().input_tokens,  0u);
    EXPECT_EQ(mgr.usage().output_tokens, 0u);
    EXPECT_NEAR(mgr.usage().total_cost_usd, 0.0, 1e-12);
}

TEST(TokenBudgetManager, ResetPreservesBudgetLimits) {
    auto b = make_budget(999, 888, 1887);
    TokenBudgetManager mgr(b);
    mgr.consume_input(100);
    mgr.reset();
    EXPECT_EQ(mgr.budget().max_input_tokens,  999u);
    EXPECT_EQ(mgr.budget().max_output_tokens, 888u);
    EXPECT_EQ(mgr.budget().max_total_tokens,  1887u);
}

// ---------------------------------------------------------------------------
// format_summary
// ---------------------------------------------------------------------------

TEST(TokenBudgetManager, FormatSummaryContainsKeyFields) {
    TokenBudgetManager mgr(make_budget());
    mgr.consume_input(50);
    std::string s = mgr.format_summary();
    EXPECT_NE(s.find("input="), std::string::npos);
    EXPECT_NE(s.find("output="), std::string::npos);
    EXPECT_NE(s.find("util="), std::string::npos);
}
