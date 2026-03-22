#include <gtest/gtest.h>

#ifdef LLMQUANT_GRADIENT_OPTIMISER_ENABLED

#include "GradientOptimiser.h"

#include <cmath>
#include <string>
#include <vector>

using namespace llmquant;

namespace {

static std::vector<std::string> make_tokens(int n) {
    std::vector<std::string> toks;
    const char* words[] = {"bullish", "rally", "crash", "fear", "neutral",
                           "bearish", "growth", "sell", "buy", "panic"};
    for (int i = 0; i < n; ++i) toks.push_back(words[i % 10]);
    return toks;
}

// ============================================================
// Construction
// ============================================================

TEST(GradientOptimiser, ConstructsOk) {
    GradientOptimiser opt;
    (void)opt;
}

// ============================================================
// run — no axes
// ============================================================

TEST(GradientOptimiser, NoAxesReturnsEmptyResult) {
    GradientOptimiser opt;
    opt.set_tokens(make_tokens(20));
    auto res = opt.run();
    EXPECT_TRUE(res.steps.empty());
    EXPECT_EQ(res.total_backtests, 0);
}

// ============================================================
// run — single axis, confirm it runs
// ============================================================

TEST(GradientOptimiser, SingleAxisRunsWithoutCrash) {
    GradientOptimiser::Config cfg;
    cfg.max_iterations = 3;
    GradientOptimiser opt(cfg);
    opt.set_tokens(make_tokens(30));
    opt.add_axis({"spread_cost", 0.0005, 0.0001, 0.005});
    auto res = opt.run();
    EXPECT_GT(res.total_backtests, 0);
}

// ============================================================
// run — two axes, Sharpe objective
// ============================================================

TEST(GradientOptimiser, TwoAxesYieldsSteps) {
    GradientOptimiser::Config cfg;
    cfg.max_iterations = 2;
    cfg.objective = GradientOptimiser::Objective::TotalPnL;
    GradientOptimiser opt(cfg);
    opt.set_tokens(make_tokens(50));
    opt.add_axis({"spread_cost",  0.001,  0.0001, 0.01});
    opt.add_axis({"price_impact", 0.005,  0.001,  0.02});
    auto res = opt.run();
    EXPECT_GE(res.steps.size(), 1u);
    EXPECT_GE(res.total_backtests, 1);
}

// ============================================================
// axis_names populated
// ============================================================

TEST(GradientOptimiser, AxisNamesPopulated) {
    GradientOptimiser::Config cfg;
    cfg.max_iterations = 1;
    GradientOptimiser opt(cfg);
    opt.set_tokens(make_tokens(20));
    opt.add_axis({"spread_cost",  0.001, 0.0001, 0.01});
    opt.add_axis({"price_impact", 0.005, 0.001,  0.02});
    auto res = opt.run();
    ASSERT_EQ(res.axis_names.size(), 2u);
    EXPECT_EQ(res.axis_names[0], "spread_cost");
    EXPECT_EQ(res.axis_names[1], "price_impact");
}

// ============================================================
// bounds clamping — params stay in [lo, hi]
// ============================================================

TEST(GradientOptimiser, ParamsStayWithinBounds) {
    GradientOptimiser::Config cfg;
    cfg.max_iterations  = 5;
    cfg.learning_rate   = 10.0;  // Large LR forces clamping.
    GradientOptimiser opt(cfg);
    opt.set_tokens(make_tokens(40));
    opt.add_axis({"spread_cost", 0.001, 0.0001, 0.003});
    auto res = opt.run();
    for (const auto& step : res.steps) {
        ASSERT_EQ(step.params.size(), 1u);
        EXPECT_GE(step.params[0], 0.0001 - 1e-12);
        EXPECT_LE(step.params[0], 0.003 + 1e-12);
    }
}

// ============================================================
// clear_axes
// ============================================================

TEST(GradientOptimiser, ClearAxesResetsAxes) {
    GradientOptimiser opt;
    opt.set_tokens(make_tokens(20));
    opt.add_axis({"spread_cost", 0.001, 0.0001, 0.01});
    opt.clear_axes();
    auto res = opt.run();
    EXPECT_TRUE(res.steps.empty());
}

// ============================================================
// progress callback
// ============================================================

TEST(GradientOptimiser, ProgressCallbackInvoked) {
    GradientOptimiser::Config cfg;
    cfg.max_iterations = 2;
    int cb_count = 0;
    cfg.progress_cb = [&](int, int, double) { ++cb_count; };
    GradientOptimiser opt(cfg);
    opt.set_tokens(make_tokens(30));
    opt.add_axis({"spread_cost", 0.001, 0.0001, 0.01});
    (void)opt.run();
    EXPECT_GT(cb_count, 0);
}

// ============================================================
// best() returns non-null after successful run
// ============================================================

TEST(GradientOptimiser, BestReturnsNonNull) {
    GradientOptimiser::Config cfg;
    cfg.max_iterations = 2;
    GradientOptimiser opt(cfg);
    opt.set_tokens(make_tokens(30));
    opt.add_axis({"spread_cost", 0.001, 0.0001, 0.01});
    auto res = opt.run();
    if (!res.steps.empty()) {
        EXPECT_NE(res.best(), nullptr);
    }
}

// ============================================================
// OptimiseResult::best on empty steps returns nullptr
// ============================================================

TEST(GradientOptimiser, BestOnEmptyReturnsNullptr) {
    GradientOptimiser::OptimiseResult r;
    EXPECT_EQ(r.best(), nullptr);
}

// ============================================================
// to_table_string
// ============================================================

TEST(GradientOptimiser, ToTableStringContainsHeaders) {
    GradientOptimiser::Config cfg;
    cfg.max_iterations = 1;
    GradientOptimiser opt(cfg);
    opt.set_tokens(make_tokens(20));
    opt.add_axis({"spread_cost", 0.001, 0.0001, 0.01});
    auto res = opt.run();
    std::string tbl = res.to_table_string();
    EXPECT_NE(tbl.find("GradientOptimiser"), std::string::npos);
    EXPECT_NE(tbl.find("Iter"),             std::string::npos);
    EXPECT_NE(tbl.find("Objective"),         std::string::npos);
}

// ============================================================
// Objective variants compile and run
// ============================================================

TEST(GradientOptimiser, AllObjectivesRunWithoutCrash) {
    const GradientOptimiser::Objective objs[] = {
        GradientOptimiser::Objective::SharpeRatio,
        GradientOptimiser::Objective::TotalPnL,
        GradientOptimiser::Objective::WinRate,
        GradientOptimiser::Objective::PassRate,
        GradientOptimiser::Objective::MinDrawdown,
    };
    for (auto obj : objs) {
        GradientOptimiser::Config cfg;
        cfg.max_iterations = 1;
        cfg.objective = obj;
        GradientOptimiser opt(cfg);
        opt.set_tokens(make_tokens(20));
        opt.add_axis({"spread_cost", 0.001, 0.0001, 0.01});
        EXPECT_NO_FATAL_FAILURE({ (void)opt.run(); });
    }
}

} // namespace

#else  // LLMQUANT_GRADIENT_OPTIMISER_ENABLED not defined

TEST(GradientOptimiser, DisabledAtBuildTime) {
    SUCCEED();
}

#endif  // LLMQUANT_GRADIENT_OPTIMISER_ENABLED
