#include <gtest/gtest.h>

#ifdef LLMQUANT_PARAMETER_SWEEP_ENABLED

#include "ParameterSweep.h"

#include <algorithm>
#include <string>
#include <vector>

using namespace llmquant;

namespace {

// Build a minimal token list for quick sweep runs.
static std::vector<std::string> test_tokens() {
    return {"bullish", "rally", "positive", "gains", "crash", "bearish", "drop"};
}

// Build a minimal base config for sweeps.
static ParameterSweep::Config make_base_cfg() {
    ParameterSweep::Config cfg;
    cfg.base_config.sim.spread_cost           = 0.001;
    cfg.base_config.sim.price_impact_per_bias = 0.002;
    cfg.base_config.engine.bias_sensitivity   = 0.5;
    return cfg;
}

TEST(ParameterSweep, DefaultConstructsOk) {
    ParameterSweep sweep;
    EXPECT_EQ(sweep.total_grid_points(), 1);
}

TEST(ParameterSweep, SetTokensAndRun) {
    ParameterSweep sweep(make_base_cfg());
    sweep.set_tokens(test_tokens());
    auto result = sweep.run();
    EXPECT_EQ(result.grid_points.size(), 1u);
}

TEST(ParameterSweep, SingleAxisGridPointCount) {
    ParameterSweep sweep(make_base_cfg());
    sweep.set_tokens(test_tokens());
    sweep.add_axis("spread_cost", {0.0001, 0.001, 0.01});
    EXPECT_EQ(sweep.total_grid_points(), 3);
    auto result = sweep.run();
    EXPECT_EQ(result.grid_points.size(), 3u);
}

TEST(ParameterSweep, TwoAxisCartesianProduct) {
    ParameterSweep sweep(make_base_cfg());
    sweep.set_tokens(test_tokens());
    sweep.add_axis("spread_cost",  {0.001, 0.005});
    sweep.add_axis("price_impact", {0.001, 0.005});
    EXPECT_EQ(sweep.total_grid_points(), 4);
    auto result = sweep.run();
    EXPECT_EQ(result.grid_points.size(), 4u);
}

TEST(ParameterSweep, ResultsSortedBestFirst) {
    ParameterSweep sweep(make_base_cfg());
    sweep.set_tokens(test_tokens());
    sweep.add_axis("spread_cost", {0.0001, 0.001, 0.01});
    auto result = sweep.run();
    ASSERT_GE(result.grid_points.size(), 2u);
    for (size_t i = 1; i < result.grid_points.size(); ++i) {
        EXPECT_GE(result.grid_points[i - 1].objective_value,
                  result.grid_points[i].objective_value);
    }
}

TEST(ParameterSweep, MaxPointsLimitsRun) {
    ParameterSweep::Config cfg = make_base_cfg();
    cfg.max_points = 2;
    ParameterSweep sweep(cfg);
    sweep.set_tokens(test_tokens());
    sweep.add_axis("spread_cost", {0.0001, 0.001, 0.01, 0.05});
    EXPECT_EQ(sweep.total_grid_points(), 4);
    auto result = sweep.run();
    EXPECT_EQ(result.grid_points.size(), 2u);
}

TEST(ParameterSweep, ClearAxesResetsToSingle) {
    ParameterSweep sweep(make_base_cfg());
    sweep.add_axis("spread_cost", {0.001, 0.01});
    EXPECT_EQ(sweep.total_grid_points(), 2);
    sweep.clear_axes();
    EXPECT_EQ(sweep.total_grid_points(), 1);
}

TEST(ParameterSweep, UnknownAxisNameIsSkipped) {
    ParameterSweep sweep(make_base_cfg());
    sweep.set_tokens(test_tokens());
    sweep.add_axis("unknown_axis_xyz", {0.5, 1.0});
    // Should not crash; unknown axis is logged and skipped.
    EXPECT_NO_FATAL_FAILURE(sweep.run());
}

TEST(ParameterSweep, AxisNamesInResult) {
    ParameterSweep sweep(make_base_cfg());
    sweep.set_tokens(test_tokens());
    sweep.add_axis("spread_cost",  {0.001, 0.005});
    sweep.add_axis("min_confidence", {0.3, 0.5});
    auto result = sweep.run();
    ASSERT_EQ(result.axis_names.size(), 2u);
    EXPECT_EQ(result.axis_names[0], "spread_cost");
    EXPECT_EQ(result.axis_names[1], "min_confidence");
}

TEST(ParameterSweep, ParamValuesMatchAxes) {
    ParameterSweep sweep(make_base_cfg());
    sweep.set_tokens(test_tokens());
    sweep.add_axis("spread_cost", {0.001, 0.01});
    auto result = sweep.run();
    ASSERT_EQ(result.grid_points.size(), 2u);
    // Each grid point should have one param value.
    for (const auto& gp : result.grid_points)
        EXPECT_EQ(gp.param_values.size(), 1u);
}

TEST(ParameterSweep, ToTableStringNonEmpty) {
    ParameterSweep sweep(make_base_cfg());
    sweep.set_tokens(test_tokens());
    sweep.add_axis("spread_cost", {0.001, 0.01});
    auto result = sweep.run();
    std::string table = result.to_table_string(5);
    EXPECT_FALSE(table.empty());
    EXPECT_NE(table.find("spread_cost"), std::string::npos);
}

TEST(ParameterSweep, ToTableStringEmptyResult) {
    ParameterSweep::SweepResult empty;
    std::string s = empty.to_table_string(10);
    EXPECT_NE(s.find("no results"), std::string::npos);
}

TEST(ParameterSweep, ProgressCallbackFires) {
    ParameterSweep::Config cfg = make_base_cfg();
    int last_done = 0, last_total = 0;
    cfg.progress_cb = [&](int done, int total) { last_done = done; last_total = total; };
    ParameterSweep sweep(cfg);
    sweep.set_tokens(test_tokens());
    sweep.add_axis("spread_cost", {0.001, 0.01, 0.05});
    sweep.run();
    EXPECT_EQ(last_done, 3);
    EXPECT_EQ(last_total, 3);
}

TEST(ParameterSweep, ObjectiveSharpeRatio) {
    ParameterSweep::Config cfg = make_base_cfg();
    cfg.objective = ParameterSweep::Objective::SharpeRatio;
    ParameterSweep sweep(cfg);
    sweep.set_tokens(test_tokens());
    sweep.add_axis("spread_cost", {0.001, 0.005});
    auto result = sweep.run();
    EXPECT_EQ(result.objective, ParameterSweep::Objective::SharpeRatio);
    EXPECT_EQ(result.grid_points.size(), 2u);
}

TEST(ParameterSweep, ObjectiveMinDrawdown) {
    ParameterSweep::Config cfg = make_base_cfg();
    cfg.objective = ParameterSweep::Objective::MinDrawdown;
    ParameterSweep sweep(cfg);
    sweep.set_tokens(test_tokens());
    sweep.add_axis("spread_cost", {0.001, 0.005});
    auto result = sweep.run();
    EXPECT_EQ(result.objective, ParameterSweep::Objective::MinDrawdown);
    EXPECT_EQ(result.grid_points.size(), 2u);
}

} // namespace

#else  // LLMQUANT_PARAMETER_SWEEP_ENABLED not defined

TEST(ParameterSweep, DisabledAtBuildTime) {
    SUCCEED();
}

#endif  // LLMQUANT_PARAMETER_SWEEP_ENABLED
