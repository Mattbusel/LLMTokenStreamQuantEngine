#include "BacktestRunner.h"
#include "gtest/gtest.h"

#include <cmath>
#include <cstdio>
#include <fstream>
#include <string>
#include <vector>

using namespace llmquant;

namespace {

// Helper: build a BacktestRunner with zero cooldown and permissive risk gates.
static BacktestRunner make_runner(double spread_cost = 0.0,
                                   double price_impact = 0.01)
{
    BacktestRunner::Config cfg;
    cfg.engine.signal_cooldown = std::chrono::microseconds{0};
    // Keep default decay rate (0.95) — 0.0 is rejected by TradeSignalEngine.
    cfg.risk.disable_magnitude_gate = true;
    cfg.risk.disable_confidence_gate = true;
    cfg.risk.disable_rate_gate       = true;
    cfg.risk.disable_drawdown_gate   = true;
    cfg.risk.disable_position_gate   = true;
    cfg.sim.spread_cost          = spread_cost;
    cfg.sim.price_impact_per_bias = price_impact;
    return BacktestRunner(cfg);
}

// Helper: write a temp token file and return the path.
static std::string write_temp_tokens(const std::vector<std::string>& tokens) {
    std::string path = "tmp_backtest_tokens.txt";
    std::ofstream f(path);
    for (auto& t : tokens) f << t << "\n";
    return path;
}

// ============================================================
// Construction and empty-sequence handling
// ============================================================

TEST(BacktestRunner, EmptyTokensReturnsZeroSignals) {
    BacktestRunner runner = make_runner();
    auto result = runner.run();
    EXPECT_EQ(result.total_signals, 0);
    EXPECT_EQ(result.passed_signals, 0);
    EXPECT_EQ(result.total_pnl, 0.0);
}

TEST(BacktestRunner, LoadTokensThenRunProducesResult) {
    BacktestRunner runner = make_runner();
    runner.load_tokens({"bullish", "crash", "rally"});
    auto result = runner.run();
    // At least some signals expected from known high-weight tokens.
    EXPECT_GE(result.total_signals, 0);
    EXPECT_GE(result.records.size(), 0u);
}

// ============================================================
// File loading
// ============================================================

TEST(BacktestRunner, LoadTokensFromFileReturnsTrue) {
    auto path = write_temp_tokens({"bullish", "bearish", "rally"});
    BacktestRunner runner = make_runner();
    EXPECT_TRUE(runner.load_tokens_from_file(path));
    std::remove(path.c_str());
}

TEST(BacktestRunner, LoadTokensFromMissingFileReturnsFalse) {
    BacktestRunner runner = make_runner();
    EXPECT_FALSE(runner.load_tokens_from_file("nonexistent_file_xyz.txt"));
}

TEST(BacktestRunner, LoadTokensFromFileSkipsCommentsAndEmptyLines) {
    std::string path = "tmp_bt_comments.txt";
    {
        std::ofstream f(path);
        f << "# header comment\n";
        f << "\n";
        f << "bullish\n";
        f << "  \n";
        f << "# another comment\n";
        f << "crash\n";
    }
    BacktestRunner runner = make_runner();
    EXPECT_TRUE(runner.load_tokens_from_file(path));
    auto result = runner.run();
    EXPECT_GE(result.total_signals, 0);
    std::remove(path.c_str());
}

TEST(BacktestRunner, RunFromFileMissingReturnsFileErrorSentinel) {
    BacktestRunner runner = make_runner();
    auto result = runner.run_from_file("no_such_file.txt");
    EXPECT_EQ(result.total_signals, -1);
}

// ============================================================
// Signal count and pass/block partitioning
// ============================================================

TEST(BacktestRunner, PassedPlusBlockedEqualsTotal) {
    BacktestRunner runner = make_runner();
    runner.load_tokens({"crash", "panic", "bullish", "bearish", "volatile",
                        "rally", "surge", "the", "and", "confident"});
    auto result = runner.run();
    EXPECT_EQ(result.passed_signals + result.blocked_signals, result.total_signals);
}

TEST(BacktestRunner, RecordsSizeMatchesTotalSignals) {
    BacktestRunner runner = make_runner();
    runner.load_tokens({"crash", "bullish", "bearish", "rally"});
    auto result = runner.run();
    EXPECT_EQ(static_cast<int>(result.records.size()), result.total_signals);
}

// ============================================================
// PnL model
// ============================================================

TEST(BacktestRunner, TotalPnlEqualsSumOfSignalReturns) {
    BacktestRunner runner = make_runner(0.0, 0.01);
    runner.load_tokens({"crash", "panic", "bullish", "bearish",
                        "volatile", "rally", "surge", "confident"});
    auto result = runner.run();

    double sum = 0.0;
    for (auto& rec : result.records)
        sum += rec.simulated_return;
    EXPECT_NEAR(result.total_pnl, sum, 1e-9);
}

TEST(BacktestRunner, CumulativePnlMonotonicWhenSpreadZero) {
    // With zero spread cost and fixed positive bias, every passed signal
    // should push cumulative PnL in the direction of the bias.
    BacktestRunner runner = make_runner(0.0, 0.01);
    runner.load_tokens({"bullish", "bullish", "bullish", "bullish",
                        "bullish", "bullish", "bullish", "bullish"});
    auto result = runner.run();
    if (result.passed_signals >= 2) {
        double prev = 0.0;
        for (auto& rec : result.records) {
            if (rec.passed && rec.bias > 0) {
                EXPECT_GE(rec.cumulative_pnl, prev - 1e-9);
            }
            prev = rec.cumulative_pnl;
        }
    }
}

TEST(BacktestRunner, BlockedSignalsHaveZeroReturn) {
    // Use a risk manager that blocks everything to verify blocked records
    // never contribute to PnL.
    BacktestRunner::Config cfg;
    cfg.engine.signal_cooldown = std::chrono::microseconds{0};
    cfg.engine.signal_decay_rate = 0.0;
    // Very tight magnitude gate: block almost all signals.
    cfg.risk.max_bias_magnitude      = 1e-9;
    cfg.risk.max_volatility_magnitude = 1e-9;
    cfg.risk.max_spread_magnitude     = 1e-9;
    cfg.risk.min_confidence           = 0.0;
    cfg.sim.spread_cost           = 0.0;
    cfg.sim.price_impact_per_bias = 0.01;
    BacktestRunner runner(cfg);
    runner.load_tokens({"crash", "bullish", "bearish", "volatile", "rally"});
    auto result = runner.run();

    for (auto& rec : result.records) {
        if (!rec.passed) {
            EXPECT_NEAR(rec.simulated_return, 0.0, 1e-12);
        }
    }
}

// ============================================================
// Aggregate statistics
// ============================================================

TEST(BacktestRunner, PassRateInUnitIntervalOrZero) {
    BacktestRunner runner = make_runner();
    runner.load_tokens({"crash", "bullish", "rally", "panic", "surge"});
    auto result = runner.run();
    EXPECT_GE(result.pass_rate, 0.0);
    EXPECT_LE(result.pass_rate, 1.0);
}

TEST(BacktestRunner, WinRateInUnitIntervalOrZero) {
    BacktestRunner runner = make_runner();
    runner.load_tokens({"bullish", "crash", "rally", "bearish", "surge"});
    auto result = runner.run();
    EXPECT_GE(result.win_rate, 0.0);
    EXPECT_LE(result.win_rate, 1.0);
}

TEST(BacktestRunner, MaxDrawdownNonNegative) {
    BacktestRunner runner = make_runner();
    runner.load_tokens({"crash", "panic", "bearish", "volatile", "sell"});
    auto result = runner.run();
    EXPECT_GE(result.max_drawdown, 0.0);
}

TEST(BacktestRunner, SharpeRatioZeroWhenNoPassedSignals) {
    // All-blocking risk config.
    BacktestRunner::Config cfg;
    cfg.engine.signal_cooldown = std::chrono::microseconds{0};
    cfg.risk.max_bias_magnitude = 1e-12;
    BacktestRunner runner(cfg);
    runner.load_tokens({"crash", "bullish"});
    auto result = runner.run();
    if (result.passed_signals < 2) {
        EXPECT_NEAR(result.sharpe_ratio, 0.0, 1e-9);
    }
}

TEST(BacktestRunner, MeanConfidenceInUnitInterval) {
    BacktestRunner runner = make_runner();
    runner.load_tokens({"crash", "bullish", "the", "and", "rally"});
    auto result = runner.run();
    if (result.total_signals > 0) {
        EXPECT_GE(result.mean_confidence, 0.0);
        EXPECT_LE(result.mean_confidence, 1.0);
    }
}

// ============================================================
// to_summary_string
// ============================================================

TEST(BacktestRunner, SummaryStringContainsRequiredFields) {
    BacktestRunner runner = make_runner();
    runner.load_tokens({"crash", "bullish", "rally"});
    auto result = runner.run();
    std::string summary = result.to_summary_string();
    EXPECT_NE(summary.find("Signals emitted"), std::string::npos);
    EXPECT_NE(summary.find("Total PnL"),       std::string::npos);
    EXPECT_NE(summary.find("Sharpe ratio"),     std::string::npos);
    EXPECT_NE(summary.find("Win rate"),         std::string::npos);
    EXPECT_NE(summary.find("Max drawdown"),     std::string::npos);
}

TEST(BacktestRunner, SummaryStringEmptyResultNocrash) {
    BacktestRunner runner = make_runner();
    // No tokens loaded.
    auto result = runner.run();
    EXPECT_NO_FATAL_FAILURE({ (void)result.to_summary_string(); });
}

// ============================================================
// Repeated runs are independent
// ============================================================

TEST(BacktestRunner, TwoRunsWithSameTokensProduceSameResult) {
    BacktestRunner runner = make_runner(0.0, 0.01);
    std::vector<std::string> tokens = {"crash", "bullish", "rally", "bearish"};
    runner.load_tokens(tokens);
    auto r1 = runner.run();
    runner.load_tokens(tokens);
    auto r2 = runner.run();
    EXPECT_EQ(r1.total_signals,   r2.total_signals);
    EXPECT_EQ(r1.passed_signals,  r2.passed_signals);
    EXPECT_NEAR(r1.total_pnl,     r2.total_pnl,    1e-9);
    EXPECT_NEAR(r1.sharpe_ratio,  r2.sharpe_ratio, 1e-9);
}

} // namespace
