#include <gtest/gtest.h>

#ifdef LLMQUANT_WALK_FORWARD_ENABLED

#include "WalkForwardValidator.h"
#include <string>
#include <vector>

using namespace llmquant;

namespace {

static std::vector<std::string> make_tokens(int n) {
    std::vector<std::string> toks;
    toks.reserve(static_cast<size_t>(n));
    const char* words[] = {"bullish","rally","fear","crash","the","sentiment",
                           "bearish","momentum","breakout","reversal"};
    for (int i = 0; i < n; ++i)
        toks.push_back(words[i % 10]);
    return toks;
}

// ============================================================
// num_folds returns correct count
// ============================================================

TEST(WalkForwardValidator, NumFoldsCorrect) {
    WalkForwardValidator::Config cfg;
    cfg.train_size = 100;
    cfg.test_size  = 50;
    cfg.step_size  = 50;
    WalkForwardValidator wfv(cfg);
    wfv.load_tokens(make_tokens(300));
    // Total=300, window=150, step=50: folds = 1 + (300-150)/50 = 1+3 = 4
    EXPECT_EQ(wfv.num_folds(), 4);
}

TEST(WalkForwardValidator, NumFoldsZeroWhenInsufficientTokens) {
    WalkForwardValidator::Config cfg;
    cfg.train_size = 200;
    cfg.test_size  = 50;
    WalkForwardValidator wfv(cfg);
    wfv.load_tokens(make_tokens(100));  // only 100 tokens, need 250
    EXPECT_EQ(wfv.num_folds(), 0);
}

// ============================================================
// Run produces expected fold count
// ============================================================

TEST(WalkForwardValidator, RunProducesCorrectFoldCount) {
    WalkForwardValidator::Config cfg;
    cfg.train_size = 100;
    cfg.test_size  = 50;
    cfg.step_size  = 50;
    cfg.optimize   = false;  // skip sweep for speed
    WalkForwardValidator wfv(cfg);
    wfv.load_tokens(make_tokens(300));
    auto result = wfv.run();
    EXPECT_EQ(static_cast<int>(result.folds.size()), 4);
}

// ============================================================
// Fold token ranges are non-overlapping and contiguous
// ============================================================

TEST(WalkForwardValidator, FoldRangesConsistent) {
    WalkForwardValidator::Config cfg;
    cfg.train_size = 50;
    cfg.test_size  = 25;
    cfg.step_size  = 25;
    cfg.optimize   = false;
    WalkForwardValidator wfv(cfg);
    wfv.load_tokens(make_tokens(200));
    auto result = wfv.run();
    ASSERT_GT(result.folds.size(), 1u);
    for (const auto& f : result.folds) {
        EXPECT_EQ(f.train_end,   f.test_begin);
        EXPECT_EQ(f.test_end,    f.test_begin + cfg.test_size);
        EXPECT_EQ(f.train_end - f.train_begin, cfg.train_size);
    }
    // Adjacent folds advance by step_size
    EXPECT_EQ(result.folds[1].train_begin,
              result.folds[0].train_begin + cfg.step_size);
}

// ============================================================
// Progress callback fires for each fold
// ============================================================

TEST(WalkForwardValidator, ProgressCallbackFires) {
    WalkForwardValidator::Config cfg;
    cfg.train_size = 50;
    cfg.test_size  = 25;
    cfg.step_size  = 25;
    cfg.optimize   = false;
    int fired = 0;
    cfg.on_fold_complete = [&](int /*done*/, int /*total*/) { ++fired; };
    WalkForwardValidator wfv(cfg);
    wfv.load_tokens(make_tokens(200));
    auto result = wfv.run();
    EXPECT_EQ(fired, static_cast<int>(result.folds.size()));
}

// ============================================================
// Aggregate statistics are sane
// ============================================================

TEST(WalkForwardValidator, AggregateStatsFinite) {
    WalkForwardValidator::Config cfg;
    cfg.train_size = 80;
    cfg.test_size  = 40;
    cfg.step_size  = 40;
    cfg.optimize   = false;
    WalkForwardValidator wfv(cfg);
    wfv.load_tokens(make_tokens(400));
    auto result = wfv.run();
    EXPECT_TRUE(std::isfinite(result.mean_is_sharpe));
    EXPECT_TRUE(std::isfinite(result.mean_oos_sharpe));
    EXPECT_TRUE(std::isfinite(result.sharpe_degradation));
    EXPECT_TRUE(std::isfinite(result.combined_oos_pnl));
    EXPECT_GE(result.positive_oos_folds, 0);
    EXPECT_LE(result.positive_oos_folds,
              static_cast<int>(result.folds.size()));
}

// ============================================================
// sharpe_degradation = IS - OOS
// ============================================================

TEST(WalkForwardValidator, SharpeDegradiationIdentity) {
    WalkForwardValidator::Config cfg;
    cfg.train_size = 60;
    cfg.test_size  = 30;
    cfg.step_size  = 30;
    cfg.optimize   = false;
    WalkForwardValidator wfv(cfg);
    wfv.load_tokens(make_tokens(300));
    auto result = wfv.run();
    EXPECT_NEAR(result.sharpe_degradation,
                result.mean_is_sharpe - result.mean_oos_sharpe, 1e-9);
}

// ============================================================
// combined_oos_pnl equals sum of per-fold OOS PnL
// ============================================================

TEST(WalkForwardValidator, CombinedOosPnlEqualsSum) {
    WalkForwardValidator::Config cfg;
    cfg.train_size = 60;
    cfg.test_size  = 30;
    cfg.step_size  = 30;
    cfg.optimize   = false;
    WalkForwardValidator wfv(cfg);
    wfv.load_tokens(make_tokens(300));
    auto result = wfv.run();
    double sum = 0.0;
    for (const auto& f : result.folds)
        sum += f.test_result.total_pnl;
    EXPECT_NEAR(result.combined_oos_pnl, sum, 1e-9);
}

// ============================================================
// With optimize=true and axes, runs without crashing
// ============================================================

TEST(WalkForwardValidator, OptimizeRunsWithAxes) {
    WalkForwardValidator::Config cfg;
    cfg.train_size = 80;
    cfg.test_size  = 40;
    cfg.step_size  = 40;
    cfg.optimize   = true;
    WalkForwardValidator wfv(cfg);
    wfv.load_tokens(make_tokens(400));
    wfv.add_axis("spread_cost",  {0.0001, 0.001});
    wfv.add_axis("price_impact", {0.001, 0.005});
    EXPECT_EQ(wfv.num_folds(), 8);
    auto result = wfv.run();
    EXPECT_EQ(static_cast<int>(result.folds.size()), 8);
    // Each fold should record some signals
    for (const auto& f : result.folds)
        EXPECT_GE(f.test_result.total_signals, 0);
}

// ============================================================
// to_summary_string produces non-empty output
// ============================================================

TEST(WalkForwardValidator, ToSummaryStringNonEmpty) {
    WalkForwardValidator::Config cfg;
    cfg.train_size = 60;
    cfg.test_size  = 30;
    cfg.step_size  = 30;
    cfg.optimize   = false;
    WalkForwardValidator wfv(cfg);
    wfv.load_tokens(make_tokens(240));
    auto result = wfv.run();
    auto s = result.to_summary_string();
    EXPECT_FALSE(s.empty());
    EXPECT_NE(s.find("Walk-Forward"), std::string::npos);
    EXPECT_NE(s.find("OOS"), std::string::npos);
}

// ============================================================
// load_tokens_from_file returns false on missing file
// ============================================================

TEST(WalkForwardValidator, LoadFromMissingFileReturnsFalse) {
    WalkForwardValidator wfv;
    EXPECT_FALSE(wfv.load_tokens_from_file("/nonexistent/path/tokens.txt"));
}

// ============================================================
// Empty run (no tokens) produces no folds
// ============================================================

TEST(WalkForwardValidator, EmptyRunProducesNoFolds) {
    WalkForwardValidator wfv;
    auto result = wfv.run();
    EXPECT_TRUE(result.folds.empty());
    EXPECT_NEAR(result.combined_oos_pnl, 0.0, 1e-12);
}

} // namespace

#else

TEST(WalkForwardValidator, DisabledAtBuildTime) { SUCCEED(); }

#endif  // LLMQUANT_WALK_FORWARD_ENABLED
