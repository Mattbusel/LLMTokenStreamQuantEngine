#include "WalkForwardValidator.h"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <numeric>
#include <sstream>

namespace llmquant {

WalkForwardValidator::WalkForwardValidator(Config config)
    : config_(std::move(config)) {
    // Ensure step_size defaults to test_size when left at 50 and test_size differs.
    if (config_.step_size <= 0) config_.step_size = config_.test_size;
}

void WalkForwardValidator::load_tokens(std::vector<std::string> tokens) {
    tokens_ = std::move(tokens);
}

bool WalkForwardValidator::load_tokens_from_file(const std::string& filepath) {
    std::ifstream f(filepath);
    if (!f.is_open()) return false;
    tokens_.clear();
    std::string line;
    while (std::getline(f, line)) {
        if (!line.empty() && line[0] != '#')
            tokens_.push_back(line);
    }
    return true;
}

void WalkForwardValidator::add_axis(const std::string& name,
                                    std::vector<double> values) {
    axis_names_.push_back(name);
    axes_.push_back({name, std::move(values)});
}

void WalkForwardValidator::clear_axes() {
    axes_.clear();
    axis_names_.clear();
}

int WalkForwardValidator::num_folds() const noexcept {
    int window = config_.train_size + config_.test_size;
    if (static_cast<int>(tokens_.size()) < window) return 0;
    return 1 + (static_cast<int>(tokens_.size()) - window) / config_.step_size;
}

// ---------------------------------------------------------------------------
// Private helpers
// ---------------------------------------------------------------------------

BacktestRunner::Config WalkForwardValidator::optimise_on_slice(
        const std::vector<std::string>& train_tokens) const {
    if (axes_.empty()) return config_.base_config;

    ParameterSweep::Config ps_cfg;
    ps_cfg.base_config = config_.base_config;
    ps_cfg.objective   = config_.objective;

    ParameterSweep sweep(ps_cfg);
    sweep.set_tokens(train_tokens);
    for (const auto& ax : axes_)
        sweep.add_axis(ax.name, ax.values);

    auto result = sweep.run();
    if (result.grid_points.empty()) return config_.base_config;

    // Apply best params back into a config copy.
    ps_cfg.base_config = config_.base_config;
    BacktestRunner::Config best = config_.base_config;
    const auto* bp = result.best();
    if (bp) {
        // Re-apply axis values to reconstruct the best config.
        // ParameterSweep::apply_axis_values is private; we replicate the
        // mapping here with the same field names.
        for (size_t i = 0; i < result.axis_names.size() && i < bp->param_values.size(); ++i) {
            const std::string& name = result.axis_names[i];
            double v = bp->param_values[i];
            if      (name == "spread_cost")        best.sim.spread_cost           = v;
            else if (name == "price_impact")       best.sim.price_impact_per_bias = v;
            else if (name == "signal_threshold")   best.engine.bias_sensitivity   = v;
            else if (name == "signal_decay_rate")  best.engine.signal_decay_rate  = v;
            else if (name == "min_bias_threshold") best.engine.min_bias_threshold = v;
            else if (name == "min_confidence")     best.risk.min_confidence       = v;
            else if (name == "max_bias_magnitude") best.risk.max_bias_magnitude   = v;
        }
    }
    return best;
}

/* static */
WalkForwardValidator::WalkForwardResult WalkForwardValidator::aggregate(
        std::vector<FoldResult> folds) {
    WalkForwardResult r;
    r.folds = std::move(folds);
    if (r.folds.empty()) return r;

    double sum_oos_sharpe = 0.0;
    double sum_is_sharpe  = 0.0;
    double sum_oos_pnl    = 0.0;
    double sum_oos_wr     = 0.0;

    for (const auto& f : r.folds) {
        sum_oos_sharpe += f.test_result.sharpe_ratio;
        sum_is_sharpe  += f.train_result.sharpe_ratio;
        sum_oos_pnl    += f.test_result.total_pnl;
        sum_oos_wr     += f.test_result.win_rate;
        if (f.test_result.sharpe_ratio > 0.0) ++r.positive_oos_folds;
    }

    double n = static_cast<double>(r.folds.size());
    r.mean_oos_sharpe  = sum_oos_sharpe  / n;
    r.mean_is_sharpe   = sum_is_sharpe   / n;
    r.sharpe_degradation = r.mean_is_sharpe - r.mean_oos_sharpe;
    r.combined_oos_pnl = sum_oos_pnl;
    r.mean_oos_win_rate = sum_oos_wr / n;

    return r;
}

// ---------------------------------------------------------------------------
// Main execution
// ---------------------------------------------------------------------------

WalkForwardValidator::WalkForwardResult WalkForwardValidator::run() {
    int total = num_folds();
    std::vector<FoldResult> folds;
    folds.reserve(static_cast<size_t>(total));

    for (int fold = 0; fold < total; ++fold) {
        int train_begin = fold * config_.step_size;
        int train_end   = train_begin + config_.train_size;
        int test_begin  = train_end;
        int test_end    = test_begin + config_.test_size;

        auto train_slice = std::vector<std::string>(
            tokens_.begin() + train_begin, tokens_.begin() + train_end);
        auto test_slice = std::vector<std::string>(
            tokens_.begin() + test_begin, tokens_.begin() + test_end);

        // Determine config: either optimised on train or fixed base.
        BacktestRunner::Config fold_cfg = config_.optimize
            ? optimise_on_slice(train_slice)
            : config_.base_config;

        // Evaluate IS (training window) with the chosen config.
        BacktestRunner train_runner(fold_cfg);
        train_runner.load_tokens(train_slice);
        auto train_result = train_runner.run();

        // Evaluate OOS (test window) with the same config — no further fitting.
        BacktestRunner test_runner(fold_cfg);
        test_runner.load_tokens(test_slice);
        auto test_result = test_runner.run();

        FoldResult fr;
        fr.fold_id      = fold;
        fr.train_begin  = train_begin;
        fr.train_end    = train_end;
        fr.test_begin   = test_begin;
        fr.test_end     = test_end;
        fr.train_result = std::move(train_result);
        fr.test_result  = std::move(test_result);
        folds.push_back(std::move(fr));

        if (config_.on_fold_complete)
            config_.on_fold_complete(fold + 1, total);
    }

    return aggregate(std::move(folds));
}

// ---------------------------------------------------------------------------
// to_summary_string
// ---------------------------------------------------------------------------

std::string WalkForwardValidator::WalkForwardResult::to_summary_string() const {
    std::ostringstream o;
    o << "=== Walk-Forward Validation Summary ===\n";
    o << "  Folds: " << folds.size() << "\n";
    o << "  Mean IS  Sharpe : " << mean_is_sharpe  << "\n";
    o << "  Mean OOS Sharpe : " << mean_oos_sharpe << "\n";
    o << "  Sharpe degradation (IS−OOS): " << sharpe_degradation << "\n";
    o << "  Combined OOS PnL : " << combined_oos_pnl << "\n";
    o << "  Mean OOS win rate: " << mean_oos_win_rate << "\n";
    o << "  Positive OOS folds: " << positive_oos_folds
      << " / " << folds.size() << "\n";
    o << "\n  Fold detail:\n";
    o << "  " << std::string(70, '-') << "\n";
    for (const auto& f : folds) {
        o << "  Fold " << f.fold_id
          << "  train[" << f.train_begin << "," << f.train_end << ")"
          << "  test[" << f.test_begin  << "," << f.test_end  << ")"
          << "  IS_sharpe=" << f.train_result.sharpe_ratio
          << "  OOS_sharpe=" << f.test_result.sharpe_ratio
          << "  OOS_pnl=" << f.test_result.total_pnl << "\n";
    }
    return o.str();
}

} // namespace llmquant
