#include "ParameterSweep.h"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <spdlog/spdlog.h>
#include <sstream>

namespace llmquant {

// ---------------------------------------------------------------------------
// Construction
// ---------------------------------------------------------------------------

ParameterSweep::ParameterSweep(Config config)
    : config_(std::move(config)) {}

// ---------------------------------------------------------------------------
// Input
// ---------------------------------------------------------------------------

void ParameterSweep::set_tokens(std::vector<std::string> tokens) {
    tokens_ = std::move(tokens);
}

bool ParameterSweep::load_tokens_from_file(const std::string& filepath) {
    std::ifstream f(filepath);
    if (!f.is_open()) return false;
    tokens_.clear();
    std::string line;
    while (std::getline(f, line)) {
        while (!line.empty() && (line.back() == '\r' || line.back() == ' '))
            line.pop_back();
        if (line.empty() || line[0] == '#') continue;
        tokens_.push_back(line);
    }
    return true;
}

void ParameterSweep::add_axis(const std::string& name, std::vector<double> values) {
    if (values.empty()) return;
    Axis ax;
    ax.name   = name;
    ax.values = std::move(values);
    axes_.push_back(std::move(ax));
}

void ParameterSweep::clear_axes() {
    axes_.clear();
}

int ParameterSweep::total_grid_points() const {
    if (axes_.empty()) return 1;
    int n = 1;
    for (auto& ax : axes_) n *= static_cast<int>(ax.values.size());
    return n;
}

// ---------------------------------------------------------------------------
// Axis value application
// ---------------------------------------------------------------------------

void ParameterSweep::apply_axis_values(BacktestRunner::Config& cfg,
                                        const std::vector<std::string>& axis_names,
                                        const std::vector<double>& values) const {
    for (size_t i = 0; i < axis_names.size(); ++i) {
        const std::string& name = axis_names[i];
        double v = values[i];

        if      (name == "spread_cost")         cfg.sim.spread_cost              = v;
        else if (name == "price_impact")        cfg.sim.price_impact_per_bias    = v;
        else if (name == "signal_threshold")    cfg.engine.bias_sensitivity      = v;
        else if (name == "signal_decay_rate")   cfg.engine.signal_decay_rate     = v;
        else if (name == "min_bias_threshold")  cfg.engine.min_bias_threshold    = v;
        else if (name == "min_confidence")      cfg.risk.min_confidence          = v;
        else if (name == "max_bias_magnitude")  cfg.risk.max_bias_magnitude      = v;
        else {
            spdlog::warn("[param_sweep] unknown axis name '{}' — skipped", name);
        }
    }
}

// ---------------------------------------------------------------------------
// Objective extraction
// ---------------------------------------------------------------------------

double ParameterSweep::extract_objective(const BacktestRunner::BacktestResult& r,
                                          Objective obj) noexcept {
    switch (obj) {
        case Objective::SharpeRatio: return r.sharpe_ratio;
        case Objective::TotalPnL:    return r.total_pnl;
        case Objective::WinRate:     return r.win_rate;
        case Objective::PassRate:    return r.pass_rate;
        case Objective::MinDrawdown: return -r.max_drawdown;  // negate: lower is better
    }
    return 0.0;
}

// ---------------------------------------------------------------------------
// Run
// ---------------------------------------------------------------------------

ParameterSweep::SweepResult ParameterSweep::run() {
    SweepResult out;
    out.objective = config_.objective;

    if (axes_.empty()) {
        // Single point: base config as-is.
        out.axis_names = {};
        BacktestRunner runner(config_.base_config);
        runner.load_tokens(tokens_);
        GridPoint gp;
        gp.result          = runner.run();
        gp.objective_value = extract_objective(gp.result, config_.objective);
        out.grid_points.push_back(std::move(gp));
        return out;
    }

    // Collect axis names.
    for (auto& ax : axes_) out.axis_names.push_back(ax.name);

    // Enumerate grid points via mixed-radix counter.
    int total = total_grid_points();
    int limit = (config_.max_points > 0) ? std::min(config_.max_points, total) : total;

    out.grid_points.reserve(static_cast<size_t>(limit));

    // indices[i] ∈ [0, axes_[i].values.size())
    std::vector<int> indices(axes_.size(), 0);

    for (int pt = 0; pt < limit; ++pt) {
        // Build axis values for this grid point.
        std::vector<double> vals;
        vals.reserve(axes_.size());
        for (size_t i = 0; i < axes_.size(); ++i)
            vals.push_back(axes_[i].values[static_cast<size_t>(indices[i])]);

        // Apply to a copy of the base config.
        BacktestRunner::Config cfg = config_.base_config;
        apply_axis_values(cfg, out.axis_names, vals);

        // Run backtest.
        BacktestRunner runner(cfg);
        runner.load_tokens(tokens_);

        GridPoint gp;
        gp.param_values    = vals;
        gp.result          = runner.run();
        gp.objective_value = extract_objective(gp.result, config_.objective);
        out.grid_points.push_back(std::move(gp));

        if (config_.progress_cb)
            config_.progress_cb(pt + 1, limit);

        // Increment mixed-radix counter (rightmost axis increments first).
        for (int ax = static_cast<int>(axes_.size()) - 1; ax >= 0; --ax) {
            if (++indices[ax] < static_cast<int>(axes_[ax].values.size())) break;
            indices[ax] = 0;
        }
    }

    // Sort best-first by objective.
    std::sort(out.grid_points.begin(), out.grid_points.end(),
              [](const GridPoint& a, const GridPoint& b) {
                  return a.objective_value > b.objective_value;
              });

    spdlog::info("[param_sweep] completed {} / {} grid points; best objective={:.4f}",
                 out.grid_points.size(), total,
                 out.grid_points.empty() ? 0.0 : out.grid_points.front().objective_value);
    return out;
}

// ---------------------------------------------------------------------------
// Result formatting
// ---------------------------------------------------------------------------

std::string ParameterSweep::SweepResult::to_table_string(int top_n) const {
    const int n = (top_n <= 0 || top_n > static_cast<int>(grid_points.size()))
                ? static_cast<int>(grid_points.size())
                : top_n;
    if (n == 0) return "(no results)\n";

    std::ostringstream s;
    s << "=== ParameterSweep Results (top " << n << ") ===\n";

    // Header.
    s << std::left << std::setw(5) << "Rank";
    for (auto& name : axis_names)
        s << std::setw(16) << name;
    s << std::setw(12) << "Objective"
      << std::setw(12) << "TotalPnL"
      << std::setw(10) << "Sharpe"
      << std::setw(10) << "WinRate"
      << std::setw(10) << "PassRate"
      << std::setw(12) << "MaxDrwDwn"
      << "\n";

    // Separator.
    int width = 5 + static_cast<int>(axis_names.size()) * 16 + 12 + 12 + 10 + 10 + 10 + 12;
    s << std::string(static_cast<size_t>(width), '-') << "\n";

    for (int i = 0; i < n; ++i) {
        const GridPoint& gp = grid_points[static_cast<size_t>(i)];
        const auto& r = gp.result;

        s << std::left << std::setw(5) << (i + 1);
        for (double v : gp.param_values)
            s << std::setw(16) << std::fixed << std::setprecision(5) << v;

        s << std::setw(12) << std::fixed << std::setprecision(4) << gp.objective_value
          << std::setw(12) << std::fixed << std::setprecision(6) << r.total_pnl
          << std::setw(10) << std::fixed << std::setprecision(4) << r.sharpe_ratio
          << std::setw(10) << std::fixed << std::setprecision(1) << (r.win_rate * 100.0)
          << std::setw(10) << std::fixed << std::setprecision(1) << (r.pass_rate * 100.0)
          << std::setw(12) << std::fixed << std::setprecision(6) << r.max_drawdown
          << "\n";
    }
    s << "==========================================\n";
    return s.str();
}

} // namespace llmquant
