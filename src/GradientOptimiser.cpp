#include "GradientOptimiser.h"

#include <algorithm>
#include <cmath>
#include <iomanip>
#include <spdlog/spdlog.h>
#include <sstream>

namespace llmquant {

// ---------------------------------------------------------------------------
// Construction
// ---------------------------------------------------------------------------

GradientOptimiser::GradientOptimiser(Config config)
    : config_(std::move(config)) {}

// ---------------------------------------------------------------------------
// Setup
// ---------------------------------------------------------------------------

void GradientOptimiser::set_tokens(std::vector<std::string> tokens) {
    tokens_ = std::move(tokens);
}

void GradientOptimiser::add_axis(const Axis& axis) {
    axes_.push_back(axis);
}

void GradientOptimiser::clear_axes() {
    axes_.clear();
}

// ---------------------------------------------------------------------------
// apply_params
// ---------------------------------------------------------------------------

void GradientOptimiser::apply_params(BacktestRunner::Config& cfg,
                                      const std::vector<double>& params) const {
    for (size_t i = 0; i < axes_.size(); ++i) {
        const std::string& name = axes_[i].name;
        double v = params[i];
        if      (name == "spread_cost")        cfg.sim.spread_cost            = v;
        else if (name == "price_impact")       cfg.sim.price_impact_per_bias  = v;
        else if (name == "signal_threshold")   cfg.engine.bias_sensitivity    = v;
        else if (name == "signal_decay_rate")  cfg.engine.signal_decay_rate   = v;
        else if (name == "min_bias_threshold") cfg.engine.min_bias_threshold  = v;
        else if (name == "min_confidence")     cfg.risk.min_confidence        = v;
        else if (name == "max_bias_magnitude") cfg.risk.max_bias_magnitude    = v;
        else {
            spdlog::warn("[grad_opt] unknown axis '{}' — skipped", name);
        }
    }
}

// ---------------------------------------------------------------------------
// extract_objective
// ---------------------------------------------------------------------------

double GradientOptimiser::extract_objective(
    const BacktestRunner::BacktestResult& r, Objective obj) noexcept {
    switch (obj) {
        case Objective::SharpeRatio: return r.sharpe_ratio;
        case Objective::TotalPnL:   return r.total_pnl;
        case Objective::WinRate:    return r.win_rate;
        case Objective::PassRate:   return r.pass_rate;
        case Objective::MinDrawdown: return -r.max_drawdown;
    }
    return 0.0;
}

// ---------------------------------------------------------------------------
// evaluate
// ---------------------------------------------------------------------------

double GradientOptimiser::evaluate(const std::vector<double>& params,
                                    BacktestRunner::BacktestResult* result_out) {
    BacktestRunner::Config cfg = config_.base_config;
    apply_params(cfg, params);
    BacktestRunner runner(cfg);
    runner.load_tokens(tokens_);
    auto res = runner.run();
    double obj = extract_objective(res, config_.objective);
    if (result_out) *result_out = std::move(res);
    return obj;
}

// ---------------------------------------------------------------------------
// OptimiseResult helpers
// ---------------------------------------------------------------------------

const GradientOptimiser::Step*
GradientOptimiser::OptimiseResult::best() const noexcept {
    if (steps.empty()) return nullptr;
    const Step* b = &steps[0];
    for (const auto& s : steps) {
        if (s.objective_value > b->objective_value) b = &s;
    }
    return b;
}

std::string GradientOptimiser::OptimiseResult::to_table_string() const {
    if (steps.empty()) return "(no steps)\n";
    std::ostringstream s;
    s << "=== GradientOptimiser Trace ===\n";
    s << std::left << std::setw(5) << "Iter";
    for (const auto& n : axis_names) s << std::setw(16) << n;
    s << std::setw(12) << "Objective"
      << std::setw(12) << "GradNorm"
      << std::setw(8)  << "BTs"
      << "\n";

    int width = 5 + static_cast<int>(axis_names.size()) * 16 + 32;
    s << std::string(static_cast<size_t>(width), '-') << "\n";

    for (const auto& step : steps) {
        s << std::left << std::setw(5) << step.iteration;
        for (double p : step.params)
            s << std::setw(16) << std::fixed << std::setprecision(6) << p;
        s << std::setw(12) << std::fixed << std::setprecision(4) << step.objective_value
          << std::setw(12) << std::scientific << std::setprecision(2) << step.grad_norm
          << std::setw(8)  << step.backtests_run
          << "\n";
    }

    const auto* b = best();
    if (b) {
        s << "=== Best: iter=" << b->iteration
          << "  obj=" << std::fixed << std::setprecision(4) << b->objective_value
          << "  total_backtests=" << total_backtests
          << "  converged=" << (converged ? "YES" : "NO")
          << " ===\n";
    }
    return s.str();
}

// ---------------------------------------------------------------------------
// run — gradient descent
// ---------------------------------------------------------------------------

GradientOptimiser::OptimiseResult GradientOptimiser::run() {
    OptimiseResult out;
    out.total_backtests = 0;
    out.converged       = false;

    if (axes_.empty()) {
        spdlog::warn("[grad_opt] no axes defined — nothing to optimise");
        return out;
    }

    for (const auto& ax : axes_) out.axis_names.push_back(ax.name);

    const int N = static_cast<int>(axes_.size());

    // Initialise parameter vector with axis initial values clamped to bounds.
    std::vector<double> params(static_cast<size_t>(N));
    for (int i = 0; i < N; ++i) {
        params[static_cast<size_t>(i)] = std::clamp(
            axes_[static_cast<size_t>(i)].initial,
            axes_[static_cast<size_t>(i)].lo,
            axes_[static_cast<size_t>(i)].hi);
    }

    // Evaluate starting point.
    BacktestRunner::BacktestResult cur_result;
    double cur_obj = evaluate(params, &cur_result);
    ++out.total_backtests;

    double step = config_.initial_step;

    for (int it = 0; it < config_.max_iterations; ++it) {
        // Compute numerical gradient via central differences.
        std::vector<double> grad(static_cast<size_t>(N), 0.0);
        int bt_this_iter = 0;

        for (int i = 0; i < N; ++i) {
            size_t idx = static_cast<size_t>(i);
            double h = step * (std::abs(params[idx]) + 1e-8);
            // Clamp +h and -h to bounds before evaluating.
            std::vector<double> p_plus  = params;
            std::vector<double> p_minus = params;
            p_plus [idx] = std::clamp(params[idx] + h,
                                       axes_[idx].lo, axes_[idx].hi);
            p_minus[idx] = std::clamp(params[idx] - h,
                                       axes_[idx].lo, axes_[idx].hi);
            double f_plus  = evaluate(p_plus,  nullptr);
            double f_minus = evaluate(p_minus, nullptr);
            out.total_backtests += 2;
            bt_this_iter        += 2;
            double denom = p_plus[idx] - p_minus[idx];
            grad[idx] = (std::abs(denom) < 1e-15) ? 0.0 :
                        (f_plus - f_minus) / denom;
        }

        // Euclidean norm of gradient.
        double grad_norm = 0.0;
        for (double g : grad) grad_norm += g * g;
        grad_norm = std::sqrt(grad_norm);

        // Record this step before attempting the move.
        {
            Step s;
            s.iteration      = it;
            s.params         = params;
            s.objective_value = cur_obj;
            s.grad_norm      = grad_norm;
            s.backtests_run  = bt_this_iter;
            s.result         = cur_result;
            out.steps.push_back(std::move(s));
        }

        if (config_.progress_cb)
            config_.progress_cb(it, out.total_backtests, cur_obj);

        // Convergence check.
        if (grad_norm < config_.tolerance) {
            out.converged = true;
            spdlog::info("[grad_opt] converged at iter={} grad_norm={:.2e} obj={:.4f}",
                         it, grad_norm, cur_obj);
            break;
        }

        // Gradient ascent step with Armijo backtracking.
        double lr = config_.learning_rate;
        bool improved = false;
        for (int bt = 0; bt <= config_.max_backtrack; ++bt) {
            std::vector<double> candidate(static_cast<size_t>(N));
            for (int i = 0; i < N; ++i) {
                size_t idx = static_cast<size_t>(i);
                candidate[idx] = std::clamp(params[idx] + lr * grad[idx],
                                             axes_[idx].lo, axes_[idx].hi);
            }
            BacktestRunner::BacktestResult cand_result;
            double cand_obj = evaluate(candidate, &cand_result);
            ++out.total_backtests;
            if (cand_obj > cur_obj) {
                params     = candidate;
                cur_obj    = cand_obj;
                cur_result = std::move(cand_result);
                improved   = true;
                break;
            }
            lr *= 0.5;
        }

        if (!improved) {
            spdlog::info("[grad_opt] no improvement at iter={} — stopping", it);
            break;
        }
    }

    spdlog::info("[grad_opt] done: {} iters  {} backtests  best_obj={:.4f}  converged={}",
                 out.steps.size(), out.total_backtests,
                 out.best() ? out.best()->objective_value : 0.0,
                 out.converged ? "YES" : "NO");
    return out;
}

} // namespace llmquant
