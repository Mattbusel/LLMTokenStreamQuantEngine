#include "BacktestRunner.h"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <numeric>
#include <spdlog/spdlog.h>
#include <sstream>

namespace llmquant {

// ---------------------------------------------------------------------------
// Construction
// ---------------------------------------------------------------------------

BacktestRunner::BacktestRunner(Config config)
    : config_(std::move(config)) {}

// ---------------------------------------------------------------------------
// Input loading
// ---------------------------------------------------------------------------

void BacktestRunner::load_tokens(std::vector<std::string> tokens) {
    tokens_ = std::move(tokens);
}

bool BacktestRunner::load_tokens_from_file(const std::string& filepath) {
    std::ifstream f(filepath);
    if (!f.is_open()) {
        spdlog::warn("[backtest] cannot open token file '{}'", filepath);
        return false;
    }
    tokens_.clear();
    std::string line;
    while (std::getline(f, line)) {
        // Strip trailing whitespace.
        while (!line.empty() && (line.back() == '\r' || line.back() == ' '))
            line.pop_back();
        if (line.empty() || line[0] == '#') continue;
        tokens_.push_back(line);
    }
    spdlog::info("[backtest] loaded {} tokens from '{}'", tokens_.size(), filepath);
    return true;
}

// ---------------------------------------------------------------------------
// Execution
// ---------------------------------------------------------------------------

BacktestRunner::BacktestResult BacktestRunner::run() {
    if (tokens_.empty()) {
        spdlog::warn("[backtest] run() called with empty token list");
        return BacktestResult{};
    }

    // Build fresh pipeline instances for this run.
    LLMAdapter adapter;

    TradeSignalEngine::Config eng_cfg = config_.engine;
    eng_cfg.signal_cooldown = std::chrono::microseconds{0};  // no cooldown in backtest
    TradeSignalEngine engine(eng_cfg);
    engine.set_backtest_mode(true);

    RiskManager::Config risk_cfg = config_.risk;
    RiskManager risk(risk_cfg);

    // Collect signals via callback into a local vector.
    std::vector<SignalRecord> records;
    records.reserve(tokens_.size() / 4 + 16);

    engine.set_signal_callback([&](const TradeSignal& signal) {
        SignalRecord rec;
        rec.timestamp_ns   = signal.timestamp_ns;
        rec.bias           = signal.delta_bias_shift;
        rec.vol            = signal.volatility_adjustment;
        rec.confidence     = signal.confidence;
        rec.spread         = signal.spread_modifier;
        rec.signal_quality = signal.signal_quality;

        std::string reason;
        rec.passed = risk.evaluate_with_reason(signal, reason);
        rec.block_reason = rec.passed ? "" : reason;

        if (rec.passed) {
            // Gross return: directional bet on bias.
            double gross = rec.bias * config_.sim.price_impact_per_bias;
            // Transaction cost: proportional to position size (|bias|).
            double cost  = config_.sim.spread_cost * std::abs(rec.bias);
            rec.simulated_return = gross - cost;
        }

        records.push_back(std::move(rec));
    });

    // Drive each token through the pipeline.
    for (const auto& tok : tokens_) {
        auto weight = adapter.map_token_to_weight(tok);
        engine.process_semantic_weight(weight);
    }

    return compute_stats(std::move(records), config_.sim);
}

BacktestRunner::BacktestResult BacktestRunner::run_from_file(const std::string& filepath) {
    if (!load_tokens_from_file(filepath)) {
        BacktestResult r;
        r.total_signals = -1;  // sentinel: file error
        return r;
    }
    return run();
}

// ---------------------------------------------------------------------------
// Statistics computation
// ---------------------------------------------------------------------------

BacktestRunner::BacktestResult
BacktestRunner::compute_stats(std::vector<SignalRecord> records,
                               const SimParams& /*sim*/) {
    BacktestResult r;
    r.records = std::move(records);
    r.total_signals = static_cast<int>(r.records.size());

    if (r.total_signals == 0) return r;

    // Cumulative PnL pass, aggregate bias and confidence.
    double cumulative = 0.0;
    double peak       = 0.0;
    int    wins       = 0;
    double sum_abs_bias   = 0.0;
    double sum_confidence = 0.0;

    std::vector<double> pass_returns;
    pass_returns.reserve(static_cast<size_t>(r.total_signals));

    for (auto& rec : r.records) {
        sum_abs_bias   += std::abs(rec.bias);
        sum_confidence += rec.confidence;

        if (rec.passed) {
            ++r.passed_signals;
            cumulative += rec.simulated_return;
            pass_returns.push_back(rec.simulated_return);

            if (cumulative > peak) peak = cumulative;
            double drawdown = peak - cumulative;
            if (drawdown > r.max_drawdown) r.max_drawdown = drawdown;

            if (rec.simulated_return > 0.0) ++wins;
        } else {
            ++r.blocked_signals;
        }
        rec.cumulative_pnl = cumulative;
    }

    r.total_pnl      = cumulative;
    r.mean_abs_bias  = sum_abs_bias   / static_cast<double>(r.total_signals);
    r.mean_confidence = sum_confidence / static_cast<double>(r.total_signals);

    if (r.passed_signals > 0) {
        r.pass_rate = static_cast<double>(r.passed_signals) /
                      static_cast<double>(r.total_signals);
        r.win_rate  = static_cast<double>(wins) /
                      static_cast<double>(r.passed_signals);
    }

    // Sharpe ratio over passed-signal returns.
    if (static_cast<int>(pass_returns.size()) >= 2) {
        double n   = static_cast<double>(pass_returns.size());
        double sum = std::accumulate(pass_returns.begin(), pass_returns.end(), 0.0);
        double mean = sum / n;

        double sq_sum = 0.0;
        for (double x : pass_returns) {
            double d = x - mean;
            sq_sum += d * d;
        }
        double std_dev = std::sqrt(sq_sum / (n - 1.0));
        r.sharpe_ratio = (std_dev > 1e-12) ? mean / std_dev : 0.0;
    }

    return r;
}

// ---------------------------------------------------------------------------
// Summary formatting
// ---------------------------------------------------------------------------

std::string BacktestRunner::BacktestResult::to_summary_string() const {
    std::ostringstream s;
    s << "===== BacktestRunner Result =====\n";
    s << "  Signals emitted : " << total_signals   << "\n";
    s << "  Passed          : " << passed_signals  << "\n";
    s << "  Blocked         : " << blocked_signals << "\n";
    if (total_signals > 0) {
        char buf[64];
        std::snprintf(buf, sizeof(buf), "%.1f%%", pass_rate * 100.0);
        s << "  Pass rate       : " << buf << "\n";
        std::snprintf(buf, sizeof(buf), "%.6f", total_pnl);
        s << "  Total PnL       : " << buf << "\n";
        std::snprintf(buf, sizeof(buf), "%.6f", max_drawdown);
        s << "  Max drawdown    : " << buf << "\n";
        std::snprintf(buf, sizeof(buf), "%.4f", sharpe_ratio);
        s << "  Sharpe ratio    : " << buf << "\n";
        std::snprintf(buf, sizeof(buf), "%.1f%%", win_rate * 100.0);
        s << "  Win rate        : " << buf << "\n";
        std::snprintf(buf, sizeof(buf), "%.6f", mean_abs_bias);
        s << "  Mean |bias|     : " << buf << "\n";
        std::snprintf(buf, sizeof(buf), "%.4f", mean_confidence);
        s << "  Mean confidence : " << buf << "\n";
    }
    s << "=================================\n";
    return s.str();
}

} // namespace llmquant
