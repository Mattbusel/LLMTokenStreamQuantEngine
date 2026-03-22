#include "PnLAttributionEngine.h"

#include <algorithm>
#include <cstdio>

namespace llmquant {

// ---------------------------------------------------------------------------

PnLAttributionEngine::PnLAttributionEngine(Config cfg) : cfg_(std::move(cfg)) {}

// ---------------------------------------------------------------------------

std::vector<std::string> PnLAttributionEngine::classify(const TradeSignal& s) const {
    std::vector<std::string> labels;

    // 1. Bias direction
    if (s.delta_bias_shift > cfg_.neutral_band)
        labels.emplace_back("bias:bullish");
    else if (s.delta_bias_shift < -cfg_.neutral_band)
        labels.emplace_back("bias:bearish");
    else
        labels.emplace_back("bias:neutral");

    // 2. Confidence band
    if (s.confidence >= cfg_.conf_high_threshold)
        labels.emplace_back("conf:high");
    else if (s.confidence >= cfg_.conf_medium_threshold)
        labels.emplace_back("conf:medium");
    else
        labels.emplace_back("conf:low");

    // 3. Volatility regime
    if (s.volatility_adjustment >= cfg_.vol_extreme_threshold)
        labels.emplace_back("vol:extreme");
    else if (s.volatility_adjustment >= cfg_.vol_elevated_threshold)
        labels.emplace_back("vol:elevated");
    else
        labels.emplace_back("vol:calm");

    // 4. Signal quality
    if (s.signal_quality >= cfg_.sq_good_threshold)
        labels.emplace_back("sq:good");
    else if (s.signal_quality >= cfg_.sq_fair_threshold)
        labels.emplace_back("sq:fair");
    else
        labels.emplace_back("sq:poor");

    return labels;
}

// ---------------------------------------------------------------------------

void PnLAttributionEngine::record_outcome(const TradeSignal& signal,
                                          double realized_pnl) {
    auto labels = classify(signal);

    // Build a composite key from all driver labels for the on_outcome callback.
    std::string composite;
    for (const auto& l : labels) {
        if (!composite.empty()) composite += '|';
        composite += l;
    }

    DriverStats composite_stats;
    composite_stats.label = composite;

    {
        std::lock_guard<std::mutex> lk(mtx_);
        total_pnl_ += realized_pnl;

        for (const auto& label : labels) {
            auto& ds = drivers_[label];
            ds.label = label;
            ds.trade_count++;
            if (realized_pnl > 0.0) ds.win_count++;
            ds.total_pnl += realized_pnl;
        }

        // Populate composite for callback (no allocation inside lock beyond
        // what's already done above).
        composite_stats.trade_count = 1;
        composite_stats.win_count   = (realized_pnl > 0.0) ? 1u : 0u;
        composite_stats.total_pnl   = realized_pnl;
    }

    total_trades_.fetch_add(1, std::memory_order_relaxed);

    if (cfg_.on_outcome) cfg_.on_outcome(composite_stats);
}

// ---------------------------------------------------------------------------

std::vector<PnLAttributionEngine::DriverStats> PnLAttributionEngine::all_drivers() const {
    std::lock_guard<std::mutex> lk(mtx_);
    std::vector<DriverStats> out;
    out.reserve(drivers_.size());
    for (const auto& [k, v] : drivers_) out.push_back(v);
    std::sort(out.begin(), out.end(),
              [](const DriverStats& a, const DriverStats& b) {
                  return a.total_pnl > b.total_pnl;
              });
    return out;
}

std::vector<PnLAttributionEngine::DriverStats>
PnLAttributionEngine::top_drivers(size_t n) const {
    auto all = all_drivers();
    if (all.size() > n) all.resize(n);
    return all;
}

std::vector<PnLAttributionEngine::DriverStats>
PnLAttributionEngine::bottom_drivers(size_t n) const {
    auto all = all_drivers();
    if (all.size() <= n) return all;
    return std::vector<DriverStats>(all.end() - static_cast<ptrdiff_t>(n), all.end());
}

double PnLAttributionEngine::total_pnl() const noexcept {
    std::lock_guard<std::mutex> lk(mtx_);
    return total_pnl_;
}

void PnLAttributionEngine::reset() {
    std::lock_guard<std::mutex> lk(mtx_);
    drivers_.clear();
    total_pnl_ = 0.0;
    total_trades_.store(0, std::memory_order_relaxed);
}

void PnLAttributionEngine::update_config(const Config& cfg) {
    std::lock_guard<std::mutex> lk(mtx_);
    cfg_ = cfg;
    drivers_.clear();
    total_pnl_ = 0.0;
    total_trades_.store(0, std::memory_order_relaxed);
}

// ---------------------------------------------------------------------------

std::string PnLAttributionEngine::to_stats_json() const {
    std::lock_guard<std::mutex> lk(mtx_);

    char buf[256];
    std::snprintf(buf, sizeof(buf),
        "{\"total_trades\":%llu,\"total_pnl\":%.6f,\"driver_count\":%zu}",
        static_cast<unsigned long long>(total_trades_.load(std::memory_order_relaxed)),
        total_pnl_,
        drivers_.size());
    return buf;
}

} // namespace llmquant
