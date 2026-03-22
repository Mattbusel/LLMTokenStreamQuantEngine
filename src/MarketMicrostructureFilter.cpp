#include "MarketMicrostructureFilter.h"

#include <cmath>
#include <sstream>

namespace llmquant {

MarketMicrostructureFilter::MarketMicrostructureFilter(Config config)
    : config_(std::move(config))
    , half_spread_(config_.initial_half_spread) {}

void MarketMicrostructureFilter::record_fill(
        double arrival_price, double fill_price, bool is_buy) {
    total_fills_.fetch_add(1, std::memory_order_relaxed);

    // Signed slippage: positive for a buy fill above arrival, negative for sell.
    double slippage = is_buy ? (fill_price - arrival_price)
                             : (arrival_price - fill_price);
    double obs = std::max(0.0, slippage);  // half-spread proxy (non-negative)

    {
        std::lock_guard<std::mutex> lk(mutex_);
        double alpha   = 1.0 - config_.ewma_decay;
        half_spread_   = alpha * obs + config_.ewma_decay * half_spread_;
    }
}

void MarketMicrostructureFilter::reset() {
    std::lock_guard<std::mutex> lk(mutex_);
    half_spread_ = config_.initial_half_spread;
}

MarketMicrostructureFilter::Decision
MarketMicrostructureFilter::should_trade(double predicted_edge, double trade_value) const {
    double hs    = estimated_half_spread();
    double cost  = hs + config_.impact_fraction * std::abs(trade_value);
    bool   pass  = (predicted_edge >= cost * config_.min_edge_multiplier);

    if (pass) {
        total_passed_.fetch_add(1, std::memory_order_relaxed);
        return Decision::Pass;
    }

    total_blocked_.fetch_add(1, std::memory_order_relaxed);

    std::function<void(double, double)> cb;
    {
        std::lock_guard<std::mutex> lk(mutex_);
        cb = config_.on_blocked;
    }
    if (cb) cb(predicted_edge, cost);

    return Decision::Block;
}

double MarketMicrostructureFilter::estimated_half_spread() const noexcept {
    std::lock_guard<std::mutex> lk(mutex_);
    return half_spread_;
}

double MarketMicrostructureFilter::estimated_cost(double trade_value) const noexcept {
    double hs = estimated_half_spread();
    return hs + config_.impact_fraction * std::abs(trade_value);
}

void MarketMicrostructureFilter::update_config(const Config& config) {
    std::lock_guard<std::mutex> lk(mutex_);
    config_      = config;
    half_spread_ = config_.initial_half_spread;
}

std::string MarketMicrostructureFilter::to_stats_json() const {
    double hs = estimated_half_spread();
    std::ostringstream os;
    os << std::fixed;
    os << "{"
       << "\"estimated_half_spread\":" << hs << ","
       << "\"total_fills\":"           << total_fills_.load(std::memory_order_relaxed) << ","
       << "\"total_passed\":"          << total_passed_.load(std::memory_order_relaxed) << ","
       << "\"total_blocked\":"         << total_blocked_.load(std::memory_order_relaxed) << ","
       << "\"min_edge_multiplier\":"   << config_.min_edge_multiplier
       << "}";
    return os.str();
}

} // namespace llmquant
