#include "OrderBookSimulator.h"

#include <cmath>
#include <spdlog/spdlog.h>
#include <sstream>

namespace llmquant {

// ---------------------------------------------------------------------------
// Construction
// ---------------------------------------------------------------------------

OrderBookSimulator::OrderBookSimulator(Config config)
    : config_(std::move(config)) {}

// ---------------------------------------------------------------------------
// update_bias
// ---------------------------------------------------------------------------

void OrderBookSimulator::update_bias(double delta_bias) {
    total_updates_.fetch_add(1, std::memory_order_relaxed);

    std::function<void(double)> cb;
    bool fire_cb = false;
    double hs = 0.0;

    {
        std::lock_guard<std::mutex> lk(mutex_);
        // EMA accumulation: decay old bias, add new delta.
        cumulative_bias_ = (1.0 - config_.bias_decay) * cumulative_bias_ + delta_bias;

        // Compute half-spread with zero-order impact for alert check.
        double mid = config_.ref_price * (1.0 + config_.bias_sensitivity * cumulative_bias_);
        (void)mid; // mid is implicit in spread logic
        hs = config_.base_spread; // no order size here

        if (config_.spread_alert_threshold > 0.0 && hs >= config_.spread_alert_threshold) {
            cb      = config_.on_wide_spread;
            fire_cb = true;
        }
    }

    if (fire_cb) {
        wide_spread_events_.fetch_add(1, std::memory_order_relaxed);
        spdlog::warn("[lob_sim] wide spread half_spread={:.4f}", hs);
        if (cb) cb(hs);
    }
}

// ---------------------------------------------------------------------------
// reset
// ---------------------------------------------------------------------------

void OrderBookSimulator::reset() {
    std::lock_guard<std::mutex> lk(mutex_);
    cumulative_bias_ = 0.0;
}

// ---------------------------------------------------------------------------
// Query
// ---------------------------------------------------------------------------

double OrderBookSimulator::mid_price() const noexcept {
    std::lock_guard<std::mutex> lk(mutex_);
    return config_.ref_price * (1.0 + config_.bias_sensitivity * cumulative_bias_);
}

double OrderBookSimulator::half_spread(double order_notional) const noexcept {
    std::lock_guard<std::mutex> lk(mutex_);
    return config_.base_spread + config_.impact_coeff * std::sqrt(std::abs(order_notional));
}

double OrderBookSimulator::estimated_fill_price(Side side, double order_notional) const noexcept {
    std::lock_guard<std::mutex> lk(mutex_);
    double mid = config_.ref_price * (1.0 + config_.bias_sensitivity * cumulative_bias_);
    double hs  = config_.base_spread + config_.impact_coeff * std::sqrt(std::abs(order_notional));
    return (side == Side::Buy) ? (mid + hs) : (mid - hs);
}

double OrderBookSimulator::cumulative_bias() const noexcept {
    std::lock_guard<std::mutex> lk(mutex_);
    return cumulative_bias_;
}

// ---------------------------------------------------------------------------
// update_config
// ---------------------------------------------------------------------------

void OrderBookSimulator::update_config(const Config& config) {
    std::lock_guard<std::mutex> lk(mutex_);
    config_          = config;
    cumulative_bias_ = 0.0;
}

// ---------------------------------------------------------------------------
// to_stats_json
// ---------------------------------------------------------------------------

std::string OrderBookSimulator::to_stats_json() const {
    double mid = 0.0, bias = 0.0, hs = 0.0;
    {
        std::lock_guard<std::mutex> lk(mutex_);
        bias = cumulative_bias_;
        mid  = config_.ref_price * (1.0 + config_.bias_sensitivity * bias);
        hs   = config_.base_spread;
    }
    std::ostringstream s;
    s << "{\"mid_price\":"         << mid
      << ",\"cumulative_bias\":"   << bias
      << ",\"base_half_spread\":"  << hs
      << ",\"total_updates\":"     << total_updates()
      << ",\"wide_spread_events\":" << wide_spread_events()
      << "}";
    return s.str();
}

} // namespace llmquant
