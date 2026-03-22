#include "OrderFlowImbalanceDetector.h"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <sstream>

namespace llmquant {

// ---------------------------------------------------------------------------
// Default keyword dictionary
// ---------------------------------------------------------------------------

void OrderFlowImbalanceDetector::build_default_dict() {
    // Strongly bullish / buy-side tokens
    dict_["buyers"]        =  1.0;
    dict_["buying"]        =  0.9;
    dict_["bought"]        =  0.8;
    dict_["accumulating"]  =  0.9;
    dict_["bid"]           =  0.6;
    dict_["bids"]          =  0.6;
    dict_["lifted"]        =  0.8;  // "ask lifted" → buyers aggressive
    dict_["uptick"]        =  0.7;
    dict_["rally"]         =  0.7;
    dict_["breakout"]      =  0.8;
    dict_["demand"]        =  0.7;
    dict_["inflows"]       =  0.8;
    dict_["calls"]         =  0.5;  // options call buying
    dict_["longs"]         =  0.7;
    dict_["squeeze"]       =  0.6;  // short squeeze = buy pressure
    dict_["chasing"]       =  0.5;

    // Strongly bearish / sell-side tokens
    dict_["sellers"]       = -1.0;
    dict_["selling"]       = -0.9;
    dict_["sold"]          = -0.8;
    dict_["distributing"]  = -0.9;
    dict_["ask"]           = -0.6;
    dict_["asks"]          = -0.6;
    dict_["hit"]           = -0.7;  // "bid hit" → sellers aggressive
    dict_["downtick"]      = -0.7;
    dict_["selloff"]       = -0.8;
    dict_["breakdown"]     = -0.8;
    dict_["supply"]        = -0.7;
    dict_["outflows"]      = -0.8;
    dict_["puts"]          = -0.5;  // options put buying
    dict_["shorts"]        = -0.7;
    dict_["dumping"]       = -0.9;
    dict_["liquidating"]   = -0.8;
    dict_["panic"]         = -0.7;
    dict_["capitulation"]  = -0.9;

    // Moderate directional tokens
    dict_["bullish"]       =  0.6;
    dict_["bearish"]       = -0.6;
    dict_["surge"]         =  0.6;
    dict_["drop"]          = -0.6;
    dict_["decline"]       = -0.5;
    dict_["recovery"]      =  0.5;
    dict_["momentum"]      =  0.4;  // ambiguous; slight bullish lean
    dict_["dip"]           = -0.4;
    dict_["bounce"]        =  0.5;
}

// ---------------------------------------------------------------------------
// Construction
// ---------------------------------------------------------------------------

OrderFlowImbalanceDetector::OrderFlowImbalanceDetector(Config config)
    : config_(std::move(config)) {
    build_default_dict();
}

// ---------------------------------------------------------------------------
// Pressure lookup
// ---------------------------------------------------------------------------

double OrderFlowImbalanceDetector::lookup_pressure(
        const std::string& token) const {
    // Lowercase the token for case-insensitive lookup.
    std::string lower;
    lower.reserve(token.size());
    for (char c : token) lower.push_back(static_cast<char>(std::tolower(c)));

    auto it = dict_.find(lower);
    if (it != dict_.end()) return it->second;

    // Partial / prefix match: check if any dictionary key is a prefix of the
    // lowercased token (handles inflections like "selling" matching "sell").
    for (const auto& [kw, pressure] : dict_) {
        if (lower.size() >= kw.size() &&
            lower.compare(0, kw.size(), kw) == 0) {
            return pressure;
        }
    }
    return 0.0;
}

// ---------------------------------------------------------------------------
// EMA update (call under lock)
// ---------------------------------------------------------------------------

void OrderFlowImbalanceDetector::update_ema_locked(double pressure) {
    double alpha = config_.ema_alpha;
    ema_imbalance_ += alpha * (pressure - ema_imbalance_);
    // Clamp to [-1, 1] to prevent unbounded drift.
    if (ema_imbalance_ >  1.0) ema_imbalance_ =  1.0;
    if (ema_imbalance_ < -1.0) ema_imbalance_ = -1.0;
}

// ---------------------------------------------------------------------------
// Token recording
// ---------------------------------------------------------------------------

void OrderFlowImbalanceDetector::record(const std::string& token) noexcept {
    double pressure = 0.0;
    {
        std::lock_guard<std::mutex> lk(mutex_);
        pressure = lookup_pressure(token);
    }
    record_pressure(pressure);
}

void OrderFlowImbalanceDetector::record_pressure(double pressure) noexcept {
    total_tokens_.fetch_add(1, std::memory_order_relaxed);

    double imb = 0.0;
    bool fire  = false;
    bool buy   = false;
    std::function<void(double, bool)> cb;

    {
        std::lock_guard<std::mutex> lk(mutex_);
        update_ema_locked(pressure);
        imb = ema_imbalance_;
        imbalance_snapshot_.store(imb, std::memory_order_relaxed);

        uint64_t n = total_tokens_.load(std::memory_order_relaxed);
        if (static_cast<int>(n) >= config_.min_tokens) {
            double thr = config_.imbalance_threshold;
            if (std::abs(imb) >= thr) {
                fire = true;
                buy  = (imb > 0.0);
                imbalance_events_.fetch_add(1, std::memory_order_relaxed);
                cb = config_.on_imbalance;
            }
        }
    }

    if (fire && cb) cb(imb, buy);
}

// ---------------------------------------------------------------------------
// Accessors
// ---------------------------------------------------------------------------

double OrderFlowImbalanceDetector::imbalance() const noexcept {
    return imbalance_snapshot_.load(std::memory_order_relaxed);
}

bool OrderFlowImbalanceDetector::is_buy_imbalance() const noexcept {
    return imbalance() >= config_.imbalance_threshold;
}

bool OrderFlowImbalanceDetector::is_sell_imbalance() const noexcept {
    return imbalance() <= -config_.imbalance_threshold;
}

// ---------------------------------------------------------------------------
// Custom dictionary
// ---------------------------------------------------------------------------

void OrderFlowImbalanceDetector::add_token_weight(const std::string& keyword,
                                                   double pressure) {
    std::lock_guard<std::mutex> lk(mutex_);
    dict_[keyword] = pressure;
}

// ---------------------------------------------------------------------------
// Config / reset
// ---------------------------------------------------------------------------

void OrderFlowImbalanceDetector::update_config(const Config& config) {
    std::lock_guard<std::mutex> lk(mutex_);
    config_ = config;
}

OrderFlowImbalanceDetector::Config
OrderFlowImbalanceDetector::get_config() const {
    std::lock_guard<std::mutex> lk(mutex_);
    return config_;
}

void OrderFlowImbalanceDetector::reset() {
    std::lock_guard<std::mutex> lk(mutex_);
    ema_imbalance_ = 0.0;
    imbalance_snapshot_.store(0.0, std::memory_order_relaxed);
    total_tokens_.store(0, std::memory_order_relaxed);
    imbalance_events_.store(0, std::memory_order_relaxed);
}

void OrderFlowImbalanceDetector::set_imbalance_callback(
        std::function<void(double, bool)> cb) {
    std::lock_guard<std::mutex> lk(mutex_);
    config_.on_imbalance = std::move(cb);
}

// ---------------------------------------------------------------------------
// JSON stats
// ---------------------------------------------------------------------------

std::string OrderFlowImbalanceDetector::to_stats_json() const {
    double imb = imbalance_snapshot_.load(std::memory_order_relaxed);
    uint64_t tt = total_tokens_.load(std::memory_order_relaxed);
    uint64_t ie = imbalance_events_.load(std::memory_order_relaxed);

    std::ostringstream o;
    o << std::fixed;
    o.precision(4);
    o << "{\"imbalance\":"       << imb
      << ",\"is_buy\":"          << (is_buy_imbalance()  ? "true" : "false")
      << ",\"is_sell\":"         << (is_sell_imbalance() ? "true" : "false")
      << ",\"total_tokens\":"    << tt
      << ",\"imbalance_events\":" << ie << "}";
    return o.str();
}

} // namespace llmquant
