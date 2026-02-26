#include "RiskManager.h"
#include <cmath>

namespace llmquant {

RiskManager::RiskManager(const Config& config)
    : config_(config)
    , rate_window_start_(std::chrono::high_resolution_clock::now())
    , drawdown_window_start_(std::chrono::high_resolution_clock::now()) {}

bool RiskManager::evaluate(const TradeSignal& signal) {
    std::lock_guard<std::mutex> lock(mutex_);

    if (!check_magnitude(signal)) {
        stats_.signals_blocked_magnitude++;
        fire_alert("magnitude_exceeded", signal);
        return false;
    }
    if (!check_confidence(signal)) {
        stats_.signals_blocked_confidence++;
        fire_alert("confidence_below_minimum", signal);
        return false;
    }
    if (!check_rate_limit()) {
        stats_.signals_blocked_rate++;
        fire_alert("rate_limit_exceeded", signal);
        return false;
    }
    if (!check_drawdown(signal)) {
        stats_.signals_blocked_drawdown++;
        fire_alert("drawdown_limit_exceeded", signal);
        return false;
    }

    update_drawdown(signal);
    signals_in_window_++;
    stats_.signals_passed++;
    return true;
}

void RiskManager::set_alert_callback(AlertCallback cb) {
    std::lock_guard<std::mutex> lock(mutex_);
    alert_cb_ = std::move(cb);
}

void RiskManager::reset() {
    std::lock_guard<std::mutex> lock(mutex_);
    auto now = std::chrono::high_resolution_clock::now();
    rate_window_start_     = now;
    drawdown_window_start_ = now;
    signals_in_window_     = 0;
    cumulative_bias_       = 0.0;
}

bool RiskManager::check_magnitude(const TradeSignal& signal) {
    return std::abs(signal.delta_bias_shift)      <= config_.max_bias_magnitude
        && std::abs(signal.volatility_adjustment)  <= config_.max_volatility_magnitude
        && std::abs(signal.spread_modifier)        <= config_.max_spread_magnitude;
}

bool RiskManager::check_confidence(const TradeSignal& signal) {
    return signal.confidence >= config_.min_confidence;
}

bool RiskManager::check_rate_limit() {
    auto now     = std::chrono::high_resolution_clock::now();
    auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(now - rate_window_start_);
    if (elapsed >= std::chrono::seconds{1}) {
        rate_window_start_ = now;
        signals_in_window_ = 0;
    }
    return signals_in_window_ < config_.max_signals_per_second;
}

bool RiskManager::check_drawdown(const TradeSignal& signal) {
    auto now     = std::chrono::high_resolution_clock::now();
    auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(now - drawdown_window_start_);
    if (elapsed >= config_.drawdown_window) {
        drawdown_window_start_ = now;
        cumulative_bias_       = 0.0;
    }
    return std::abs(cumulative_bias_ + signal.delta_bias_shift) <= config_.max_drawdown;
}

void RiskManager::update_drawdown(const TradeSignal& signal) {
    cumulative_bias_ += signal.delta_bias_shift;
}

void RiskManager::fire_alert(const std::string& reason, const TradeSignal& signal) {
    if (alert_cb_) alert_cb_(reason, signal);
}

} // namespace llmquant
