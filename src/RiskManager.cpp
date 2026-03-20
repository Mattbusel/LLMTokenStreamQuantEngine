#include "RiskManager.h"
#include "MetricsLogger.h"
#include <cmath>
#include <limits>
#include <spdlog/spdlog.h>
#include <stdexcept>

// Helper: saturating increment for uint64_t atomics.
// Saturating increment — wraps at UINT64_MAX then holds
static inline void sat_increment(std::atomic<uint64_t>& counter) {
    if (counter.load(std::memory_order_relaxed) < std::numeric_limits<uint64_t>::max() - 1)
        counter.fetch_add(1, std::memory_order_relaxed);
}

namespace llmquant {

RiskManager::RiskManager(const Config& config)
    : config_(config)
    , rate_window_start_(std::chrono::high_resolution_clock::now())
    , drawdown_window_start_(std::chrono::high_resolution_clock::now()) {
    if (config_.max_bias_magnitude < 0.0)
        throw std::invalid_argument("RiskManager: max_bias_magnitude must be >= 0");
    if (config_.min_confidence < 0.0 || config_.min_confidence > 1.0)
        throw std::invalid_argument("RiskManager: min_confidence must be in [0, 1]");
    if (config_.max_signals_per_second == 0)
        throw std::invalid_argument("RiskManager: max_signals_per_second must be > 0");
    if (config_.max_drawdown < 0.0)
        throw std::invalid_argument("RiskManager: max_drawdown must be >= 0");
}

bool RiskManager::evaluate(const TradeSignal& signal) {
    // Collect all callbacks and the reject reason under the lock, then fire
    // them outside the lock to prevent deadlock if a callback re-enters evaluate().
    std::string    reject_reason;
    AlertCallback  alert_cb_copy;
    OmsCallback    oms_cb_copy;
    MetricsLogger* logger_copy = nullptr;
    PositionState  pos_copy;
    bool hard_breach = false;
    bool soft_warn   = false;
    bool pnl_breach  = false;

    {
        std::lock_guard<std::mutex> lock(mutex_);

        if (!config_.disable_magnitude_gate && !check_magnitude(signal)) {
            sat_increment(stats_.signals_blocked_magnitude);
            reject_reason = "magnitude_exceeded";
        } else if (!config_.disable_confidence_gate && !check_confidence(signal)) {
            sat_increment(stats_.signals_blocked_confidence);
            reject_reason = "confidence_below_minimum";
        } else if (!config_.disable_rate_gate && !check_rate_limit()) {
            sat_increment(stats_.signals_blocked_rate);
            reject_reason = "rate_limit_exceeded";
        } else if (!config_.disable_drawdown_gate && !check_drawdown(signal)) {
            sat_increment(stats_.signals_blocked_drawdown);
            reject_reason = "drawdown_limit_exceeded";
        } else {
            double projected = position_.net_position + signal.delta_bias_shift;
            double limit     = position_.position_limit;
            if (!config_.disable_position_gate) {
                // Check position hard breach.
                if (std::fabs(projected) > limit) {
                    hard_breach = true;
                    sat_increment(stats_.signals_blocked_position);
                    reject_reason = "position_limit";
                }
                // Check soft warn independently of hard breach.
                if (!hard_breach && std::fabs(projected) > limit * config_.position_warn_fraction) {
                    soft_warn = true;
                }
                // Check PnL breach always — not nested inside else of position check.
                if (position_.pnl < position_.pnl_limit) {
                    pnl_breach = true;
                    if (!hard_breach) {
                        // Increment both the specific PnL counter and the aggregate
                        // position counter; tests and Prometheus consumers rely on
                        // signals_blocked_position for all OMS-gate rejections.
                        sat_increment(stats_.signals_blocked_pnl);
                        sat_increment(stats_.signals_blocked_position);
                        reject_reason = "pnl_limit";
                    }
                }
            }
            if (reject_reason.empty()) {
                update_drawdown(signal);
                signals_in_window_++;
                sat_increment(stats_.signals_passed);
            }
        }

        // Capture pointers so callbacks can be fired outside the lock.
        if (!reject_reason.empty() || soft_warn) {
            alert_cb_copy = alert_cb_;
            oms_cb_copy   = oms_cb_;
            logger_copy   = logger_;
            pos_copy      = position_;
        }
    }

    // Fire OMS callbacks outside the lock.
    if ((hard_breach || pnl_breach) && oms_cb_copy) {
        const char* ev = hard_breach ? "position_limit_breached" : "pnl_limit_breached";
        try { oms_cb_copy(ev, pos_copy, signal); }
        catch (const std::exception& e) { spdlog::warn("RiskManager: callback threw: {}", e.what()); }
        catch (...) { spdlog::warn("RiskManager: callback threw unknown exception"); }
    }
    if (soft_warn && oms_cb_copy) {
        try { oms_cb_copy("position_limit_approaching", pos_copy, signal); }
        catch (const std::exception& e) { spdlog::warn("RiskManager: callback threw: {}", e.what()); }
        catch (...) { spdlog::warn("RiskManager: callback threw unknown exception"); }
    }

    // Fire alert / log callbacks outside the lock.
    if (!reject_reason.empty()) {
        if (alert_cb_copy) {
            try { alert_cb_copy(reject_reason, signal); }
            catch (const std::exception& e) { spdlog::warn("RiskManager: callback threw: {}", e.what()); }
            catch (...) { spdlog::warn("RiskManager: callback threw unknown exception"); }
        }
        if (logger_copy)   { logger_copy->log_risk_rejection(reject_reason, signal.delta_bias_shift, signal.confidence); }
        return false;
    }
    return true;
}

bool RiskManager::evaluate_with_reason(const TradeSignal& signal, std::string& reason) {
    reason.clear();
    // Temporarily wrap the alert callback to capture the rejection reason,
    // then restore the original after evaluate() returns.
    AlertCallback saved_cb;
    std::string captured;
    {
        std::lock_guard<std::mutex> lock(mutex_);
        saved_cb = alert_cb_;
        AlertCallback sc_copy = saved_cb;
        alert_cb_ = [&captured, sc_copy](const std::string& r, const TradeSignal& s) {
            captured = r;
            if (sc_copy) sc_copy(r, s);
        };
    }
    bool result = evaluate(signal);
    {
        std::lock_guard<std::mutex> lock(mutex_);
        alert_cb_ = std::move(saved_cb);
    }
    reason = std::move(captured);
    return result;
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

void RiskManager::reset_drawdown() {
    std::lock_guard<std::mutex> lock(mutex_);
    drawdown_window_start_ = std::chrono::high_resolution_clock::now();
    cumulative_bias_       = 0.0;
}

void RiskManager::reset_stats() noexcept {
    stats_.signals_passed.store(0, std::memory_order_relaxed);
    stats_.signals_blocked_magnitude.store(0, std::memory_order_relaxed);
    stats_.signals_blocked_confidence.store(0, std::memory_order_relaxed);
    stats_.signals_blocked_rate.store(0, std::memory_order_relaxed);
    stats_.signals_blocked_drawdown.store(0, std::memory_order_relaxed);
    stats_.signals_blocked_position.store(0, std::memory_order_relaxed);
    stats_.signals_blocked_pnl.store(0, std::memory_order_relaxed);
}

void RiskManager::disable_all_gates() {
    std::lock_guard<std::mutex> lock(mutex_);
    config_.disable_magnitude_gate = true;
    config_.disable_confidence_gate = true;
    config_.disable_rate_gate = true;
    config_.disable_drawdown_gate = true;
    config_.disable_position_gate = true;
}

void RiskManager::enable_all_gates() {
    std::lock_guard<std::mutex> lock(mutex_);
    config_.disable_magnitude_gate = false;
    config_.disable_confidence_gate = false;
    config_.disable_rate_gate = false;
    config_.disable_drawdown_gate = false;
    config_.disable_position_gate = false;
}

void RiskManager::update_config(const Config& config) {
    if (config.max_bias_magnitude < 0.0)
        throw std::invalid_argument("RiskManager: max_bias_magnitude must be >= 0");
    if (config.max_volatility_magnitude < 0.0)
        throw std::invalid_argument("RiskManager: max_volatility_magnitude must be >= 0");
    if (config.max_spread_magnitude < 0.0)
        throw std::invalid_argument("RiskManager: max_spread_magnitude must be >= 0");
    if (config.min_confidence < 0.0 || config.min_confidence > 1.0)
        throw std::invalid_argument("RiskManager: min_confidence must be in [0, 1]");
    if (config.max_drawdown < 0.0)
        throw std::invalid_argument("RiskManager: max_drawdown must be >= 0");
    std::lock_guard<std::mutex> lock(mutex_);
    config_ = config;
}

bool RiskManager::check_magnitude(const TradeSignal& signal) {
    return std::fabs(signal.delta_bias_shift)      <= config_.max_bias_magnitude
        && std::fabs(signal.volatility_adjustment)  <= config_.max_volatility_magnitude
        && std::fabs(signal.spread_modifier)        <= config_.max_spread_magnitude;
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
    return std::fabs(cumulative_bias_ + signal.delta_bias_shift) <= config_.max_drawdown;
}

void RiskManager::update_drawdown(const TradeSignal& signal) {
    cumulative_bias_ += signal.delta_bias_shift;
}

void RiskManager::set_metrics_logger(MetricsLogger* logger) {
    std::lock_guard<std::mutex> lock(mutex_);
    logger_ = logger;
}

void RiskManager::update_position(const PositionState& state) {
    std::lock_guard<std::mutex> lock(mutex_);
    position_ = state;
}

void RiskManager::set_oms_callback(OmsCallback cb) {
    std::lock_guard<std::mutex> lock(mutex_);
    oms_cb_ = std::move(cb);
}

RiskManager::PositionState RiskManager::get_position() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return position_;
}

double RiskManager::get_cumulative_bias() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return cumulative_bias_;
}

double RiskManager::get_drawdown_budget_remaining() const {
    std::lock_guard<std::mutex> lock(mutex_);
    double remaining = config_.max_drawdown - std::fabs(cumulative_bias_);
    return remaining < 0.0 ? 0.0 : remaining;
}

double RiskManager::get_rate_limit_utilization() const {
    std::lock_guard<std::mutex> lock(mutex_);
    if (config_.max_signals_per_second == 0) return 1.0;
    double util = static_cast<double>(signals_in_window_)
                  / static_cast<double>(config_.max_signals_per_second);
    return util > 1.0 ? 1.0 : util;
}

} // namespace llmquant
