#include "RiskManager.h"
#include "MetricsLogger.h"
#include <cmath>
#include <optional>
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

std::optional<std::string> RiskManager::try_evaluate(const TradeSignal& signal) {
    std::string reason;
    if (evaluate_with_reason(signal, reason)) return std::nullopt;
    return reason;
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

RiskManager::BlockedByGate RiskManager::get_blocked_by_gate() const noexcept {
    BlockedByGate bg;
    bg.magnitude  = stats_.signals_blocked_magnitude.load(std::memory_order_relaxed);
    bg.confidence = stats_.signals_blocked_confidence.load(std::memory_order_relaxed);
    bg.rate       = stats_.signals_blocked_rate.load(std::memory_order_relaxed);
    bg.drawdown   = stats_.signals_blocked_drawdown.load(std::memory_order_relaxed);
    bg.position   = stats_.signals_blocked_position.load(std::memory_order_relaxed);
    bg.pnl        = stats_.signals_blocked_pnl.load(std::memory_order_relaxed);
    return bg;
}

RiskManager::GateStatus RiskManager::get_gate_status() const {
    std::lock_guard<std::mutex> lock(mutex_);
    GateStatus gs;
    gs.magnitude_gate  = !config_.disable_magnitude_gate;
    gs.confidence_gate = !config_.disable_confidence_gate;
    gs.rate_gate       = !config_.disable_rate_gate;
    gs.drawdown_gate   = !config_.disable_drawdown_gate;
    gs.position_gate   = !config_.disable_position_gate;
    return gs;
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

bool RiskManager::is_healthy() const {
    std::lock_guard<std::mutex> lock(mutex_);
    if (config_.disable_magnitude_gate || config_.disable_confidence_gate ||
        config_.disable_rate_gate      || config_.disable_drawdown_gate    ||
        config_.disable_position_gate)
        return false;
    double remaining = config_.max_drawdown - std::fabs(cumulative_bias_);
    if (remaining <= 0.0) return false;
    if (std::fabs(position_.net_position) >= position_.position_limit) return false;
    if (position_.pnl < position_.pnl_limit) return false;
    return true;
}

void RiskManager::set_position_limit(double position_limit, double pnl_limit) {
    std::lock_guard<std::mutex> lock(mutex_);
    position_.position_limit = position_limit;
    position_.pnl_limit      = pnl_limit;
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

double RiskManager::get_net_exposure() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return position_.net_position;
}

bool RiskManager::is_rate_limited() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return signals_in_window_ >= config_.max_signals_per_second;
}

double RiskManager::get_window_time_elapsed_ms() const {
    std::lock_guard<std::mutex> lock(mutex_);
    auto now = std::chrono::high_resolution_clock::now();
    return static_cast<double>(
        std::chrono::duration_cast<std::chrono::milliseconds>(
            now - rate_window_start_).count());
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

double RiskManager::get_drawdown_utilization() const {
    std::lock_guard<std::mutex> lock(mutex_);
    if (config_.max_drawdown <= 0.0) return 0.0;
    double util = std::fabs(cumulative_bias_) / config_.max_drawdown;
    return util > 1.0 ? 1.0 : util;
}

RiskManager::Snapshot RiskManager::snapshot() const {
    Snapshot snap;
    {
        std::lock_guard<std::mutex> lock(mutex_);
        snap.config   = config_;
        snap.position = position_;
        snap.cumulative_bias = cumulative_bias_;
        double remaining = config_.max_drawdown - std::fabs(cumulative_bias_);
        snap.drawdown_budget_remaining = remaining < 0.0 ? 0.0 : remaining;
        double util = (config_.max_signals_per_second > 0)
            ? static_cast<double>(signals_in_window_) / static_cast<double>(config_.max_signals_per_second)
            : 1.0;
        snap.rate_limit_utilization = util > 1.0 ? 1.0 : util;
    }
    snap.signals_passed         = stats_.signals_passed.load(std::memory_order_relaxed);
    snap.signals_blocked_total  = stats_.blocked_total();
    return snap;
}

double RiskManager::get_blocked_rate() const noexcept {
    uint64_t blocked = stats_.blocked_total();
    uint64_t passed  = stats_.signals_passed.load(std::memory_order_relaxed);
    uint64_t total   = blocked + passed;
    return (total == 0) ? 0.0 : static_cast<double>(blocked) / static_cast<double>(total);
}

uint64_t RiskManager::get_total_evaluated() const noexcept {
    return stats_.blocked_total()
         + stats_.signals_passed.load(std::memory_order_relaxed);
}

double RiskManager::get_rate_limit_utilization() const {
    std::lock_guard<std::mutex> lock(mutex_);
    if (config_.max_signals_per_second == 0) return 1.0;
    double util = static_cast<double>(signals_in_window_)
                  / static_cast<double>(config_.max_signals_per_second);
    return util > 1.0 ? 1.0 : util;
}

double RiskManager::get_signals_per_second() const {
    std::lock_guard<std::mutex> lock(mutex_);
    auto now     = std::chrono::high_resolution_clock::now();
    double elapsed_ms = static_cast<double>(
        std::chrono::duration_cast<std::chrono::milliseconds>(
            now - rate_window_start_).count());
    if (elapsed_ms < 1.0) return 0.0;
    return static_cast<double>(signals_in_window_) / (elapsed_ms / 1000.0);
}

std::vector<bool> RiskManager::evaluate_batch(const std::vector<TradeSignal>& signals) {
    std::vector<bool> results;
    results.reserve(signals.size());
    for (const auto& sig : signals) {
        results.push_back(evaluate(sig));
    }
    return results;
}

double RiskManager::get_rejection_rate() const noexcept {
    const auto& s = stats_;
    uint64_t passed  = s.signals_passed.load(std::memory_order_relaxed);
    uint64_t blocked = s.signals_blocked_magnitude.load(std::memory_order_relaxed)
                     + s.signals_blocked_confidence.load(std::memory_order_relaxed)
                     + s.signals_blocked_rate.load(std::memory_order_relaxed)
                     + s.signals_blocked_drawdown.load(std::memory_order_relaxed)
                     + s.signals_blocked_position.load(std::memory_order_relaxed);
    uint64_t total = passed + blocked;
    if (total == 0) return 0.0;
    return static_cast<double>(blocked) / static_cast<double>(total);
}

} // namespace llmquant
