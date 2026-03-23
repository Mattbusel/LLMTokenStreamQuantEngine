#include "RegimeSwitcher.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <mutex>

namespace llmquant {

// ---------------------------------------------------------------------------
// Construction
// ---------------------------------------------------------------------------

RegimeSwitcher::RegimeSwitcher(Config cfg)
    : cfg_(std::move(cfg)),
      prob_risk_on_(0.5),
      prob_risk_off_(0.5) {}

// ---------------------------------------------------------------------------
// regime_name
// ---------------------------------------------------------------------------

const char* RegimeSwitcher::regime_name(Regime r) noexcept {
    switch (r) {
        case Regime::RiskOn:  return "RiskOn";
        case Regime::RiskOff: return "RiskOff";
    }
    return "Unknown";
}

// ---------------------------------------------------------------------------
// update
// ---------------------------------------------------------------------------

void RegimeSwitcher::update(double signal_value) {
    std::lock_guard<std::mutex> lk(mtx_);

    updates_.fetch_add(1, std::memory_order_relaxed);

    // Determine emission likelihood: is the observed signal positive?
    bool pos = (signal_value >= 0.0);

    // Emission probabilities for each state.
    double p_emit_on  = pos ? cfg_.p_pos_given_risk_on  : (1.0 - cfg_.p_pos_given_risk_on);
    double p_emit_off = pos ? cfg_.p_pos_given_risk_off : (1.0 - cfg_.p_pos_given_risk_off);

    // Forward algorithm (unnormalised):
    // alpha_t(s) = P(z_t | s) * sum_{s'} P(s | s') * alpha_{t-1}(s')
    double p_stay_on  = cfg_.p_stay_risk_on;
    double p_leave_on = 1.0 - p_stay_on;
    double p_stay_off = cfg_.p_stay_risk_off;
    double p_leave_off = 1.0 - p_stay_off;

    double alpha_on  = p_emit_on  * (p_stay_on  * prob_risk_on_ + p_leave_off * prob_risk_off_);
    double alpha_off = p_emit_off * (p_leave_on * prob_risk_on_ + p_stay_off  * prob_risk_off_);

    // Normalise to maintain valid posteriors.
    double norm = alpha_on + alpha_off;
    if (norm > 1e-300) {
        prob_risk_on_  = alpha_on  / norm;
        prob_risk_off_ = alpha_off / norm;
    }
    // (If norm is effectively zero — numerically underflow — keep previous posteriors.)

    update_regime_locked();
}

// ---------------------------------------------------------------------------
// update_regime_locked (called while mtx_ held)
// ---------------------------------------------------------------------------

void RegimeSwitcher::update_regime_locked() {
    Regime old_regime = static_cast<Regime>(regime_.load(std::memory_order_relaxed));
    Regime new_regime = old_regime;

    if (old_regime == Regime::RiskOn) {
        // Transition to RiskOff only when posterior crosses the off-threshold.
        if (prob_risk_off_ >= cfg_.risk_off_threshold) {
            new_regime = Regime::RiskOff;
        }
    } else {
        // Transition back to RiskOn only when posterior crosses the on-threshold.
        if (prob_risk_on_ >= cfg_.risk_on_threshold) {
            new_regime = Regime::RiskOn;
        }
    }

    if (new_regime != old_regime) {
        regime_.store(static_cast<int>(new_regime), std::memory_order_release);
        transitions_.fetch_add(1, std::memory_order_relaxed);
        if (regime_cb_) {
            regime_cb_(new_regime, old_regime);
        }
    }
}

// ---------------------------------------------------------------------------
// apply_weight
// ---------------------------------------------------------------------------

double RegimeSwitcher::apply_weight(double base_weight, bool token_is_positive) const {
    Regime r = static_cast<Regime>(regime_.load(std::memory_order_acquire));
    if (r == Regime::RiskOn) {
        return base_weight;
    }
    // RiskOff regime: discount positives, amplify negatives.
    if (token_is_positive) {
        return base_weight * cfg_.risk_off_pos_discount;
    } else {
        return base_weight * cfg_.risk_off_neg_multiplier;
    }
}

// ---------------------------------------------------------------------------
// risk_off_posterior
// ---------------------------------------------------------------------------

double RegimeSwitcher::risk_off_posterior() const noexcept {
    std::lock_guard<std::mutex> lk(mtx_);
    return prob_risk_off_;
}

// ---------------------------------------------------------------------------
// set_regime_callback / update_config / reset
// ---------------------------------------------------------------------------

void RegimeSwitcher::set_regime_callback(RegimeCallback cb) {
    std::lock_guard<std::mutex> lk(mtx_);
    regime_cb_ = std::move(cb);
}

void RegimeSwitcher::update_config(const Config& cfg) {
    std::lock_guard<std::mutex> lk(mtx_);
    cfg_ = cfg;
    prob_risk_on_  = 0.5;
    prob_risk_off_ = 0.5;
    regime_.store(static_cast<int>(Regime::RiskOn), std::memory_order_relaxed);
    transitions_.store(0, std::memory_order_relaxed);
    updates_.store(0, std::memory_order_relaxed);
}

void RegimeSwitcher::reset() {
    std::lock_guard<std::mutex> lk(mtx_);
    prob_risk_on_  = 0.5;
    prob_risk_off_ = 0.5;
    regime_.store(static_cast<int>(Regime::RiskOn), std::memory_order_relaxed);
    transitions_.store(0, std::memory_order_relaxed);
    updates_.store(0, std::memory_order_relaxed);
}

// ---------------------------------------------------------------------------
// to_stats_json
// ---------------------------------------------------------------------------

std::string RegimeSwitcher::to_stats_json() const {
    std::lock_guard<std::mutex> lk(mtx_);
    char buf[512];
    std::snprintf(buf, sizeof(buf),
        "{\"regime\":\"%s\",\"risk_off_posterior\":%.6f,\"risk_on_posterior\":%.6f,"
        "\"total_updates\":%llu,\"total_transitions\":%llu}",
        regime_name(static_cast<Regime>(regime_.load(std::memory_order_relaxed))),
        prob_risk_off_,
        prob_risk_on_,
        static_cast<unsigned long long>(updates_.load(std::memory_order_relaxed)),
        static_cast<unsigned long long>(transitions_.load(std::memory_order_relaxed)));
    return buf;
}

} // namespace llmquant
