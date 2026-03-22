#include "SignalDecayEnvelope.h"

#include <algorithm>
#include <cstdio>
#include <mutex>

namespace llmquant {

SignalDecayEnvelope::SignalDecayEnvelope(Config cfg) : cfg_(std::move(cfg)) {}

// ---------------------------------------------------------------------------

double SignalDecayEnvelope::compute_decayed(double raw, int64_t age_ms) const {
    if (cfg_.half_life_ms <= 0.0) return raw;
    double lambda = std::log(2.0) / cfg_.half_life_ms;
    double result = raw * std::exp(-lambda * static_cast<double>(age_ms));
    if (cfg_.clamp) result = std::clamp(result, cfg_.min_bias, cfg_.max_bias);
    return result;
}

void SignalDecayEnvelope::reinforce(double delta) {
    std::lock_guard<std::mutex> lk(mtx_);
    raw_bias_ += delta;
    if (cfg_.clamp) raw_bias_ = std::clamp(raw_bias_, cfg_.min_bias, cfg_.max_bias);
    last_reinforce_time_ = Clock::now();
    total_reinforcements_.fetch_add(1, std::memory_order_relaxed);
}

double SignalDecayEnvelope::decayed_bias() const {
    std::lock_guard<std::mutex> lk(mtx_);
    int64_t age_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
        Clock::now() - last_reinforce_time_).count();
    double val = compute_decayed(raw_bias_, age_ms);

    // Zero-cross detection.
    double sign = (val > 0.0) ? 1.0 : (val < 0.0 ? -1.0 : 0.0);
    if (last_reported_sign_ != 0.0 && sign != 0.0 && sign != last_reported_sign_) {
        if (zero_cross_cb_) zero_cross_cb_(last_reported_sign_ * std::abs(raw_bias_), val);
    }
    if (sign != 0.0) const_cast<SignalDecayEnvelope*>(this)->last_reported_sign_ = sign;

    return val;
}

double SignalDecayEnvelope::raw_bias() const {
    std::lock_guard<std::mutex> lk(mtx_);
    return raw_bias_;
}

int64_t SignalDecayEnvelope::ms_since_last_reinforce() const {
    std::lock_guard<std::mutex> lk(mtx_);
    return std::chrono::duration_cast<std::chrono::milliseconds>(
        Clock::now() - last_reinforce_time_).count();
}

void SignalDecayEnvelope::reset() {
    std::lock_guard<std::mutex> lk(mtx_);
    raw_bias_           = 0.0;
    last_reported_sign_ = 0.0;
    last_reinforce_time_ = Clock::now();
}

void SignalDecayEnvelope::set_zero_cross_callback(DecayCallback cb) {
    std::lock_guard<std::mutex> lk(mtx_);
    zero_cross_cb_ = std::move(cb);
}

void SignalDecayEnvelope::update_config(const Config& cfg) {
    std::lock_guard<std::mutex> lk(mtx_);
    cfg_ = cfg;
    raw_bias_           = 0.0;
    last_reported_sign_ = 0.0;
    last_reinforce_time_ = Clock::now();
}

std::string SignalDecayEnvelope::to_stats_json() const {
    std::lock_guard<std::mutex> lk(mtx_);
    int64_t age_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
        Clock::now() - last_reinforce_time_).count();
    double decayed = compute_decayed(raw_bias_, age_ms);
    char buf[256];
    std::snprintf(buf, sizeof(buf),
        "{\"raw_bias\":%.6f,\"decayed_bias\":%.6f,\"age_ms\":%lld,"
        "\"half_life_ms\":%.1f,\"total_reinforcements\":%llu}",
        raw_bias_, decayed, static_cast<long long>(age_ms),
        cfg_.half_life_ms,
        static_cast<unsigned long long>(total_reinforcements_.load(std::memory_order_relaxed)));
    return buf;
}

} // namespace llmquant
