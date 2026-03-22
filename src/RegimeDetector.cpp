#include "RegimeDetector.h"

#include <cmath>
#include <spdlog/spdlog.h>
#include <sstream>

namespace llmquant {

// ---------------------------------------------------------------------------
// Construction
// ---------------------------------------------------------------------------

RegimeDetector::RegimeDetector(Config config)
    : config_(std::move(config)) {}

// ---------------------------------------------------------------------------
// Classification (caller holds mutex)
// ---------------------------------------------------------------------------

RegimeDetector::Regime RegimeDetector::classify_locked() const {
    double momentum   = momentum_ema_.load(std::memory_order_relaxed);
    double volatility = volatility_ema_.load(std::memory_order_relaxed);
    double block_rate = block_rate_ema_.load(std::memory_order_relaxed);

    // Priority order: RiskOff > Volatile > Bull/Bear > Neutral.
    if (block_rate >= config_.risk_off_threshold) return Regime::RiskOff;
    if (volatility  >= config_.volatility_threshold)  return Regime::Volatile;
    if (momentum    >  config_.momentum_threshold)     return Regime::Bull;
    if (momentum    < -config_.momentum_threshold)     return Regime::Bear;
    return Regime::Neutral;
}

// ---------------------------------------------------------------------------
// Hot path: update
// ---------------------------------------------------------------------------

void RegimeDetector::update(double bias_shift, double volatility, bool signal_blocked) {
    updates_.fetch_add(1, std::memory_order_relaxed);

    // Update EMAs (relaxed: minor races acceptable for statistical estimators).
    const double ma = config_.momentum_alpha;
    const double va = config_.volatility_alpha;
    const double ba = config_.block_rate_alpha;

    double old_m = momentum_ema_.load(std::memory_order_relaxed);
    momentum_ema_.store(ma * bias_shift + (1.0 - ma) * old_m,
                        std::memory_order_relaxed);

    double old_v = volatility_ema_.load(std::memory_order_relaxed);
    volatility_ema_.store(va * std::fabs(volatility) + (1.0 - va) * old_v,
                          std::memory_order_relaxed);

    double block_sample = signal_blocked ? 1.0 : 0.0;
    double old_b = block_rate_ema_.load(std::memory_order_relaxed);
    block_rate_ema_.store(ba * block_sample + (1.0 - ba) * old_b,
                          std::memory_order_relaxed);

    // Reclassify and check for transition.
    Regime old_regime = static_cast<Regime>(regime_.load(std::memory_order_acquire));
    Regime new_regime;
    RegimeChangeCallback cb;
    {
        std::lock_guard<std::mutex> lk(mutex_);
        new_regime = classify_locked();
        if (new_regime != old_regime) {
            regime_.store(static_cast<int>(new_regime), std::memory_order_release);
            transitions_.fetch_add(1, std::memory_order_relaxed);
            spdlog::info("[regime] {} → {} (momentum={:.3f} vol={:.3f} block={:.0f}%)",
                         regime_name(old_regime), regime_name(new_regime),
                         momentum_ema_.load(std::memory_order_relaxed),
                         volatility_ema_.load(std::memory_order_relaxed),
                         block_rate_ema_.load(std::memory_order_relaxed) * 100.0);
            cb = regime_cb_;
        }
    }
    if (cb) {
        try { cb(new_regime, old_regime); } catch (...) {}
    }
}

// ---------------------------------------------------------------------------
// Utilities
// ---------------------------------------------------------------------------

const char* RegimeDetector::regime_name(Regime r) noexcept {
    switch (r) {
        case Regime::Neutral:  return "Neutral";
        case Regime::Bull:     return "Bull";
        case Regime::Bear:     return "Bear";
        case Regime::Volatile: return "Volatile";
        case Regime::RiskOff:  return "RiskOff";
    }
    return "Unknown";
}

void RegimeDetector::set_regime_change_callback(RegimeChangeCallback cb) {
    std::lock_guard<std::mutex> lk(mutex_);
    regime_cb_ = std::move(cb);
}

void RegimeDetector::update_config(const Config& config) {
    std::lock_guard<std::mutex> lk(mutex_);
    config_ = config;
}

RegimeDetector::Config RegimeDetector::get_config() const {
    std::lock_guard<std::mutex> lk(mutex_);
    return config_;
}

std::string RegimeDetector::to_stats_json() const {
    std::ostringstream o;
    o << "{\"regime\":\""      << regime_name(current_regime()) << "\""
      << ",\"momentum\":"      << momentum_ema_.load(std::memory_order_relaxed)
      << ",\"volatility_ema\":" << volatility_ema_.load(std::memory_order_relaxed)
      << ",\"block_rate_ema\":" << block_rate_ema_.load(std::memory_order_relaxed)
      << ",\"total_updates\":" << updates_.load(std::memory_order_relaxed)
      << ",\"total_transitions\":" << transitions_.load(std::memory_order_relaxed)
      << "}";
    return o.str();
}

void RegimeDetector::reset() {
    momentum_ema_.store(0.0, std::memory_order_relaxed);
    volatility_ema_.store(0.0, std::memory_order_relaxed);
    block_rate_ema_.store(0.0, std::memory_order_relaxed);
    regime_.store(static_cast<int>(Regime::Neutral), std::memory_order_release);
    transitions_.store(0, std::memory_order_relaxed);
    updates_.store(0, std::memory_order_relaxed);
    std::lock_guard<std::mutex> lk(mutex_);
    // (config and callback preserved)
}

} // namespace llmquant
