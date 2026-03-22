#include "RegimeSwitchingSignalRouter.h"

#include <cmath>
#include <sstream>
#include <iomanip>

namespace llmquant {

RegimeSwitchingSignalRouter::RegimeSwitchingSignalRouter(Config cfg)
    : cfg_(std::move(cfg))
{
    apply_regime_locked(Regime::Ranging);
}

void RegimeSwitchingSignalRouter::record_return(double ret) {
    std::function<void(Regime, Regime, double)> cb;
    bool   fire     = false;
    Regime old_r    = Regime::Ranging;
    Regime new_r    = Regime::Ranging;
    double new_mult = 1.0;

    {
        std::lock_guard<std::mutex> lk(mutex_);
        cb = cfg_.on_regime_change;

        if (!seeded_) {
            prev_ret_ = ret;
            seeded_   = true;
            return;
        }

        // Volatility proxy: EMA of |Δret|.
        double abs_delta = std::abs(ret - prev_ret_);
        vol_ema_val_ += cfg_.vol_alpha * (abs_delta - vol_ema_val_);
        vol_ema_.store(vol_ema_val_, std::memory_order_relaxed);

        // Momentum: EMA of ret itself.
        mom_ema_val_ += cfg_.momentum_alpha * (ret - mom_ema_val_);
        mom_ema_.store(mom_ema_val_, std::memory_order_relaxed);

        prev_ret_ = ret;

        Regime candidate = classify_locked();
        if (candidate != current_regime_val_) {
            old_r = current_regime_val_;
            new_r = candidate;
            apply_regime_locked(candidate);
            regime_changes_.fetch_add(1, std::memory_order_relaxed);
            new_mult = bias_mult_.load(std::memory_order_relaxed);
            fire = true;
        }
    }

    if (fire && cb) cb(old_r, new_r, new_mult);
}

RegimeSwitchingSignalRouter::Regime
RegimeSwitchingSignalRouter::classify_locked() const {
    double h = cfg_.hysteresis;
    // Crash: vol above threshold (with hysteresis).
    double crash_hi  = cfg_.crash_vol_threshold;
    double crash_lo  = crash_hi * (1.0 - h);
    double trend_hi  = cfg_.trend_threshold;
    double trend_lo  = trend_hi * (1.0 - h);

    if (current_regime_val_ == Regime::Crash) {
        if (vol_ema_val_ < crash_lo) {
            // Fall through to trending/ranging check.
        } else {
            return Regime::Crash;
        }
    } else {
        if (vol_ema_val_ >= crash_hi) return Regime::Crash;
    }

    if (current_regime_val_ == Regime::Trending) {
        if (std::abs(mom_ema_val_) > trend_lo) return Regime::Trending;
    } else {
        if (std::abs(mom_ema_val_) > trend_hi) return Regime::Trending;
    }

    return Regime::Ranging;
}

void RegimeSwitchingSignalRouter::apply_regime_locked(Regime r) {
    current_regime_val_ = r;
    regime_.store(static_cast<int>(r), std::memory_order_relaxed);

    const RegimeConfig* rc;
    switch (r) {
        case Regime::Trending: rc = &cfg_.trending_cfg; break;
        case Regime::Crash:    rc = &cfg_.crash_cfg;    break;
        default:               rc = &cfg_.ranging_cfg;  break;
    }

    bias_mult_.store(rc->bias_multiplier,   std::memory_order_relaxed);
    cooldown_ms_.store(rc->cooldown_ms,     std::memory_order_relaxed);
    risk_frac_.store(rc->max_risk_fraction, std::memory_order_relaxed);
}

std::string RegimeSwitchingSignalRouter::regime_name() const noexcept {
    switch (current_regime()) {
        case Regime::Trending: return "Trending";
        case Regime::Crash:    return "Crash";
        default:               return "Ranging";
    }
}

std::string RegimeSwitchingSignalRouter::to_stats_json() const {
    Regime r;
    double v, m, bm, rf;
    int cd;
    uint64_t rc;
    {
        std::lock_guard<std::mutex> lk(mutex_);
        r  = current_regime_val_;
        v  = vol_ema_val_;
        m  = mom_ema_val_;
        bm = bias_mult_.load(std::memory_order_relaxed);
        cd = cooldown_ms_.load(std::memory_order_relaxed);
        rf = risk_frac_.load(std::memory_order_relaxed);
        rc = regime_changes_.load(std::memory_order_relaxed);
    }
    const char* names[] = {"Trending", "Ranging", "Crash"};
    std::ostringstream ss;
    ss << std::fixed << std::setprecision(6);
    ss << "{"
       << "\"regime\":\"" << names[static_cast<int>(r)] << "\","
       << "\"vol_ema\":"          << v  << ","
       << "\"momentum_ema\":"     << m  << ","
       << "\"bias_multiplier\":"  << bm << ","
       << "\"cooldown_ms\":"      << cd << ","
       << "\"max_risk_fraction\":" << rf << ","
       << "\"regime_changes\":"   << rc
       << "}";
    return ss.str();
}

void RegimeSwitchingSignalRouter::reset() {
    std::lock_guard<std::mutex> lk(mutex_);
    vol_ema_val_ = mom_ema_val_ = prev_ret_ = 0.0;
    seeded_ = false;
    apply_regime_locked(Regime::Ranging);
    regime_changes_.store(0, std::memory_order_relaxed);
    vol_ema_.store(0.0, std::memory_order_relaxed);
    mom_ema_.store(0.0, std::memory_order_relaxed);
}

void RegimeSwitchingSignalRouter::update_config(const Config& cfg) {
    std::lock_guard<std::mutex> lk(mutex_);
    cfg_ = cfg;
    vol_ema_val_ = mom_ema_val_ = prev_ret_ = 0.0;
    seeded_ = false;
    apply_regime_locked(Regime::Ranging);
}

} // namespace llmquant
