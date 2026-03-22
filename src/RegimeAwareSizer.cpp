#include "RegimeAwareSizer.h"

#include <algorithm>
#include <cmath>
#include <sstream>

namespace llmquant {

RegimeAwareSizer::RegimeAwareSizer(Config config)
    : config_(std::move(config)) {}

// ---------------------------------------------------------------------------
// Internal recompute
// ---------------------------------------------------------------------------

double RegimeAwareSizer::recompute_locked() {
    double h   = hurst_val_;
    double vol = vol_val_;

    // ── Regime factor ──────────────────────────────────────────────────────
    // Map H to [regime_min, regime_max].
    //   H = 0.5  → regime_neutral
    //   H = hurst_max → regime_max
    //   H < 0.5  → interpolate toward regime_min
    double rf;
    if (h >= 0.5) {
        double span = config_.hurst_max - 0.5;
        double t    = (span > 1e-12) ? std::min(1.0, (h - 0.5) / span) : 0.0;
        rf = config_.regime_neutral + t * (config_.regime_max - config_.regime_neutral);
    } else {
        // Below random-walk: interpolate between regime_min and regime_neutral.
        double t = std::max(0.0, h / 0.5);  // 0 at H=0, 1 at H=0.5
        rf = config_.regime_min + t * (config_.regime_neutral - config_.regime_min);
    }

    // ── Volatility factor ──────────────────────────────────────────────────
    double vf = 1.0;
    if (vol > 1e-12 && config_.target_vol > 1e-12) {
        vf = config_.target_vol / vol;
        vf = std::max(config_.vol_floor, std::min(config_.vol_cap, vf));
    }

    double mult = rf * vf;

    regime_factor_.store(rf,   std::memory_order_relaxed);
    vol_factor_.store(vf,      std::memory_order_relaxed);
    multiplier_.store(mult,    std::memory_order_relaxed);
    update_count_.fetch_add(1, std::memory_order_relaxed);

    return mult;
}

// ---------------------------------------------------------------------------
// Public interface
// ---------------------------------------------------------------------------

void RegimeAwareSizer::update_hurst(double h) noexcept {
    double old_mult, new_mult;
    Config cfg_copy;

    {
        std::lock_guard<std::mutex> lk(mutex_);
        hurst_val_ = h;
        hurst_.store(h, std::memory_order_relaxed);
        old_mult  = multiplier_.load(std::memory_order_relaxed);
        cfg_copy  = config_;
        new_mult  = recompute_locked();
    }

    double threshold = cfg_copy.change_threshold;
    if (std::abs(new_mult - old_mult) / (std::abs(old_mult) + 1e-12) > threshold) {
        change_events_.fetch_add(1, std::memory_order_relaxed);
        if (cfg_copy.on_size_change)
            cfg_copy.on_size_change(new_mult, old_mult);
    }
}

void RegimeAwareSizer::update_vol(double vol) noexcept {
    if (vol <= 0.0) return;
    double old_mult, new_mult;
    Config cfg_copy;

    {
        std::lock_guard<std::mutex> lk(mutex_);
        vol_val_ = vol;
        vol_.store(vol, std::memory_order_relaxed);
        old_mult = multiplier_.load(std::memory_order_relaxed);
        cfg_copy = config_;
        new_mult = recompute_locked();
    }

    double threshold = cfg_copy.change_threshold;
    if (std::abs(new_mult - old_mult) / (std::abs(old_mult) + 1e-12) > threshold) {
        change_events_.fetch_add(1, std::memory_order_relaxed);
        if (cfg_copy.on_size_change)
            cfg_copy.on_size_change(new_mult, old_mult);
    }
}

void RegimeAwareSizer::update_config(Config config) {
    std::lock_guard<std::mutex> lk(mutex_);
    config_ = std::move(config);
    hurst_val_ = 0.5;
    vol_val_   = 0.0;
    hurst_.store(0.5, std::memory_order_relaxed);
    vol_.store(0.0, std::memory_order_relaxed);
    regime_factor_.store(1.0, std::memory_order_relaxed);
    vol_factor_.store(1.0, std::memory_order_relaxed);
    multiplier_.store(1.0, std::memory_order_relaxed);
    update_count_.store(0, std::memory_order_relaxed);
    change_events_.store(0, std::memory_order_relaxed);
}

void RegimeAwareSizer::reset() {
    std::lock_guard<std::mutex> lk(mutex_);
    hurst_val_ = 0.5;
    vol_val_   = 0.0;
    hurst_.store(0.5, std::memory_order_relaxed);
    vol_.store(0.0, std::memory_order_relaxed);
    regime_factor_.store(1.0, std::memory_order_relaxed);
    vol_factor_.store(1.0, std::memory_order_relaxed);
    multiplier_.store(1.0, std::memory_order_relaxed);
    update_count_.store(0, std::memory_order_relaxed);
    change_events_.store(0, std::memory_order_relaxed);
}

std::string RegimeAwareSizer::to_stats_json() const {
    double m   = multiplier_.load(std::memory_order_relaxed);
    double rf  = regime_factor_.load(std::memory_order_relaxed);
    double vf  = vol_factor_.load(std::memory_order_relaxed);
    double h   = hurst_.load(std::memory_order_relaxed);
    double v   = vol_.load(std::memory_order_relaxed);
    uint64_t u = update_count_.load(std::memory_order_relaxed);
    uint64_t c = change_events_.load(std::memory_order_relaxed);

    std::ostringstream o;
    o << std::fixed;
    o.precision(4);
    o << "{\"size_multiplier\":"  << m
      << ",\"regime_factor\":"    << rf
      << ",\"vol_factor\":"       << vf
      << ",\"hurst\":"            << h
      << ",\"vol\":"              << v
      << ",\"update_count\":"     << u
      << ",\"change_events\":"    << c << "}";
    return o.str();
}

} // namespace llmquant
