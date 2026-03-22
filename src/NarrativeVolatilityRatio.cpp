#include "NarrativeVolatilityRatio.h"
#include <cmath>
#include <sstream>
#include <iomanip>

namespace llmquant {

NarrativeVolatilityRatio::NarrativeVolatilityRatio(Config cfg)
    : cfg_(std::move(cfg))
{}

void NarrativeVolatilityRatio::record(double bias) {
    double new_ratio{};
    double new_sv{};
    double new_lv{};
    bool changed = false;

    {
        std::lock_guard<std::mutex> lk(mutex_);

        double as = cfg_.alpha_short;
        double al = cfg_.alpha_long;

        // Update mean with fast alpha
        mu_ = mu_ * (1.0 - as) + bias * as;
        double dev = bias - mu_;

        // Update variances
        var_s_ = var_s_ * (1.0 - as) + dev * dev * as;
        var_l_ = var_l_ * (1.0 - al) + dev * dev * al;

        new_sv = std::sqrt(var_s_);
        new_lv = std::sqrt(var_l_);
        new_ratio = (new_lv > 1e-12) ? new_sv / new_lv : 1.0;

        uint64_t n = total_.load(std::memory_order_relaxed);
        if (n >= static_cast<uint64_t>(cfg_.min_samples)) {
            if (seeded_ && std::fabs(new_ratio - prev_ratio_) > cfg_.change_threshold) {
                changed = true;
                changes_.fetch_add(1, std::memory_order_relaxed);
            }
        }
        double old_ratio = prev_ratio_;
        prev_ratio_ = new_ratio;
        seeded_     = true;
        (void)old_ratio;
    }

    short_vol_.store(new_sv, std::memory_order_relaxed);
    long_vol_.store(new_lv, std::memory_order_relaxed);
    ratio_.store(new_ratio, std::memory_order_relaxed);
    total_.fetch_add(1, std::memory_order_relaxed);

    if (changed && cfg_.on_vol_regime_change) {
        // Approximate old ratio
        cfg_.on_vol_regime_change(prev_ratio_, new_ratio);
    }
}

std::string NarrativeVolatilityRatio::to_stats_json() const {
    std::lock_guard<std::mutex> lk(mutex_);
    std::ostringstream ss;
    ss << std::fixed << std::setprecision(4);
    ss << "{\"short_vol\":" << short_vol_.load(std::memory_order_relaxed)
       << ",\"long_vol\":"  << long_vol_.load(std::memory_order_relaxed)
       << ",\"vol_ratio\":" << ratio_.load(std::memory_order_relaxed)
       << ",\"regime_changes\":" << changes_.load(std::memory_order_relaxed)
       << ",\"total_records\":" << total_.load(std::memory_order_relaxed)
       << "}";
    return ss.str();
}

void NarrativeVolatilityRatio::reset() {
    std::lock_guard<std::mutex> lk(mutex_);
    mu_          = 0.0;
    var_s_       = 0.0;
    var_l_       = 0.0;
    prev_ratio_  = -1.0;
    seeded_      = false;
    short_vol_.store(0.0, std::memory_order_relaxed);
    long_vol_.store(0.0, std::memory_order_relaxed);
    ratio_.store(1.0, std::memory_order_relaxed);
    changes_.store(0, std::memory_order_relaxed);
    total_.store(0, std::memory_order_relaxed);
}

void NarrativeVolatilityRatio::update_config(const Config& cfg) {
    std::lock_guard<std::mutex> lk(mutex_);
    cfg_ = cfg;
}

} // namespace llmquant
