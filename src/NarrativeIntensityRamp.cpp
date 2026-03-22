#include "NarrativeIntensityRamp.h"

#include <algorithm>
#include <cmath>
#include <sstream>
#include <iomanip>

namespace llmquant {

NarrativeIntensityRamp::NarrativeIntensityRamp(Config cfg)
    : cfg_(std::move(cfg)) {}

void NarrativeIntensityRamp::record(double bias) {
    double fire_surge = 0.0, fire_fade = 0.0;
    bool   do_surge = false, do_fade = false;

    {
        std::lock_guard<std::mutex> lk(mutex_);

        obs_count_.fetch_add(1, std::memory_order_relaxed);

        double abs_bias = std::abs(bias);
        if (!seeded_) {
            ema_       = abs_bias;
            prev_ema_  = abs_bias;
            seeded_    = true;
        } else {
            prev_ema_ = ema_;
            ema_      = cfg_.ema_alpha * abs_bias + (1.0 - cfg_.ema_alpha) * ema_;
        }

        intensity_.store(ema_, std::memory_order_relaxed);

        double r = ema_ - prev_ema_;
        ramp_.store(r, std::memory_order_relaxed);

        double sat = std::max(std::abs(cfg_.saturation_ramp), 1e-9);
        double rs  = std::clamp(r / sat, -1.0, 1.0);
        ramp_score_.store(rs, std::memory_order_relaxed);

        bool now_surging = (r >=  cfg_.surge_threshold);
        bool now_fading  = (r <= -cfg_.surge_threshold);
        surging_.store(now_surging, std::memory_order_relaxed);
        fading_.store(now_fading,   std::memory_order_relaxed);

        if (now_surging && !prev_surging_ && cfg_.on_surge) {
            fire_surge = r; do_surge = true;
        }
        if (now_fading && !prev_fading_ && cfg_.on_fade) {
            fire_fade  = r; do_fade  = true;
        }
        prev_surging_ = now_surging;
        prev_fading_  = now_fading;
    }

    if (do_surge) cfg_.on_surge(fire_surge);
    if (do_fade)  cfg_.on_fade(fire_fade);
}

std::string NarrativeIntensityRamp::to_stats_json() const {
    std::lock_guard<std::mutex> lk(mutex_);
    std::ostringstream oss;
    oss << std::fixed << std::setprecision(4);
    oss << "{\"intensity\":" << intensity_.load(std::memory_order_relaxed)
        << ",\"ramp\":" << ramp_.load(std::memory_order_relaxed)
        << ",\"ramp_score\":" << ramp_score_.load(std::memory_order_relaxed)
        << ",\"is_surging\":" << (surging_.load(std::memory_order_relaxed) ? "true" : "false")
        << ",\"is_fading\":" << (fading_.load(std::memory_order_relaxed) ? "true" : "false")
        << ",\"obs_count\":" << obs_count_.load(std::memory_order_relaxed)
        << "}";
    return oss.str();
}

void NarrativeIntensityRamp::reset() {
    std::lock_guard<std::mutex> lk(mutex_);
    ema_ = 0.0; prev_ema_ = 0.0; seeded_ = false;
    prev_surging_ = false; prev_fading_ = false;
    intensity_.store(0.0, std::memory_order_relaxed);
    ramp_.store(0.0, std::memory_order_relaxed);
    ramp_score_.store(0.0, std::memory_order_relaxed);
    surging_.store(false, std::memory_order_relaxed);
    fading_.store(false, std::memory_order_relaxed);
    obs_count_.store(0, std::memory_order_relaxed);
}

void NarrativeIntensityRamp::update_config(const Config& cfg) {
    std::lock_guard<std::mutex> lk(mutex_);
    cfg_ = cfg;
}

} // namespace llmquant
