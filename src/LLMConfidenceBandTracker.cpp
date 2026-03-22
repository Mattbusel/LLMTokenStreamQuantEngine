#include "LLMConfidenceBandTracker.h"

#include <cmath>
#include <sstream>
#include <iomanip>

namespace llmquant {

LLMConfidenceBandTracker::LLMConfidenceBandTracker(Config cfg)
    : cfg_(std::move(cfg))
    , P_(cfg_.initial_variance)
{}

void LLMConfidenceBandTracker::update(double bias) {
    std::function<void(double, double, double)> cb;
    bool   fire  = false;
    double lo = 0.0, ctr = 0.0, hi = 0.0;

    {
        std::lock_guard<std::mutex> lk(mutex_);
        cb = cfg_.on_band_change;
        obs_count_.fetch_add(1, std::memory_order_relaxed);

        if (!seeded_) {
            x_      = bias;
            seeded_ = true;
        }

        // Kalman predict.
        double P_pred = P_ + cfg_.process_noise;

        // Kalman update.
        double K = P_pred / (P_pred + cfg_.measurement_noise);
        x_ = x_ + K * (bias - x_);
        P_ = (1.0 - K) * P_pred;

        double hw = cfg_.z_score * std::sqrt(P_);
        ctr = x_;
        lo  = x_ - hw;
        hi  = x_ + hw;

        center_.store(ctr, std::memory_order_relaxed);
        lower_.store(lo,   std::memory_order_relaxed);
        upper_.store(hi,   std::memory_order_relaxed);
        half_width_.store(hw, std::memory_order_relaxed);
        variance_.store(P_,   std::memory_order_relaxed);

        if (prev_half_width_ >= 0.0 && prev_half_width_ > 0.0) {
            double frac = std::abs(hw - prev_half_width_) / prev_half_width_;
            if (frac > cfg_.band_change_threshold) {
                fire = true;
                band_changes_.fetch_add(1, std::memory_order_relaxed);
            }
        }
        prev_half_width_ = hw;
    }

    if (fire && cb) cb(lo, ctr, hi);
}

std::string LLMConfidenceBandTracker::to_stats_json() const {
    double lo, ctr, hi, hw, var;
    uint64_t oc, bc;
    {
        std::lock_guard<std::mutex> lk(mutex_);
        lo  = lower_.load(std::memory_order_relaxed);
        ctr = center_.load(std::memory_order_relaxed);
        hi  = upper_.load(std::memory_order_relaxed);
        hw  = half_width_.load(std::memory_order_relaxed);
        var = variance_.load(std::memory_order_relaxed);
        oc  = obs_count_.load(std::memory_order_relaxed);
        bc  = band_changes_.load(std::memory_order_relaxed);
    }
    std::ostringstream ss;
    ss << std::fixed << std::setprecision(6);
    ss << "{"
       << "\"lower\":"             << lo  << ","
       << "\"center\":"            << ctr << ","
       << "\"upper\":"             << hi  << ","
       << "\"half_width\":"        << hw  << ","
       << "\"variance\":"          << var << ","
       << "\"observation_count\":" << oc  << ","
       << "\"band_change_count\":" << bc
       << "}";
    return ss.str();
}

void LLMConfidenceBandTracker::reset() {
    std::lock_guard<std::mutex> lk(mutex_);
    x_               = 0.0;
    P_               = cfg_.initial_variance;
    seeded_          = false;
    prev_half_width_ = -1.0;
    center_.store(0.0, std::memory_order_relaxed);
    lower_.store(0.0, std::memory_order_relaxed);
    upper_.store(0.0, std::memory_order_relaxed);
    half_width_.store(0.0, std::memory_order_relaxed);
    variance_.store(cfg_.initial_variance, std::memory_order_relaxed);
    obs_count_.store(0, std::memory_order_relaxed);
    band_changes_.store(0, std::memory_order_relaxed);
}

void LLMConfidenceBandTracker::update_config(const Config& cfg) {
    std::lock_guard<std::mutex> lk(mutex_);
    cfg_             = cfg;
    x_               = 0.0;
    P_               = cfg_.initial_variance;
    seeded_          = false;
    prev_half_width_ = -1.0;
}

} // namespace llmquant
