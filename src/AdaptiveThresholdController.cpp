#include "AdaptiveThresholdController.h"

#include <algorithm>
#include <cmath>
#include <sstream>
#include <iomanip>

namespace llmquant {

AdaptiveThresholdController::AdaptiveThresholdController(Config cfg)
    : cfg_(std::move(cfg))
{
    ob_.store(cfg_.base_overbought, std::memory_order_relaxed);
    os_.store(cfg_.base_oversold,   std::memory_order_relaxed);
}

void AdaptiveThresholdController::record(double bias) {
    bool   fire   = false;
    double new_ob = 0.0, new_os = 0.0;
    std::function<void(double, double)> cb;

    {
        std::lock_guard<std::mutex> lk(mutex_);
        cb = cfg_.on_threshold_change;
        total_.fetch_add(1, std::memory_order_relaxed);

        if (!seeded_) {
            prev_bias_ = bias;
            seeded_    = true;
            return;
        }

        // EMA of |Δbias|
        double alpha     = 1.0 / static_cast<double>(std::max(1, cfg_.period));
        double abs_delta = std::abs(bias - prev_bias_);
        sigma_ema_       = sigma_ema_ + alpha * (abs_delta - sigma_ema_);
        sigma_.store(sigma_ema_, std::memory_order_relaxed);
        prev_bias_ = bias;

        // Adjusted thresholds — symmetric clamp strategy:
        //   ob ∈ [clamp_lo, clamp_hi]
        //   os ∈ [100 - clamp_hi, 100 - clamp_lo]
        double adj = cfg_.k_sigma * sigma_ema_;
        double ob  = std::clamp(cfg_.base_overbought + adj, cfg_.clamp_lo, cfg_.clamp_hi);
        double os  = std::clamp(cfg_.base_oversold   - adj,
                                100.0 - cfg_.clamp_hi,
                                100.0 - cfg_.clamp_lo);

        ob_.store(ob, std::memory_order_relaxed);
        os_.store(os, std::memory_order_relaxed);

        bool ob_changed = (prev_ob_ < 0.0) || (std::abs(ob - prev_ob_) > cfg_.change_threshold);
        bool os_changed = (prev_os_ < 0.0) || (std::abs(os - prev_os_) > cfg_.change_threshold);

        if (ob_changed || os_changed) {
            new_ob = ob;
            new_os = os;
            fire   = true;
            prev_ob_ = ob;
            prev_os_ = os;
            change_events_.fetch_add(1, std::memory_order_relaxed);
        }
    }

    if (fire && cb) cb(new_ob, new_os);
}

std::string AdaptiveThresholdController::to_stats_json() const {
    std::lock_guard<std::mutex> lk(mutex_);
    std::ostringstream ss;
    ss << std::fixed << std::setprecision(6);
    ss << "{"
       << "\"overbought_threshold\":" << ob_.load(std::memory_order_relaxed) << ","
       << "\"oversold_threshold\":"   << os_.load(std::memory_order_relaxed) << ","
       << "\"rolling_sigma\":"        << sigma_.load(std::memory_order_relaxed) << ","
       << "\"change_events\":"        << change_events_.load(std::memory_order_relaxed) << ","
       << "\"total_records\":"        << total_.load(std::memory_order_relaxed)
       << "}";
    return ss.str();
}

void AdaptiveThresholdController::reset() {
    std::lock_guard<std::mutex> lk(mutex_);
    sigma_ema_ = 0.0;
    prev_bias_ = 0.0;
    seeded_    = false;
    prev_ob_   = -1.0;
    prev_os_   = -1.0;
    ob_.store(cfg_.base_overbought, std::memory_order_relaxed);
    os_.store(cfg_.base_oversold,   std::memory_order_relaxed);
    sigma_.store(0.0, std::memory_order_relaxed);
    change_events_.store(0, std::memory_order_relaxed);
    total_.store(0, std::memory_order_relaxed);
}

void AdaptiveThresholdController::update_config(const Config& cfg) {
    std::lock_guard<std::mutex> lk(mutex_);
    cfg_       = cfg;
    sigma_ema_ = 0.0;
    prev_bias_ = 0.0;
    seeded_    = false;
    prev_ob_   = -1.0;
    prev_os_   = -1.0;
    ob_.store(cfg_.base_overbought, std::memory_order_relaxed);
    os_.store(cfg_.base_oversold,   std::memory_order_relaxed);
}

} // namespace llmquant
