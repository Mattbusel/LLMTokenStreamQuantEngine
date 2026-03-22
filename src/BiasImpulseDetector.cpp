#include "BiasImpulseDetector.h"
#include <cmath>
#include <sstream>
#include <iomanip>

namespace llmquant {

BiasImpulseDetector::BiasImpulseDetector(Config cfg)
    : cfg_(std::move(cfg))
    , win_buf_(cfg_.window, 0.0)
{}

void BiasImpulseDetector::record(double bias) {
    double new_z{};
    double new_mean{};
    double new_std{};
    bool impulse = false;

    {
        std::lock_guard<std::mutex> lk(mutex_);
        win_buf_[win_head_] = bias;
        win_head_ = (win_head_ + 1) % cfg_.window;
        if (win_fill_ < cfg_.window) ++win_fill_;

        if (win_fill_ >= cfg_.min_samples) {
            int n = win_fill_;
            double sum = 0.0;
            for (int i = 0; i < n; ++i) sum += win_buf_[i];
            new_mean = sum / n;

            double var = 0.0;
            for (int i = 0; i < n; ++i) {
                double d = win_buf_[i] - new_mean;
                var += d * d;
            }
            new_std = std::sqrt(var / n);

            new_z = (new_std > 1e-10) ? (bias - new_mean) / new_std : 0.0;

            if (std::fabs(new_z) > max_z_val_) {
                max_z_val_ = std::fabs(new_z);
                max_z_.store(max_z_val_, std::memory_order_relaxed);
            }

            if (std::fabs(new_z) > cfg_.z_threshold) {
                impulse = true;
                impulse_events_.fetch_add(1, std::memory_order_relaxed);
            }
        }
    }

    last_z_.store(new_z, std::memory_order_relaxed);
    mean_.store(new_mean, std::memory_order_relaxed);
    std_.store(new_std, std::memory_order_relaxed);
    total_.fetch_add(1, std::memory_order_relaxed);

    if (impulse && cfg_.on_impulse) cfg_.on_impulse(new_z, bias);
}

std::string BiasImpulseDetector::to_stats_json() const {
    std::lock_guard<std::mutex> lk(mutex_);
    std::ostringstream ss;
    ss << std::fixed << std::setprecision(4);
    ss << "{\"last_z\":"     << last_z_.load(std::memory_order_relaxed)
       << ",\"rolling_mean\":" << mean_.load(std::memory_order_relaxed)
       << ",\"rolling_std\":" << std_.load(std::memory_order_relaxed)
       << ",\"max_z\":"      << max_z_.load(std::memory_order_relaxed)
       << ",\"impulse_events\":" << impulse_events_.load(std::memory_order_relaxed)
       << ",\"total_records\":" << total_.load(std::memory_order_relaxed)
       << "}";
    return ss.str();
}

void BiasImpulseDetector::reset() {
    std::lock_guard<std::mutex> lk(mutex_);
    std::fill(win_buf_.begin(), win_buf_.end(), 0.0);
    win_head_  = 0; win_fill_ = 0;
    max_z_val_ = 0.0;
    last_z_.store(0.0, std::memory_order_relaxed);
    mean_.store(0.0, std::memory_order_relaxed);
    std_.store(0.0, std::memory_order_relaxed);
    max_z_.store(0.0, std::memory_order_relaxed);
    impulse_events_.store(0, std::memory_order_relaxed);
    total_.store(0, std::memory_order_relaxed);
}

void BiasImpulseDetector::update_config(const Config& cfg) {
    std::lock_guard<std::mutex> lk(mutex_);
    cfg_ = cfg;
    win_buf_.assign(cfg_.window, 0.0);
    win_head_ = 0; win_fill_ = 0;
}

} // namespace llmquant
