#include "SignalSlopeMeter.h"

#include <algorithm>
#include <cmath>
#include <sstream>
#include <iomanip>

namespace llmquant {

SignalSlopeMeter::SignalSlopeMeter(Config cfg)
    : cfg_(std::move(cfg)) {
    buf_.resize(static_cast<size_t>(std::max(cfg_.window, 2)), 0.0);
}

void SignalSlopeMeter::record(double bias) {
    double fire_slope = 0.0;
    bool   do_fire    = false;

    {
        std::lock_guard<std::mutex> lk(mutex_);

        obs_count_.fetch_add(1, std::memory_order_relaxed);

        int sz = static_cast<int>(buf_.size());
        buf_[static_cast<size_t>(head_)] = bias;
        head_ = (head_ + 1) % sz;
        if (fill_ < sz) ++fill_;

        double raw_slope = 0.0;
        if (fill_ >= 2) {
            // OLS: x = 0,1,...,n-1; y = circular-buffer values in order.
            int n = fill_;
            double sum_x  = 0.0;
            double sum_y  = 0.0;
            double sum_xy = 0.0;
            double sum_x2 = 0.0;

            // Oldest sample is at index (head_ - fill_ + sz) % sz.
            int start = (head_ - fill_ + sz * 2) % sz;
            for (int i = 0; i < n; ++i) {
                double x = static_cast<double>(i);
                double y = buf_[static_cast<size_t>((start + i) % sz)];
                sum_x  += x;
                sum_y  += y;
                sum_xy += x * y;
                sum_x2 += x * x;
            }
            double denom = static_cast<double>(n) * sum_x2 - sum_x * sum_x;
            if (std::abs(denom) > 1e-12) {
                raw_slope = (static_cast<double>(n) * sum_xy - sum_x * sum_y) / denom;
            }
        }

        slope_.store(raw_slope, std::memory_order_relaxed);

        double sat = std::max(std::abs(cfg_.saturation_slope), 1e-9);
        double score = std::clamp(raw_slope / sat, -1.0, 1.0);
        slope_score_.store(score, std::memory_order_relaxed);

        bool now_accel = (std::abs(raw_slope) >= cfg_.acceleration_threshold);
        accelerating_.store(now_accel, std::memory_order_relaxed);

        if (now_accel && !prev_accelerating_ && cfg_.on_trend_acceleration) {
            fire_slope = raw_slope;
            do_fire    = true;
        }
        prev_accelerating_ = now_accel;
    }

    if (do_fire) cfg_.on_trend_acceleration(fire_slope);
}

std::string SignalSlopeMeter::to_stats_json() const {
    std::lock_guard<std::mutex> lk(mutex_);
    std::ostringstream oss;
    oss << std::fixed << std::setprecision(6);
    oss << "{\"slope\":" << slope_.load(std::memory_order_relaxed)
        << ",\"slope_score\":" << slope_score_.load(std::memory_order_relaxed)
        << ",\"is_accelerating\":" << (accelerating_.load(std::memory_order_relaxed) ? "true" : "false")
        << ",\"obs_count\":" << obs_count_.load(std::memory_order_relaxed)
        << "}";
    return oss.str();
}

void SignalSlopeMeter::reset() {
    std::lock_guard<std::mutex> lk(mutex_);
    std::fill(buf_.begin(), buf_.end(), 0.0);
    head_ = 0;
    fill_ = 0;
    prev_accelerating_ = false;
    slope_.store(0.0, std::memory_order_relaxed);
    slope_score_.store(0.0, std::memory_order_relaxed);
    accelerating_.store(false, std::memory_order_relaxed);
    obs_count_.store(0, std::memory_order_relaxed);
}

void SignalSlopeMeter::update_config(const Config& cfg) {
    std::lock_guard<std::mutex> lk(mutex_);
    cfg_ = cfg;
    buf_.resize(static_cast<size_t>(std::max(cfg_.window, 2)), 0.0);
}

} // namespace llmquant
