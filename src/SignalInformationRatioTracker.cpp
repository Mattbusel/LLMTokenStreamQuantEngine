#include "SignalInformationRatioTracker.h"

#include <algorithm>
#include <cmath>
#include <iomanip>
#include <numeric>
#include <sstream>

namespace llmquant {

SignalInformationRatioTracker::SignalInformationRatioTracker(Config cfg)
    : cfg_(std::move(cfg))
{
    int w = std::max(4, cfg_.window);
    buf_.resize(static_cast<std::size_t>(w), 0.0);
}

void SignalInformationRatioTracker::record(double bias) {
    std::function<void(double)> high_cb, norm_cb;
    bool fire_high = false, fire_norm = false;
    double ir_val = 0.0;

    {
        std::lock_guard<std::mutex> lk(mutex_);
        high_cb = cfg_.on_high_ir;
        norm_cb = cfg_.on_ir_normalized;

        total_.fetch_add(1, std::memory_order_relaxed);

        double excess = bias - cfg_.reference;
        buf_[static_cast<std::size_t>(head_)] = excess;
        head_ = (head_ + 1) % static_cast<int>(buf_.size());
        if (fill_ < static_cast<int>(buf_.size())) ++fill_;

        int min_obs = std::max(2, cfg_.min_samples);
        if (fill_ >= min_obs) {
            recompute_locked();
            ir_val = ir_.load(std::memory_order_relaxed);

            bool now_high = std::abs(ir_val) > cfg_.ir_threshold;
            high_ir_.store(now_high, std::memory_order_relaxed);

            if (now_high && !prev_high_ir_) {
                fire_high = true;
                events_.fetch_add(1, std::memory_order_relaxed);
            } else if (!now_high && prev_high_ir_ &&
                       std::abs(ir_val) < cfg_.ir_threshold * cfg_.clear_hysteresis) {
                fire_norm = true;
            }
            prev_high_ir_ = now_high;
        }
    }

    if (fire_high && high_cb) high_cb(ir_val);
    if (fire_norm && norm_cb) norm_cb(ir_val);
}

void SignalInformationRatioTracker::recompute_locked() {
    int n    = fill_;
    int size = static_cast<int>(buf_.size());

    double sum = 0.0;
    for (int i = 0; i < n; ++i) {
        int idx = (head_ - n + i + size * 2) % size;
        sum += buf_[static_cast<std::size_t>(idx)];
    }
    double mu = sum / n;

    double var = 0.0;
    for (int i = 0; i < n; ++i) {
        int idx = (head_ - n + i + size * 2) % size;
        double d = buf_[static_cast<std::size_t>(idx)] - mu;
        var += d * d;
    }
    var /= n;
    double sigma = std::sqrt(var);

    double ir = 0.0;
    if (sigma > 1e-12) {
        ir = mu / sigma;
        // Clamp to avoid extreme values.
        ir = std::max(-10.0, std::min(10.0, ir));
    }

    ir_.store(ir,     std::memory_order_relaxed);
    mean_.store(mu,   std::memory_order_relaxed);
    stddev_.store(sigma, std::memory_order_relaxed);
}

std::string SignalInformationRatioTracker::to_stats_json() const {
    double ir, mu, sigma;
    bool   high;
    uint64_t events, total;
    {
        std::lock_guard<std::mutex> lk(mutex_);
        ir     = ir_.load(std::memory_order_relaxed);
        mu     = mean_.load(std::memory_order_relaxed);
        sigma  = stddev_.load(std::memory_order_relaxed);
        high   = high_ir_.load(std::memory_order_relaxed);
        events = events_.load(std::memory_order_relaxed);
        total  = total_.load(std::memory_order_relaxed);
    }
    std::ostringstream ss;
    ss << std::fixed << std::setprecision(6);
    ss << "{"
       << "\"ir\":"            << ir    << ","
       << "\"mean\":"          << mu    << ","
       << "\"stddev\":"        << sigma << ","
       << "\"is_high_ir\":"    << (high ? "true" : "false") << ","
       << "\"high_ir_events\":" << events << ","
       << "\"total_records\":" << total
       << "}";
    return ss.str();
}

void SignalInformationRatioTracker::reset() {
    std::lock_guard<std::mutex> lk(mutex_);
    std::fill(buf_.begin(), buf_.end(), 0.0);
    head_ = fill_ = 0;
    prev_high_ir_ = false;
    ir_.store(0.0,     std::memory_order_relaxed);
    mean_.store(0.0,   std::memory_order_relaxed);
    stddev_.store(0.0, std::memory_order_relaxed);
    high_ir_.store(false, std::memory_order_relaxed);
    events_.store(0,   std::memory_order_relaxed);
    total_.store(0,    std::memory_order_relaxed);
}

void SignalInformationRatioTracker::update_config(const Config& cfg) {
    std::lock_guard<std::mutex> lk(mutex_);
    cfg_ = cfg;
    int w = std::max(4, cfg_.window);
    buf_.assign(static_cast<std::size_t>(w), 0.0);
    head_ = fill_ = 0;
    prev_high_ir_ = false;
    ir_.store(0.0,     std::memory_order_relaxed);
    mean_.store(0.0,   std::memory_order_relaxed);
    stddev_.store(0.0, std::memory_order_relaxed);
    high_ir_.store(false, std::memory_order_relaxed);
    events_.store(0,   std::memory_order_relaxed);
    total_.store(0,    std::memory_order_relaxed);
}

} // namespace llmquant
