#include "SignalHurstEstimator.h"

#include <algorithm>
#include <cmath>
#include <iomanip>
#include <numeric>
#include <sstream>

namespace llmquant {

SignalHurstEstimator::SignalHurstEstimator(Config cfg)
    : cfg_(std::move(cfg))
{
    int w = std::max(8, cfg_.window);
    buf_.resize(static_cast<std::size_t>(w), 0.0);
}

void SignalHurstEstimator::record(double bias) {
    std::function<void(double)> trend_cb, revert_cb;
    bool   fire_trend = false, fire_revert = false;
    double h_val = 0.5;

    {
        std::lock_guard<std::mutex> lk(mutex_);
        trend_cb  = cfg_.on_trending;
        revert_cb = cfg_.on_mean_reverting;

        total_.fetch_add(1, std::memory_order_relaxed);

        buf_[static_cast<std::size_t>(head_)] = bias;
        head_ = (head_ + 1) % static_cast<int>(buf_.size());
        if (fill_ < static_cast<int>(buf_.size())) ++fill_;

        int min_obs = std::max(8, cfg_.min_samples);
        if (fill_ >= min_obs) {
            recompute_locked();
            h_val = hurst_.load(std::memory_order_relaxed);

            bool now_trending  = h_val > cfg_.trending_threshold;
            bool now_reverting = h_val < cfg_.reverting_threshold;

            trending_.store(now_trending,  std::memory_order_relaxed);
            reverting_.store(now_reverting, std::memory_order_relaxed);

            if (now_trending && !prev_trending_) {
                fire_trend = true;
                trend_events_.fetch_add(1, std::memory_order_relaxed);
            } else if (!now_trending && prev_trending_ &&
                       h_val < cfg_.trending_threshold * cfg_.clear_hysteresis) {
                // cleared trending — no separate callback needed
            }

            if (now_reverting && !prev_reverting_) {
                fire_revert = true;
                revert_events_.fetch_add(1, std::memory_order_relaxed);
            }

            prev_trending_  = now_trending;
            prev_reverting_ = now_reverting;
        }
    }

    if (fire_trend  && trend_cb)  trend_cb(h_val);
    if (fire_revert && revert_cb) revert_cb(h_val);
}

void SignalHurstEstimator::recompute_locked() {
    int n    = fill_;
    int size = static_cast<int>(buf_.size());

    // Collect window in temporal order.
    double sum = 0.0;
    for (int i = 0; i < n; ++i) {
        int idx = (head_ - n + i + size * 2) % size;
        sum += buf_[static_cast<std::size_t>(idx)];
    }
    double mu = sum / n;

    // Cumulative deviation series and variance.
    double z   = 0.0;
    double zmax = 0.0, zmin = 0.0;
    double var  = 0.0;
    for (int i = 0; i < n; ++i) {
        int idx = (head_ - n + i + size * 2) % size;
        double d = buf_[static_cast<std::size_t>(idx)] - mu;
        z   += d;
        var += d * d;
        if (z > zmax) zmax = z;
        if (z < zmin) zmin = z;
    }
    var /= n;

    double s = std::sqrt(var);
    double h = 0.5;  // default: random walk
    if (s > 1e-12) {
        double rs = (zmax - zmin) / s;
        if (rs > 0.0 && n > 1) {
            h = std::log(rs) / std::log(static_cast<double>(n));
            h = std::max(0.0, std::min(1.0, h));
        }
    }
    hurst_.store(h, std::memory_order_relaxed);
}

std::string SignalHurstEstimator::to_stats_json() const {
    double h;
    bool   trending, reverting;
    uint64_t total, tevents, revents;
    {
        std::lock_guard<std::mutex> lk(mutex_);
        h        = hurst_.load(std::memory_order_relaxed);
        trending  = trending_.load(std::memory_order_relaxed);
        reverting = reverting_.load(std::memory_order_relaxed);
        total    = total_.load(std::memory_order_relaxed);
        tevents  = trend_events_.load(std::memory_order_relaxed);
        revents  = revert_events_.load(std::memory_order_relaxed);
    }
    std::ostringstream ss;
    ss << std::fixed << std::setprecision(6);
    ss << "{"
       << "\"hurst\":"          << h        << ","
       << "\"is_trending\":"    << (trending  ? "true" : "false") << ","
       << "\"is_reverting\":"   << (reverting ? "true" : "false") << ","
       << "\"trend_events\":"   << tevents  << ","
       << "\"revert_events\":"  << revents  << ","
       << "\"total_records\":"  << total
       << "}";
    return ss.str();
}

void SignalHurstEstimator::reset() {
    std::lock_guard<std::mutex> lk(mutex_);
    std::fill(buf_.begin(), buf_.end(), 0.0);
    head_ = fill_ = 0;
    prev_trending_ = prev_reverting_ = false;
    hurst_.store(0.5,  std::memory_order_relaxed);
    trending_.store(false, std::memory_order_relaxed);
    reverting_.store(false, std::memory_order_relaxed);
    total_.store(0, std::memory_order_relaxed);
    trend_events_.store(0, std::memory_order_relaxed);
    revert_events_.store(0, std::memory_order_relaxed);
}

void SignalHurstEstimator::update_config(const Config& cfg) {
    std::lock_guard<std::mutex> lk(mutex_);
    cfg_ = cfg;
    int w = std::max(8, cfg_.window);
    buf_.assign(static_cast<std::size_t>(w), 0.0);
    head_ = fill_ = 0;
    prev_trending_ = prev_reverting_ = false;
    hurst_.store(0.5,  std::memory_order_relaxed);
    trending_.store(false, std::memory_order_relaxed);
    reverting_.store(false, std::memory_order_relaxed);
    total_.store(0, std::memory_order_relaxed);
    trend_events_.store(0, std::memory_order_relaxed);
    revert_events_.store(0, std::memory_order_relaxed);
}

} // namespace llmquant
