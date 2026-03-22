#include "TokenClockRecalibrator.h"

#include <algorithm>
#include <cmath>
#include <numeric>
#include <sstream>
#include <iomanip>

namespace llmquant {

TokenClockRecalibrator::TokenClockRecalibrator(Config cfg)
    : cfg_(std::move(cfg))
{
    gaps_.resize(static_cast<std::size_t>(std::max(2, cfg_.window_size)), 0.0);
}

void TokenClockRecalibrator::record_token(uint64_t timestamp_ns) {
    std::function<void(double, double)> cb;
    bool   fire       = false;
    double new_rate   = 0.0;
    double new_scale  = 1.0;

    {
        std::lock_guard<std::mutex> lk(mutex_);
        cb = cfg_.on_rate_change;
        token_count_.fetch_add(1, std::memory_order_relaxed);

        if (!seeded_) {
            last_ts_ = timestamp_ns;
            seeded_  = true;
            return;
        }

        if (timestamp_ns <= last_ts_) {
            last_ts_ = timestamp_ns;
            return;
        }

        double gap_s = static_cast<double>(timestamp_ns - last_ts_) * 1e-9;
        last_ts_ = timestamp_ns;

        gaps_[static_cast<std::size_t>(head_)] = gap_s;
        head_ = (head_ + 1) % static_cast<int>(gaps_.size());
        if (fill_ < static_cast<int>(gaps_.size())) ++fill_;

        recompute_locked();

        new_rate  = rate_hz_.load(std::memory_order_relaxed);
        new_scale = budget_scale_.load(std::memory_order_relaxed);

        if (prev_rate_hz_ > 0.0) {
            double frac_change = std::abs(new_rate - prev_rate_hz_) / prev_rate_hz_;
            if (frac_change > cfg_.rate_change_threshold) {
                fire = true;
                rate_change_count_.fetch_add(1, std::memory_order_relaxed);
                prev_rate_hz_ = new_rate;
            }
        } else if (new_rate > 0.0) {
            prev_rate_hz_ = new_rate;
        }
    }

    if (fire && cb) cb(new_rate, new_scale);
}

void TokenClockRecalibrator::recompute_locked() {
    if (fill_ == 0) return;

    // Compute mean and std of the filled portion of the ring buffer.
    double sum = 0.0;
    for (int i = 0; i < fill_; ++i)
        sum += gaps_[static_cast<std::size_t>(i)];
    double mean = sum / fill_;
    if (mean <= 0.0) return;

    double var = 0.0;
    for (int i = 0; i < fill_; ++i) {
        double d = gaps_[static_cast<std::size_t>(i)] - mean;
        var += d * d;
    }
    var /= fill_;
    double std_dev = std::sqrt(var);

    double rate   = 1.0 / mean;
    double cv     = (mean > 0.0) ? (std_dev / mean) : 0.0;
    double target = cfg_.target_rate_hz;
    double scale  = (rate > 0.0) ? (target / rate) : 1.0;
    scale = std::clamp(scale, cfg_.min_scale, cfg_.max_scale);

    rate_hz_.store(rate, std::memory_order_relaxed);
    budget_scale_.store(scale, std::memory_order_relaxed);
    jitter_cv_.store(cv, std::memory_order_relaxed);
}

std::string TokenClockRecalibrator::to_stats_json() const {
    double rate, scale, cv;
    uint64_t tc, rcc;
    {
        std::lock_guard<std::mutex> lk(mutex_);
        rate  = rate_hz_.load(std::memory_order_relaxed);
        scale = budget_scale_.load(std::memory_order_relaxed);
        cv    = jitter_cv_.load(std::memory_order_relaxed);
        tc    = token_count_.load(std::memory_order_relaxed);
        rcc   = rate_change_count_.load(std::memory_order_relaxed);
    }
    std::ostringstream ss;
    ss << std::fixed << std::setprecision(6);
    ss << "{"
       << "\"rate_hz\":"            << rate  << ","
       << "\"budget_scale\":"       << scale << ","
       << "\"jitter_cv\":"          << cv    << ","
       << "\"token_count\":"        << tc    << ","
       << "\"rate_change_count\":"  << rcc
       << "}";
    return ss.str();
}

void TokenClockRecalibrator::reset() {
    std::lock_guard<std::mutex> lk(mutex_);
    std::fill(gaps_.begin(), gaps_.end(), 0.0);
    head_ = fill_ = 0;
    last_ts_ = 0;
    seeded_  = false;
    prev_rate_hz_ = 0.0;
    rate_hz_.store(0.0, std::memory_order_relaxed);
    budget_scale_.store(1.0, std::memory_order_relaxed);
    jitter_cv_.store(0.0, std::memory_order_relaxed);
    token_count_.store(0, std::memory_order_relaxed);
    rate_change_count_.store(0, std::memory_order_relaxed);
}

void TokenClockRecalibrator::update_config(const Config& cfg) {
    std::lock_guard<std::mutex> lk(mutex_);
    cfg_ = cfg;
    gaps_.assign(static_cast<std::size_t>(std::max(2, cfg_.window_size)), 0.0);
    head_ = fill_ = 0;
    last_ts_ = 0;
    seeded_  = false;
    prev_rate_hz_ = 0.0;
}

} // namespace llmquant
