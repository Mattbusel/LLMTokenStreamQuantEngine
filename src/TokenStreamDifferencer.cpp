#include "TokenStreamDifferencer.h"

#include <cmath>
#include <sstream>
#include <iomanip>

namespace llmquant {

TokenStreamDifferencer::TokenStreamDifferencer(Config cfg)
    : cfg_(std::move(cfg))
{}

void TokenStreamDifferencer::record(double weight) {
    std::function<void(double, double)> cb;
    bool   fire      = false;
    double rj = 0.0, ej = 0.0;

    {
        std::lock_guard<std::mutex> lk(mutex_);
        cb = cfg_.on_jerk_spike;
        obs_count_.fetch_add(1, std::memory_order_relaxed);

        if (fill_ == 0) {
            w_prev_ = weight;
            fill_   = 1;
            return;
        }

        double d1 = weight - w_prev_;
        w_prev_   = weight;
        vel_ema_val_ += cfg_.ema_alpha * (d1 - vel_ema_val_);
        velocity_.store(d1, std::memory_order_relaxed);
        velocity_ema_.store(vel_ema_val_, std::memory_order_relaxed);

        if (fill_ == 1) {
            d1_prev_ = d1;
            fill_    = 2;
            return;
        }

        double d2 = d1 - d1_prev_;
        d1_prev_  = d1;
        acc_ema_val_ += cfg_.ema_alpha * (d2 - acc_ema_val_);
        acceleration_.store(d2, std::memory_order_relaxed);
        acceleration_ema_.store(acc_ema_val_, std::memory_order_relaxed);

        if (fill_ == 2) {
            d2_prev_ = d2;
            fill_    = 3;
            return;
        }

        double d3 = d2 - d2_prev_;
        d2_prev_  = d2;
        jerk_ema_val_ += cfg_.ema_alpha * (d3 - jerk_ema_val_);
        jerk_.store(d3, std::memory_order_relaxed);
        jerk_ema_.store(jerk_ema_val_, std::memory_order_relaxed);

        // Jerk spike detection with hysteresis.
        double threshold = cfg_.jerk_threshold;
        if (armed_ && std::abs(d3) > threshold) {
            armed_ = false;
            fire   = true;
            rj     = d3;
            ej     = jerk_ema_val_;
            spike_count_.fetch_add(1, std::memory_order_relaxed);
        } else if (!armed_ && std::abs(jerk_ema_val_) < threshold * (1.0 - cfg_.hysteresis)) {
            armed_ = true;
        }
    }

    if (fire && cb) cb(rj, ej);
}

std::string TokenStreamDifferencer::to_stats_json() const {
    double v, ve, a, ae, j, je;
    uint64_t oc, sc;
    {
        std::lock_guard<std::mutex> lk(mutex_);
        v  = velocity_.load(std::memory_order_relaxed);
        ve = velocity_ema_.load(std::memory_order_relaxed);
        a  = acceleration_.load(std::memory_order_relaxed);
        ae = acceleration_ema_.load(std::memory_order_relaxed);
        j  = jerk_.load(std::memory_order_relaxed);
        je = jerk_ema_.load(std::memory_order_relaxed);
        oc = obs_count_.load(std::memory_order_relaxed);
        sc = spike_count_.load(std::memory_order_relaxed);
    }
    std::ostringstream ss;
    ss << std::fixed << std::setprecision(6);
    ss << "{"
       << "\"velocity\":"          << v  << ","
       << "\"velocity_ema\":"      << ve << ","
       << "\"acceleration\":"      << a  << ","
       << "\"acceleration_ema\":"  << ae << ","
       << "\"jerk\":"              << j  << ","
       << "\"jerk_ema\":"          << je << ","
       << "\"observation_count\":" << oc << ","
       << "\"jerk_spike_count\":"  << sc
       << "}";
    return ss.str();
}

void TokenStreamDifferencer::reset() {
    std::lock_guard<std::mutex> lk(mutex_);
    w_prev_ = d1_prev_ = d2_prev_ = 0.0;
    vel_ema_val_ = acc_ema_val_ = jerk_ema_val_ = 0.0;
    fill_  = 0;
    armed_ = true;
    velocity_.store(0.0, std::memory_order_relaxed);
    velocity_ema_.store(0.0, std::memory_order_relaxed);
    acceleration_.store(0.0, std::memory_order_relaxed);
    acceleration_ema_.store(0.0, std::memory_order_relaxed);
    jerk_.store(0.0, std::memory_order_relaxed);
    jerk_ema_.store(0.0, std::memory_order_relaxed);
    obs_count_.store(0, std::memory_order_relaxed);
    spike_count_.store(0, std::memory_order_relaxed);
}

void TokenStreamDifferencer::update_config(const Config& cfg) {
    std::lock_guard<std::mutex> lk(mutex_);
    cfg_   = cfg;
    fill_  = 0;
    armed_ = true;
    w_prev_ = d1_prev_ = d2_prev_ = 0.0;
    vel_ema_val_ = acc_ema_val_ = jerk_ema_val_ = 0.0;
}

} // namespace llmquant
