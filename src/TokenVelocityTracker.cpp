#include "TokenVelocityTracker.h"

#include <algorithm>
#include <cmath>
#include <sstream>

namespace llmquant {

TokenVelocityTracker::TokenVelocityTracker(Config config)
    : config_(std::move(config)) {
    ring_.resize(static_cast<size_t>(config_.window_size));
    threshold_.store(config_.fast_move_threshold, std::memory_order_relaxed);
    epoch_ = Clock::now();
}

void TokenVelocityTracker::recompute_locked() noexcept {
    int n = ring_fill_;
    if (n < config_.min_samples) {
        velocity_.store(0.0, std::memory_order_relaxed);
        acceleration_.store(0.0, std::memory_order_relaxed);
        prev_velocity_ = 0.0;
        return;
    }

    // Retrieve first and last samples in chronological order.
    int oldest = (ring_head_ - n + config_.window_size) % config_.window_size;
    int newest = (ring_head_ - 1 + config_.window_size) % config_.window_size;

    double t0 = ring_[static_cast<size_t>(oldest)].time_s;
    double b0 = ring_[static_cast<size_t>(oldest)].bias;
    double t1 = ring_[static_cast<size_t>(newest)].time_s;
    double b1 = ring_[static_cast<size_t>(newest)].bias;

    double dt = t1 - t0;
    double vel = (dt > 1e-9) ? (b1 - b0) / dt : 0.0;

    // Acceleration: change in velocity since last recompute.
    // We approximate using the half-window midpoint.
    double accel = vel - prev_velocity_;
    // Note: accel is in vel-units per window; we leave raw since dt varies.

    prev_velocity_ = vel;
    velocity_.store(vel,   std::memory_order_relaxed);
    acceleration_.store(accel, std::memory_order_relaxed);
}

void TokenVelocityTracker::record(double bias) noexcept {
    total_.fetch_add(1, std::memory_order_relaxed);

    bool   fire_cb = false;
    double vel_snap = 0.0, acc_snap = 0.0;
    Config cfg_copy;

    {
        std::lock_guard<std::mutex> lk(mutex_);
        cfg_copy = config_;

        double t = std::chrono::duration<double>(Clock::now() - epoch_).count();
        ring_[static_cast<size_t>(ring_head_)] = {t, bias};
        ring_head_ = (ring_head_ + 1) % config_.window_size;
        if (ring_fill_ < config_.window_size) ++ring_fill_;

        recompute_locked();

        double v = velocity_.load(std::memory_order_relaxed);
        if (std::abs(v) >= config_.fast_move_threshold) {
            fire_cb  = true;
            vel_snap = v;
            acc_snap = acceleration_.load(std::memory_order_relaxed);
        }
    }

    if (fire_cb && cfg_copy.on_fast_move)
        cfg_copy.on_fast_move(vel_snap, acc_snap);
}

bool TokenVelocityTracker::is_fast_move() const noexcept {
    double v = velocity_.load(std::memory_order_relaxed);
    double t = threshold_.load(std::memory_order_relaxed);
    return std::abs(v) >= t;
}

void TokenVelocityTracker::update_config(Config config) {
    std::lock_guard<std::mutex> lk(mutex_);
    config_ = std::move(config);
    ring_.assign(static_cast<size_t>(config_.window_size), {});
    ring_head_     = 0;
    ring_fill_     = 0;
    epoch_         = Clock::now();
    prev_velocity_ = 0.0;
    threshold_.store(config_.fast_move_threshold, std::memory_order_relaxed);
    velocity_.store(0.0,     std::memory_order_relaxed);
    acceleration_.store(0.0, std::memory_order_relaxed);
    total_.store(0,          std::memory_order_relaxed);
}

void TokenVelocityTracker::reset() {
    std::lock_guard<std::mutex> lk(mutex_);
    std::fill(ring_.begin(), ring_.end(), Sample{});
    ring_head_     = 0;
    ring_fill_     = 0;
    epoch_         = Clock::now();
    prev_velocity_ = 0.0;
    velocity_.store(0.0,     std::memory_order_relaxed);
    acceleration_.store(0.0, std::memory_order_relaxed);
    total_.store(0,          std::memory_order_relaxed);
}

std::string TokenVelocityTracker::to_stats_json() const {
    double v  = velocity_.load(std::memory_order_relaxed);
    double a  = acceleration_.load(std::memory_order_relaxed);
    uint64_t t = total_.load(std::memory_order_relaxed);

    std::ostringstream o;
    o << std::fixed;
    o.precision(4);
    o << "{\"velocity\":"      << v
      << ",\"acceleration\":"  << a
      << ",\"is_fast_move\":"  << (is_fast_move() ? "true" : "false")
      << ",\"total_records\":" << t << "}";
    return o.str();
}

} // namespace llmquant
