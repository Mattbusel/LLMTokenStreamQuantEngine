#include "AdaptiveVelocityBreaker.h"

#include <cmath>
#include <cstdio>

namespace llmquant {

bool AdaptiveVelocityBreaker::record(double bias) noexcept {
    TripCallback     trip_cb;
    RecoveryCallback recovery_cb;
    bool             fire_trip     = false;
    bool             fire_recovery = false;
    double           new_vel       = 0.0;

    {
        std::lock_guard<std::mutex> lk(mtx_);

        auto now = Clock::now();

        if (!seeded_) {
            prev_bias_ = bias;
            prev_time_ = now;
            seeded_    = true;
            return false;
        }

        double dt_s = std::chrono::duration<double>(now - prev_time_).count();
        if (dt_s > 0.0 && dt_s >= cfg_.min_dt_s) {
            double instant_vel = std::abs(bias - prev_bias_) / dt_s;
            double cur_ema     = velocity_ema_.load(std::memory_order_relaxed);
            new_vel = cur_ema + cfg_.velocity_alpha * (instant_vel - cur_ema);
            velocity_ema_.store(new_vel, std::memory_order_relaxed);

            prev_bias_ = bias;
            prev_time_ = now;
        } else {
            new_vel = velocity_ema_.load(std::memory_order_relaxed);
        }

        bool currently_open = open_.load(std::memory_order_relaxed);

        if (!currently_open && new_vel >= cfg_.trip_threshold) {
            open_.store(true, std::memory_order_release);
            trip_count_.fetch_add(1, std::memory_order_relaxed);
            fire_trip = true;
            trip_cb   = cfg_.on_trip;
        } else if (currently_open &&
                   new_vel < cfg_.trip_threshold * cfg_.recovery_factor) {
            open_.store(false, std::memory_order_release);
            recovery_count_.fetch_add(1, std::memory_order_relaxed);
            fire_recovery = true;
            recovery_cb   = cfg_.on_recovery;
        }
    }

    if (fire_trip     && trip_cb)     trip_cb(new_vel);
    if (fire_recovery && recovery_cb) recovery_cb(new_vel);

    return open_.load(std::memory_order_acquire);
}

void AdaptiveVelocityBreaker::reset() noexcept {
    std::lock_guard<std::mutex> lk(mtx_);
    seeded_ = false;
    velocity_ema_.store(0.0, std::memory_order_relaxed);
    open_.store(false, std::memory_order_release);
}

void AdaptiveVelocityBreaker::update_config(const Config& cfg) {
    std::lock_guard<std::mutex> lk(mtx_);
    cfg_    = cfg;
    seeded_ = false;
    velocity_ema_.store(0.0, std::memory_order_relaxed);
    open_.store(false, std::memory_order_release);
}

std::string AdaptiveVelocityBreaker::to_stats_json() const {
    double vel  = velocity_ema_.load(std::memory_order_relaxed);
    bool   open = open_.load(std::memory_order_acquire);
    double thr  = 0.0;
    {
        std::lock_guard<std::mutex> lk(mtx_);
        thr = cfg_.trip_threshold;
    }
    char buf[256];
    std::snprintf(buf, sizeof(buf),
        "{\"is_open\":%s,\"velocity_ema\":%.6f,\"trip_threshold\":%.3f,"
        "\"trip_count\":%llu,\"recovery_count\":%llu}",
        open ? "true" : "false", vel, thr,
        static_cast<unsigned long long>(trip_count_.load(std::memory_order_relaxed)),
        static_cast<unsigned long long>(recovery_count_.load(std::memory_order_relaxed)));
    return buf;
}

} // namespace llmquant
