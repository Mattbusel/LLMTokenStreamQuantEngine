#include "BiasVelocityBreaker.h"

#include <cmath>
#include <iomanip>
#include <sstream>

namespace llmquant {

BiasVelocityBreaker::BiasVelocityBreaker(Config cfg)
    : cfg_(std::move(cfg))
{}

bool BiasVelocityBreaker::record(double bias) {
    std::function<void(double)> trip_cb, clear_cb;
    bool fire_trip = false, fire_clear = false;
    double vel_val = 0.0;
    bool tripped_now = false;

    {
        std::lock_guard<std::mutex> lk(mutex_);
        trip_cb  = cfg_.on_trip;
        clear_cb = cfg_.on_clear;

        total_.fetch_add(1, std::memory_order_relaxed);

        if (!seeded_) {
            prev_bias_ = bias;
            seeded_    = true;
        } else {
            double delta = std::abs(bias - prev_bias_);
            velocity_ema_ = cfg_.ema_alpha * delta + (1.0 - cfg_.ema_alpha) * velocity_ema_;
            prev_bias_ = bias;
        }

        vel_val = velocity_ema_;
        velocity_.store(vel_val, std::memory_order_relaxed);

        // Hysteresis state machine: trip on exceed, clear only after dropping
        // below max_velocity * clear_hysteresis.
        bool now_tripped;
        if (prev_tripped_) {
            now_tripped = vel_val >= cfg_.max_velocity * cfg_.clear_hysteresis;
        } else {
            now_tripped = vel_val > cfg_.max_velocity;
        }
        tripped_.store(now_tripped, std::memory_order_relaxed);
        tripped_now = now_tripped;

        if (now_tripped && !prev_tripped_) {
            fire_trip = true;
            trip_count_.fetch_add(1, std::memory_order_relaxed);
        } else if (!now_tripped && prev_tripped_) {
            fire_clear = true;
        }
        prev_tripped_ = now_tripped;
    }

    if (fire_trip  && trip_cb)  trip_cb(vel_val);
    if (fire_clear && clear_cb) clear_cb(vel_val);
    return tripped_now;
}

std::string BiasVelocityBreaker::to_stats_json() const {
    double vel;
    bool   tripped;
    uint64_t trips, total;
    {
        std::lock_guard<std::mutex> lk(mutex_);
        vel     = velocity_.load(std::memory_order_relaxed);
        tripped = tripped_.load(std::memory_order_relaxed);
        trips   = trip_count_.load(std::memory_order_relaxed);
        total   = total_.load(std::memory_order_relaxed);
    }
    std::ostringstream ss;
    ss << std::fixed << std::setprecision(6);
    ss << "{"
       << "\"velocity\":"      << vel     << ","
       << "\"is_tripped\":"    << (tripped ? "true" : "false") << ","
       << "\"trip_count\":"    << trips   << ","
       << "\"total_records\":" << total
       << "}";
    return ss.str();
}

void BiasVelocityBreaker::reset() {
    std::lock_guard<std::mutex> lk(mutex_);
    prev_bias_    = 0.0;
    velocity_ema_ = 0.0;
    seeded_       = false;
    prev_tripped_ = false;
    velocity_.store(0.0,   std::memory_order_relaxed);
    tripped_.store(false,  std::memory_order_relaxed);
    trip_count_.store(0,   std::memory_order_relaxed);
    total_.store(0,        std::memory_order_relaxed);
}

void BiasVelocityBreaker::update_config(const Config& cfg) {
    std::lock_guard<std::mutex> lk(mutex_);
    cfg_          = cfg;
    prev_bias_    = 0.0;
    velocity_ema_ = 0.0;
    seeded_       = false;
    prev_tripped_ = false;
    velocity_.store(0.0,   std::memory_order_relaxed);
    tripped_.store(false,  std::memory_order_relaxed);
    trip_count_.store(0,   std::memory_order_relaxed);
    total_.store(0,        std::memory_order_relaxed);
}

} // namespace llmquant
