#include "SignalAccelerationMeter.h"
#include <cmath>
#include <sstream>
#include <iomanip>

namespace llmquant {

SignalAccelerationMeter::SignalAccelerationMeter(Config cfg)
    : cfg_(std::move(cfg))
{}

void SignalAccelerationMeter::record(double bias) {
    double new_vel  = 0.0;
    double new_acc  = 0.0;
    bool inflection = false;
    bool surge      = false;

    {
        std::lock_guard<std::mutex> lk(mutex_);

        if (!seeded_) {
            prev_bias_    = bias;
            ema_vel_      = 0.0;
            ema_acc_      = 0.0;
            prev_vel_     = 0.0;
            prev_acc_pos_ = false;
            seeded_       = true;
        } else {
            double raw_vel = bias - prev_bias_;
            ema_vel_ += cfg_.alpha_vel * (raw_vel - ema_vel_);

            double raw_acc = ema_vel_ - prev_vel_;
            ema_acc_ += cfg_.alpha_acc * (raw_acc - ema_acc_);

            new_vel = ema_vel_;
            new_acc = ema_acc_;

            double abs_acc = std::fabs(new_acc);
            if (abs_acc > max_accel_) max_accel_ = abs_acc;

            if (total_.load(std::memory_order_relaxed) + 1 >= static_cast<std::size_t>(cfg_.min_samples)) {
                bool acc_pos = (new_acc >= 0.0);
                if (acc_pos != prev_acc_pos_) {
                    inflection = true;
                    inflection_events_.fetch_add(1, std::memory_order_relaxed);
                }
                prev_acc_pos_ = acc_pos;

                if (abs_acc > cfg_.surge_threshold) {
                    surge = true;
                    surge_events_.fetch_add(1, std::memory_order_relaxed);
                }
            }

            prev_vel_  = ema_vel_;
            prev_bias_ = bias;
        }
    }

    vel_.store(new_vel, std::memory_order_relaxed);
    acc_.store(new_acc, std::memory_order_relaxed);
    total_.fetch_add(1, std::memory_order_relaxed);

    if (inflection && cfg_.on_inflection) cfg_.on_inflection(new_vel, new_acc);
    if (surge      && cfg_.on_surge)      cfg_.on_surge(new_acc);
}

double SignalAccelerationMeter::velocity()     const noexcept { return vel_.load(std::memory_order_relaxed); }
double SignalAccelerationMeter::acceleration() const noexcept { return acc_.load(std::memory_order_relaxed); }
double SignalAccelerationMeter::max_accel()    const noexcept {
    std::lock_guard<std::mutex> lk(mutex_);
    return max_accel_;
}

std::size_t SignalAccelerationMeter::inflection_events() const noexcept { return inflection_events_.load(std::memory_order_relaxed); }
std::size_t SignalAccelerationMeter::surge_events()      const noexcept { return surge_events_.load(std::memory_order_relaxed); }
std::size_t SignalAccelerationMeter::total_records()     const noexcept { return total_.load(std::memory_order_relaxed); }

std::string SignalAccelerationMeter::to_stats_json() const {
    std::lock_guard<std::mutex> lk(mutex_);
    std::ostringstream ss;
    ss << std::fixed << std::setprecision(6);
    ss << "{\"velocity\":" << vel_.load(std::memory_order_relaxed)
       << ",\"acceleration\":" << acc_.load(std::memory_order_relaxed)
       << ",\"max_accel\":" << max_accel_
       << ",\"inflection_events\":" << inflection_events_.load(std::memory_order_relaxed)
       << ",\"surge_events\":" << surge_events_.load(std::memory_order_relaxed)
       << ",\"total_records\":" << total_.load(std::memory_order_relaxed)
       << "}";
    return ss.str();
}

void SignalAccelerationMeter::reset() {
    std::lock_guard<std::mutex> lk(mutex_);
    prev_bias_ = 0.0; ema_vel_ = 0.0; ema_acc_ = 0.0;
    prev_vel_ = 0.0; max_accel_ = 0.0; seeded_ = false;
    prev_acc_pos_ = false;
    vel_.store(0.0, std::memory_order_relaxed);
    acc_.store(0.0, std::memory_order_relaxed);
    inflection_events_.store(0, std::memory_order_relaxed);
    surge_events_.store(0, std::memory_order_relaxed);
    total_.store(0, std::memory_order_relaxed);
}

void SignalAccelerationMeter::update_config(const Config& cfg) {
    std::lock_guard<std::mutex> lk(mutex_);
    cfg_ = cfg;
}

} // namespace llmquant
