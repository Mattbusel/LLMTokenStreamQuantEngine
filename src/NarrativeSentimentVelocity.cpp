#include "NarrativeSentimentVelocity.h"
#include <cmath>
#include <sstream>
#include <iomanip>

namespace llmquant {

NarrativeSentimentVelocity::NarrativeSentimentVelocity(Config cfg)
    : cfg_(std::move(cfg))
{}

void NarrativeSentimentVelocity::record(double bias) {
    double new_ema = 0.0, new_vel = 0.0, new_accel = 0.0;
    bool pos_ev = false, neg_ev = false;

    {
        std::lock_guard<std::mutex> lk(mutex_);

        if (!seeded_) {
            ema_raw_ = bias;
            seeded_  = true;
        } else {
            ema_raw_ += cfg_.alpha * (bias - ema_raw_);
        }

        new_ema   = ema_raw_;
        new_vel   = ema_raw_ - prev_ema_;
        new_accel = new_vel - prev_vel_;
        prev_ema_ = ema_raw_;
        prev_vel_ = new_vel;

        if (total_.load(std::memory_order_relaxed) + 1 >= static_cast<uint64_t>(cfg_.min_samples)) {
            if (new_vel > cfg_.surge_threshold) {
                pos_ev = true;
                pos_surges_.fetch_add(1, std::memory_order_relaxed);
            } else if (new_vel < -cfg_.surge_threshold) {
                neg_ev = true;
                neg_surges_.fetch_add(1, std::memory_order_relaxed);
            }
        }
    }

    ema_.store(new_ema, std::memory_order_relaxed);
    vel_.store(new_vel, std::memory_order_relaxed);
    accel_.store(new_accel, std::memory_order_relaxed);
    total_.fetch_add(1, std::memory_order_relaxed);

    if (pos_ev && cfg_.on_positive_surge) cfg_.on_positive_surge(new_vel);
    if (neg_ev && cfg_.on_negative_surge) cfg_.on_negative_surge(new_vel);
}

double NarrativeSentimentVelocity::ema()          const noexcept { return ema_.load(std::memory_order_relaxed); }
double NarrativeSentimentVelocity::velocity()     const noexcept { return vel_.load(std::memory_order_relaxed); }
double NarrativeSentimentVelocity::acceleration() const noexcept { return accel_.load(std::memory_order_relaxed); }

std::size_t NarrativeSentimentVelocity::positive_surges() const noexcept { return pos_surges_.load(std::memory_order_relaxed); }
std::size_t NarrativeSentimentVelocity::negative_surges() const noexcept { return neg_surges_.load(std::memory_order_relaxed); }
std::size_t NarrativeSentimentVelocity::total_records()   const noexcept { return total_.load(std::memory_order_relaxed); }

std::string NarrativeSentimentVelocity::to_stats_json() const {
    std::lock_guard<std::mutex> lk(mutex_);
    std::ostringstream ss;
    ss << std::fixed << std::setprecision(6);
    ss << "{\"ema\":"            << ema_.load(std::memory_order_relaxed)
       << ",\"velocity\":"       << vel_.load(std::memory_order_relaxed)
       << ",\"acceleration\":"   << accel_.load(std::memory_order_relaxed)
       << ",\"positive_surges\":" << pos_surges_.load(std::memory_order_relaxed)
       << ",\"negative_surges\":" << neg_surges_.load(std::memory_order_relaxed)
       << ",\"total_records\":"  << total_.load(std::memory_order_relaxed)
       << "}";
    return ss.str();
}

void NarrativeSentimentVelocity::reset() {
    std::lock_guard<std::mutex> lk(mutex_);
    seeded_ = false; ema_raw_ = 0.0; prev_ema_ = 0.0; prev_vel_ = 0.0;
    ema_.store(0.0, std::memory_order_relaxed);
    vel_.store(0.0, std::memory_order_relaxed);
    accel_.store(0.0, std::memory_order_relaxed);
    pos_surges_.store(0, std::memory_order_relaxed);
    neg_surges_.store(0, std::memory_order_relaxed);
    total_.store(0, std::memory_order_relaxed);
}

void NarrativeSentimentVelocity::update_config(const Config& cfg) {
    std::lock_guard<std::mutex> lk(mutex_);
    cfg_ = cfg;
}

} // namespace llmquant
