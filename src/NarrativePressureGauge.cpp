#include "NarrativePressureGauge.h"
#include <sstream>
#include <iomanip>

namespace llmquant {

NarrativePressureGauge::NarrativePressureGauge(Config cfg)
    : cfg_(std::move(cfg))
{}

void NarrativePressureGauge::record(double bias) {
    double new_fast = 0.0, new_slow = 0.0, new_press = 0.0;
    bool pos_ev = false, neg_ev = false;

    {
        std::lock_guard<std::mutex> lk(mutex_);

        if (!seeded_) {
            ema_fast_ = bias;
            ema_slow_ = bias;
            seeded_   = true;
        } else {
            ema_fast_ += cfg_.alpha_fast * (bias - ema_fast_);
            ema_slow_ += cfg_.alpha_slow * (bias - ema_slow_);
        }

        new_fast  = ema_fast_;
        new_slow  = ema_slow_;
        new_press = ema_fast_ - ema_slow_;

        if (total_.load(std::memory_order_relaxed) + 1 >= cfg_.min_samples) {
            bool pos_p = (new_press >  cfg_.pressure_threshold);
            bool neg_p = (new_press < -cfg_.pressure_threshold);

            if (pos_p && !prev_pos_pressure_) {
                pos_ev = true;
                pos_press_ev_.fetch_add(1, std::memory_order_relaxed);
            }
            if (neg_p && !prev_neg_pressure_) {
                neg_ev = true;
                neg_press_ev_.fetch_add(1, std::memory_order_relaxed);
            }
            prev_pos_pressure_ = pos_p;
            prev_neg_pressure_ = neg_p;
        }
    }

    fast_.store(new_fast,   std::memory_order_relaxed);
    slow_.store(new_slow,   std::memory_order_relaxed);
    pressure_.store(new_press, std::memory_order_relaxed);
    total_.fetch_add(1, std::memory_order_relaxed);

    if (pos_ev && cfg_.on_positive_pressure) cfg_.on_positive_pressure(new_press);
    if (neg_ev && cfg_.on_negative_pressure) cfg_.on_negative_pressure(new_press);
}

double NarrativePressureGauge::fast_ema() const noexcept { return fast_.load(std::memory_order_relaxed); }
double NarrativePressureGauge::slow_ema() const noexcept { return slow_.load(std::memory_order_relaxed); }
double NarrativePressureGauge::pressure() const noexcept { return pressure_.load(std::memory_order_relaxed); }

std::size_t NarrativePressureGauge::positive_pressure_events() const noexcept { return pos_press_ev_.load(std::memory_order_relaxed); }
std::size_t NarrativePressureGauge::negative_pressure_events() const noexcept { return neg_press_ev_.load(std::memory_order_relaxed); }
std::size_t NarrativePressureGauge::total_records()            const noexcept { return total_.load(std::memory_order_relaxed); }

std::string NarrativePressureGauge::to_stats_json() const {
    std::lock_guard<std::mutex> lk(mutex_);
    std::ostringstream ss;
    ss << std::fixed << std::setprecision(6);
    ss << "{\"fast_ema\":"     << fast_.load(std::memory_order_relaxed)
       << ",\"slow_ema\":"     << slow_.load(std::memory_order_relaxed)
       << ",\"pressure\":"     << pressure_.load(std::memory_order_relaxed)
       << ",\"pos_press_ev\":" << pos_press_ev_.load(std::memory_order_relaxed)
       << ",\"neg_press_ev\":" << neg_press_ev_.load(std::memory_order_relaxed)
       << ",\"total_records\":" << total_.load(std::memory_order_relaxed)
       << "}";
    return ss.str();
}

void NarrativePressureGauge::reset() {
    std::lock_guard<std::mutex> lk(mutex_);
    seeded_ = false; ema_fast_ = 0.0; ema_slow_ = 0.0;
    prev_pos_pressure_ = false; prev_neg_pressure_ = false;
    fast_.store(0.0, std::memory_order_relaxed);
    slow_.store(0.0, std::memory_order_relaxed);
    pressure_.store(0.0, std::memory_order_relaxed);
    pos_press_ev_.store(0, std::memory_order_relaxed);
    neg_press_ev_.store(0, std::memory_order_relaxed);
    total_.store(0, std::memory_order_relaxed);
}

void NarrativePressureGauge::update_config(const Config& cfg) {
    std::lock_guard<std::mutex> lk(mutex_);
    cfg_ = cfg;
}

} // namespace llmquant
