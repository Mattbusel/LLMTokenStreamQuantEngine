#include "NarrativeFatigueDetector.h"
#include <cmath>
#include <sstream>
#include <iomanip>

namespace llmquant {

NarrativeFatigueDetector::NarrativeFatigueDetector(Config cfg)
    : cfg_(std::move(cfg))
{}

void NarrativeFatigueDetector::record(double bias) {
    double new_ratio = 0.0;
    bool new_fatigued = false;
    bool fatigue_event  = false;
    bool recovery_event = false;

    {
        std::lock_guard<std::mutex> lk(mutex_);

        ema_fast_ += cfg_.alpha_fast * (bias - ema_fast_);
        ema_slow_ += cfg_.alpha_slow * (bias - ema_slow_);

        double denom = std::fabs(ema_slow_) + 1e-12;
        new_ratio = std::fabs(ema_fast_ - ema_slow_) / denom;

        if (total_.load(std::memory_order_relaxed) >= static_cast<std::size_t>(cfg_.min_samples)) {
            new_fatigued = (new_ratio < cfg_.fatigue_threshold);
            if (new_fatigued && !was_fatigued_) {
                fatigue_event = true;
                fatigue_events_.fetch_add(1, std::memory_order_relaxed);
            }
            if (!new_fatigued && was_fatigued_) {
                recovery_event = true;
                recovery_events_.fetch_add(1, std::memory_order_relaxed);
            }
            was_fatigued_ = new_fatigued;
        }
    }

    ratio_delta_.store(new_ratio, std::memory_order_relaxed);
    is_fatigued_.store(new_fatigued, std::memory_order_relaxed);
    total_.fetch_add(1, std::memory_order_relaxed);

    if (fatigue_event  && cfg_.on_fatigue)   cfg_.on_fatigue(new_ratio);
    if (recovery_event && cfg_.on_recovery)  cfg_.on_recovery(new_ratio);
}

double NarrativeFatigueDetector::fast_ema() const noexcept {
    std::lock_guard<std::mutex> lk(mutex_);
    return ema_fast_;
}
double NarrativeFatigueDetector::slow_ema() const noexcept {
    std::lock_guard<std::mutex> lk(mutex_);
    return ema_slow_;
}
double NarrativeFatigueDetector::ratio_delta() const noexcept { return ratio_delta_.load(std::memory_order_relaxed); }
bool   NarrativeFatigueDetector::is_fatigued() const noexcept { return is_fatigued_.load(std::memory_order_relaxed); }

std::size_t NarrativeFatigueDetector::fatigue_events()  const noexcept { return fatigue_events_.load(std::memory_order_relaxed); }
std::size_t NarrativeFatigueDetector::recovery_events() const noexcept { return recovery_events_.load(std::memory_order_relaxed); }
std::size_t NarrativeFatigueDetector::total_records()   const noexcept { return total_.load(std::memory_order_relaxed); }

std::string NarrativeFatigueDetector::to_stats_json() const {
    std::lock_guard<std::mutex> lk(mutex_);
    std::ostringstream ss;
    ss << std::fixed << std::setprecision(6);
    ss << "{\"ratio_delta\":" << ratio_delta_.load(std::memory_order_relaxed)
       << ",\"is_fatigued\":" << (is_fatigued_.load(std::memory_order_relaxed) ? "true" : "false")
       << ",\"fatigue_events\":" << fatigue_events_.load(std::memory_order_relaxed)
       << ",\"recovery_events\":" << recovery_events_.load(std::memory_order_relaxed)
       << ",\"total_records\":" << total_.load(std::memory_order_relaxed)
       << "}";
    return ss.str();
}

void NarrativeFatigueDetector::reset() {
    std::lock_guard<std::mutex> lk(mutex_);
    ema_fast_ = 0.0; ema_slow_ = 0.0; was_fatigued_ = false;
    ratio_delta_.store(0.0, std::memory_order_relaxed);
    is_fatigued_.store(false, std::memory_order_relaxed);
    fatigue_events_.store(0, std::memory_order_relaxed);
    recovery_events_.store(0, std::memory_order_relaxed);
    total_.store(0, std::memory_order_relaxed);
}

void NarrativeFatigueDetector::update_config(const Config& cfg) {
    std::lock_guard<std::mutex> lk(mutex_);
    cfg_ = cfg;
}

} // namespace llmquant
