#include "SignalAdaptiveThresholdFilter.h"
#include <cmath>
#include <sstream>
#include <iomanip>

namespace llmquant {

SignalAdaptiveThresholdFilter::SignalAdaptiveThresholdFilter(Config cfg)
    : cfg_(std::move(cfg))
{}

void SignalAdaptiveThresholdFilter::record(double bias) {
    double new_atr = 0.0, new_thresh = 0.0;
    bool above_ev = false, below_ev = false;
    bool new_above = false, new_below = false;

    {
        std::lock_guard<std::mutex> lk(mutex_);

        if (!seeded_) {
            atr_raw_  = std::fabs(bias);
            seeded_   = true;
        } else {
            double change = std::fabs(bias - prev_bias_);
            atr_raw_ += cfg_.alpha_atr * (change - atr_raw_);
        }
        prev_bias_ = bias;

        new_atr    = atr_raw_;
        new_thresh = cfg_.multiplier * atr_raw_;

        if (total_.load(std::memory_order_relaxed) + 1 >= static_cast<std::size_t>(cfg_.min_samples)) {
            new_above = (bias >  new_thresh);
            new_below = (bias < -new_thresh);

            if (new_above && !prev_above_) {
                above_ev = true;
                above_ev_.fetch_add(1, std::memory_order_relaxed);
            }
            if (new_below && !prev_below_) {
                below_ev = true;
                below_ev_.fetch_add(1, std::memory_order_relaxed);
            }
            prev_above_ = new_above;
            prev_below_ = new_below;
        }
    }

    atr_.store(new_atr,       std::memory_order_relaxed);
    thresh_.store(new_thresh, std::memory_order_relaxed);
    above_.store(new_above,   std::memory_order_relaxed);
    below_.store(new_below,   std::memory_order_relaxed);
    total_.fetch_add(1,       std::memory_order_relaxed);

    if (above_ev && cfg_.on_above) cfg_.on_above(bias, new_thresh);
    if (below_ev && cfg_.on_below) cfg_.on_below(bias, new_thresh);
}

double SignalAdaptiveThresholdFilter::atr()       const noexcept { return atr_.load(std::memory_order_relaxed); }
double SignalAdaptiveThresholdFilter::threshold() const noexcept { return thresh_.load(std::memory_order_relaxed); }
bool   SignalAdaptiveThresholdFilter::is_above()  const noexcept { return above_.load(std::memory_order_relaxed); }
bool   SignalAdaptiveThresholdFilter::is_below()  const noexcept { return below_.load(std::memory_order_relaxed); }

std::size_t SignalAdaptiveThresholdFilter::above_events()  const noexcept { return above_ev_.load(std::memory_order_relaxed); }
std::size_t SignalAdaptiveThresholdFilter::below_events()  const noexcept { return below_ev_.load(std::memory_order_relaxed); }
std::size_t SignalAdaptiveThresholdFilter::total_records() const noexcept { return total_.load(std::memory_order_relaxed); }

std::string SignalAdaptiveThresholdFilter::to_stats_json() const {
    std::ostringstream ss;
    ss << std::fixed << std::setprecision(6);
    ss << "{\"atr\":"           << atr_.load(std::memory_order_relaxed)
       << ",\"threshold\":"     << thresh_.load(std::memory_order_relaxed)
       << ",\"is_above\":"      << (above_.load(std::memory_order_relaxed) ? "true" : "false")
       << ",\"is_below\":"      << (below_.load(std::memory_order_relaxed) ? "true" : "false")
       << ",\"above_events\":"  << above_ev_.load(std::memory_order_relaxed)
       << ",\"below_events\":"  << below_ev_.load(std::memory_order_relaxed)
       << ",\"total_records\":" << total_.load(std::memory_order_relaxed)
       << "}";
    return ss.str();
}

void SignalAdaptiveThresholdFilter::reset() {
    std::lock_guard<std::mutex> lk(mutex_);
    seeded_ = false; atr_raw_ = 0.0; prev_bias_ = 0.0;
    prev_above_ = prev_below_ = false;
    atr_.store(0.0, std::memory_order_relaxed);
    thresh_.store(0.0, std::memory_order_relaxed);
    above_.store(false, std::memory_order_relaxed);
    below_.store(false, std::memory_order_relaxed);
    above_ev_.store(0, std::memory_order_relaxed);
    below_ev_.store(0, std::memory_order_relaxed);
    total_.store(0, std::memory_order_relaxed);
}

void SignalAdaptiveThresholdFilter::update_config(const Config& cfg) {
    std::lock_guard<std::mutex> lk(mutex_);
    cfg_ = cfg;
}

} // namespace llmquant
