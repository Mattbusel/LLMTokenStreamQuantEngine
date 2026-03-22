#include "NarrativePolarityIndex.h"
#include <cmath>
#include <sstream>
#include <iomanip>

namespace llmquant {

NarrativePolarityIndex::NarrativePolarityIndex(Config cfg)
    : cfg_(std::move(cfg))
{}

void NarrativePolarityIndex::record(double bias) {
    double new_npi{};
    bool flip    = false;
    bool extreme = false;

    {
        std::lock_guard<std::mutex> lk(mutex_);

        double a = cfg_.alpha;

        if (bias > 0.0) {
            bull_ema_ = bull_ema_ * (1.0 - a) + bias * a;
            bear_ema_ = bear_ema_ * (1.0 - a);
        } else {
            bull_ema_ = bull_ema_ * (1.0 - a);
            bear_ema_ = bear_ema_ * (1.0 - a) + (-bias) * a;
        }

        double denom = bull_ema_ + bear_ema_;
        new_npi = (denom > 1e-10)
                    ? (bull_ema_ - bear_ema_) / denom
                    : 0.0;

        uint64_t n = total_.load(std::memory_order_relaxed);
        if (n >= static_cast<uint64_t>(cfg_.min_samples)) {
            if (seeded_ && ((prev_npi_ >= 0.0) != (new_npi >= 0.0))) {
                flip = true;
                flips_.fetch_add(1, std::memory_order_relaxed);
            }
            if (std::fabs(new_npi) > cfg_.extreme_threshold) {
                extreme = true;
                extremes_.fetch_add(1, std::memory_order_relaxed);
            }
        }
        double old_npi = prev_npi_;
        prev_npi_ = new_npi;
        seeded_   = true;
        (void)old_npi;
        bull_.store(bull_ema_, std::memory_order_relaxed);
        bear_.store(bear_ema_, std::memory_order_relaxed);
    }

    npi_.store(new_npi, std::memory_order_relaxed);
    total_.fetch_add(1, std::memory_order_relaxed);

    if (flip && cfg_.on_polarity_flip) {
        double old_npi_cb{};
        {
            std::lock_guard<std::mutex> lk(mutex_);
            old_npi_cb = prev_npi_;
        }
        cfg_.on_polarity_flip(old_npi_cb, new_npi);
    }
    if (extreme && cfg_.on_extreme) cfg_.on_extreme(new_npi);
}

std::string NarrativePolarityIndex::to_stats_json() const {
    std::lock_guard<std::mutex> lk(mutex_);
    std::ostringstream ss;
    ss << std::fixed << std::setprecision(4);
    ss << "{\"npi\":" << npi_.load(std::memory_order_relaxed)
       << ",\"bull_ema\":" << bull_.load(std::memory_order_relaxed)
       << ",\"bear_ema\":" << bear_.load(std::memory_order_relaxed)
       << ",\"flip_events\":" << flips_.load(std::memory_order_relaxed)
       << ",\"extreme_events\":" << extremes_.load(std::memory_order_relaxed)
       << ",\"total_records\":" << total_.load(std::memory_order_relaxed)
       << "}";
    return ss.str();
}

void NarrativePolarityIndex::reset() {
    std::lock_guard<std::mutex> lk(mutex_);
    bull_ema_ = 0.0;
    bear_ema_ = 0.0;
    prev_npi_ = 0.0;
    seeded_   = false;
    npi_.store(0.0, std::memory_order_relaxed);
    bull_.store(0.0, std::memory_order_relaxed);
    bear_.store(0.0, std::memory_order_relaxed);
    flips_.store(0, std::memory_order_relaxed);
    extremes_.store(0, std::memory_order_relaxed);
    total_.store(0, std::memory_order_relaxed);
}

void NarrativePolarityIndex::update_config(const Config& cfg) {
    std::lock_guard<std::mutex> lk(mutex_);
    cfg_ = cfg;
}

} // namespace llmquant
