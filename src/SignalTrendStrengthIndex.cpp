#include "SignalTrendStrengthIndex.h"
#include <cmath>
#include <sstream>
#include <iomanip>

namespace llmquant {

SignalTrendStrengthIndex::SignalTrendStrengthIndex(Config cfg)
    : cfg_(std::move(cfg))
    , alpha_r_(2.0 / (cfg_.r + 1))
    , alpha_s_(2.0 / (cfg_.s + 1))
{}

void SignalTrendStrengthIndex::record(double bias) {
    double new_tsi{};
    bool   cross = false;
    bool   strong = false;

    {
        std::lock_guard<std::mutex> lk(mutex_);

        if (!seeded_) {
            prev_bias_ = bias;
            seeded_    = true;
            total_.fetch_add(1, std::memory_order_relaxed);
            return;
        }

        double momentum    = bias - prev_bias_;
        double abs_momentum = std::fabs(momentum);
        prev_bias_ = bias;

        // Double-smooth momentum
        ema1_m_  = ema1_m_  + alpha_r_ * (momentum    - ema1_m_);
        ema1_am_ = ema1_am_ + alpha_r_ * (abs_momentum - ema1_am_);
        ema2_m_  = ema2_m_  + alpha_s_ * (ema1_m_     - ema2_m_);
        ema2_am_ = ema2_am_ + alpha_s_ * (ema1_am_    - ema2_am_);

        uint64_t n = total_.load(std::memory_order_relaxed);
        if (n >= static_cast<uint64_t>(cfg_.min_samples)) {
            new_tsi = (ema2_am_ > 1e-12) ? 100.0 * ema2_m_ / ema2_am_ : 0.0;

            if ((prev_tsi_ >= 0.0) != (new_tsi >= 0.0)) {
                cross = true;
                cross_events_.fetch_add(1, std::memory_order_relaxed);
            }
            if (std::fabs(new_tsi) > cfg_.strength_threshold) {
                strong = true;
                strong_events_.fetch_add(1, std::memory_order_relaxed);
            }
        }
        double old_tsi = prev_tsi_;
        prev_tsi_ = new_tsi;
        (void)old_tsi;
    }

    tsi_.store(new_tsi, std::memory_order_relaxed);
    total_.fetch_add(1, std::memory_order_relaxed);

    if (cross && cfg_.on_zero_cross) cfg_.on_zero_cross(prev_tsi_, new_tsi);
    if (strong && cfg_.on_strong)    cfg_.on_strong(new_tsi);
}

std::string SignalTrendStrengthIndex::to_stats_json() const {
    std::lock_guard<std::mutex> lk(mutex_);
    std::ostringstream ss;
    ss << std::fixed << std::setprecision(4);
    ss << "{\"tsi\":" << tsi_.load(std::memory_order_relaxed)
       << ",\"zero_cross_events\":" << cross_events_.load(std::memory_order_relaxed)
       << ",\"strong_events\":" << strong_events_.load(std::memory_order_relaxed)
       << ",\"total_records\":" << total_.load(std::memory_order_relaxed)
       << "}";
    return ss.str();
}

void SignalTrendStrengthIndex::reset() {
    std::lock_guard<std::mutex> lk(mutex_);
    ema1_m_  = 0.0; ema2_m_  = 0.0;
    ema1_am_ = 0.0; ema2_am_ = 0.0;
    prev_bias_ = 0.0; prev_tsi_ = 0.0;
    seeded_ = false;
    tsi_.store(0.0, std::memory_order_relaxed);
    cross_events_.store(0, std::memory_order_relaxed);
    strong_events_.store(0, std::memory_order_relaxed);
    total_.store(0, std::memory_order_relaxed);
}

void SignalTrendStrengthIndex::update_config(const Config& cfg) {
    std::lock_guard<std::mutex> lk(mutex_);
    cfg_     = cfg;
    alpha_r_ = 2.0 / (cfg_.r + 1);
    alpha_s_ = 2.0 / (cfg_.s + 1);
}

} // namespace llmquant
