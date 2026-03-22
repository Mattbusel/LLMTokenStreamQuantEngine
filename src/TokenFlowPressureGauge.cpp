#include "TokenFlowPressureGauge.h"

#include <algorithm>
#include <cmath>
#include <sstream>
#include <iomanip>

namespace llmquant {

TokenFlowPressureGauge::TokenFlowPressureGauge(Config cfg)
    : cfg_(std::move(cfg)) {}

void TokenFlowPressureGauge::record_token(uint64_t now_ns) {
    double fire_pressure = -1.0;
    bool   do_fire       = false;

    {
        std::lock_guard<std::mutex> lk(mutex_);

        token_count_.fetch_add(1, std::memory_order_relaxed);

        if (!seeded_) {
            last_ns_ = now_ns;
            seeded_  = true;
            return;
        }

        uint64_t interval_ns = (now_ns > last_ns_) ? (now_ns - last_ns_) : 0ULL;
        last_ns_ = now_ns;

        if (ema_interval_ns_ == 0.0) {
            ema_interval_ns_ = static_cast<double>(interval_ns);
        } else {
            ema_interval_ns_ = cfg_.ema_alpha * static_cast<double>(interval_ns)
                             + (1.0 - cfg_.ema_alpha) * ema_interval_ns_;
        }

        double p = 0.0;
        if (ema_interval_ns_ > 0.0) {
            p = static_cast<double>(cfg_.target_interval_ns) / ema_interval_ns_;
            p = std::clamp(p, 0.0, 1.0);
        }

        pressure_.store(p, std::memory_order_relaxed);

        bool now_spiking = (p >= cfg_.spike_threshold);
        spiking_.store(now_spiking, std::memory_order_relaxed);

        if (now_spiking && !prev_spiking_ && cfg_.on_pressure_spike) {
            fire_pressure = p;
            do_fire       = true;
        }
        prev_spiking_ = now_spiking;
    }

    if (do_fire) cfg_.on_pressure_spike(fire_pressure);
}

std::string TokenFlowPressureGauge::to_stats_json() const {
    std::lock_guard<std::mutex> lk(mutex_);
    std::ostringstream oss;
    oss << std::fixed << std::setprecision(4);
    oss << "{\"pressure\":" << pressure_.load(std::memory_order_relaxed)
        << ",\"ema_interval_ns\":" << ema_interval_ns_
        << ",\"target_interval_ns\":" << cfg_.target_interval_ns
        << ",\"is_spiking\":" << (spiking_.load(std::memory_order_relaxed) ? "true" : "false")
        << ",\"token_count\":" << token_count_.load(std::memory_order_relaxed)
        << "}";
    return oss.str();
}

void TokenFlowPressureGauge::reset() {
    std::lock_guard<std::mutex> lk(mutex_);
    last_ns_      = 0;
    ema_interval_ns_ = 0.0;
    seeded_       = false;
    prev_spiking_ = false;
    pressure_.store(0.0, std::memory_order_relaxed);
    spiking_.store(false, std::memory_order_relaxed);
    token_count_.store(0, std::memory_order_relaxed);
}

void TokenFlowPressureGauge::update_config(const Config& cfg) {
    std::lock_guard<std::mutex> lk(mutex_);
    cfg_ = cfg;
}

} // namespace llmquant
