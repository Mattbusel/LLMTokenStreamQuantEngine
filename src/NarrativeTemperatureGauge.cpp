#include "NarrativeTemperatureGauge.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <mutex>
#include <numeric>

namespace llmquant {

NarrativeTemperatureGauge::NarrativeTemperatureGauge(Config cfg)
    : cfg_(std::move(cfg))
{
    buf_.resize(cfg_.window > 0 ? cfg_.window : 32, 0.0);
}

void NarrativeTemperatureGauge::record(double bias) {
    std::function<void(double)> hot_cb, cool_cb;
    double temp_out = 0.0;

    {
        std::lock_guard<std::mutex> lk(mutex_);

        // EMA update.
        if (!ema_seeded_) {
            ema_ = bias;
            ema_seeded_ = true;
        } else {
            ema_ = cfg_.ema_alpha * bias + (1.0 - cfg_.ema_alpha) * ema_;
        }

        // Rolling window.
        buf_[head_] = bias;
        head_ = (head_ + 1) % static_cast<int>(buf_.size());
        if (fill_ < static_cast<int>(buf_.size())) ++fill_;

        // Rolling stddev.
        double sigma = 0.0;
        if (fill_ > 1) {
            double sum = 0.0;
            for (int i = 0; i < fill_; ++i) sum += buf_[i];
            double mean = sum / fill_;
            double var  = 0.0;
            for (int i = 0; i < fill_; ++i) var += (buf_[i] - mean) * (buf_[i] - mean);
            sigma = std::sqrt(var / fill_);
        }

        // Temperature = clamp(|ema| * sigma / normalizer, 0, 1).
        double normalizer = cfg_.normalizer > 0.0 ? cfg_.normalizer : 0.5;
        double raw = std::fabs(ema_) * sigma / normalizer;
        temp_out = std::min(1.0, std::max(0.0, raw));

        // Publish atomics.
        temperature_.store(temp_out, std::memory_order_relaxed);
        bias_ema_.store(ema_, std::memory_order_relaxed);
        bias_sigma_.store(sigma, std::memory_order_relaxed);
        total_.fetch_add(1, std::memory_order_relaxed);

        // Threshold crossings.
        bool now_hot = temp_out >= cfg_.hot_threshold;
        double cool_thr = cfg_.hot_threshold * cfg_.cool_hysteresis;
        if (now_hot && !prev_hot_) {
            hot_.store(true, std::memory_order_relaxed);
            heat_events_.fetch_add(1, std::memory_order_relaxed);
            hot_cb = cfg_.on_hot;
        } else if (!now_hot && prev_hot_ && temp_out < cool_thr) {
            hot_.store(false, std::memory_order_relaxed);
            cool_cb = cfg_.on_cool;
        }
        prev_hot_ = now_hot;
    }

    if (hot_cb)  try { hot_cb(temp_out);  } catch (...) {}
    if (cool_cb) try { cool_cb(temp_out); } catch (...) {}
}

void NarrativeTemperatureGauge::reset() {
    std::lock_guard<std::mutex> lk(mutex_);
    std::fill(buf_.begin(), buf_.end(), 0.0);
    head_ = 0; fill_ = 0;
    ema_ = 0.0; ema_seeded_ = false; prev_hot_ = false;
    temperature_.store(0.0, std::memory_order_relaxed);
    bias_ema_.store(0.0, std::memory_order_relaxed);
    bias_sigma_.store(0.0, std::memory_order_relaxed);
    hot_.store(false, std::memory_order_relaxed);
    heat_events_.store(0, std::memory_order_relaxed);
    total_.store(0, std::memory_order_relaxed);
}

void NarrativeTemperatureGauge::update_config(const Config& cfg) {
    std::lock_guard<std::mutex> lk(mutex_);
    cfg_ = cfg;
    buf_.assign(cfg_.window > 0 ? cfg_.window : 32, 0.0);
    head_ = 0; fill_ = 0;
    ema_ = 0.0; ema_seeded_ = false; prev_hot_ = false;
    temperature_.store(0.0, std::memory_order_relaxed);
    bias_ema_.store(0.0, std::memory_order_relaxed);
    bias_sigma_.store(0.0, std::memory_order_relaxed);
    hot_.store(false, std::memory_order_relaxed);
    heat_events_.store(0, std::memory_order_relaxed);
    total_.store(0, std::memory_order_relaxed);
}

std::string NarrativeTemperatureGauge::to_stats_json() const {
    double t  = temperature_.load(std::memory_order_relaxed);
    double em = bias_ema_.load(std::memory_order_relaxed);
    double sg = bias_sigma_.load(std::memory_order_relaxed);
    bool   h  = hot_.load(std::memory_order_relaxed);
    uint64_t he = heat_events_.load(std::memory_order_relaxed);
    uint64_t to = total_.load(std::memory_order_relaxed);
    char buf[256];
    std::snprintf(buf, sizeof(buf),
        "{\"temperature\":%.4f,\"is_hot\":%s,\"bias_ema\":%.6f,"
        "\"bias_sigma\":%.6f,\"heat_events\":%llu,\"total\":%llu}",
        t, h ? "true" : "false", em, sg,
        static_cast<unsigned long long>(he),
        static_cast<unsigned long long>(to));
    return buf;
}

} // namespace llmquant
