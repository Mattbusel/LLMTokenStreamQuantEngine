#include "SignalOscillationDetector.h"

#include <algorithm>
#include <cmath>
#include <iomanip>
#include <sstream>

namespace llmquant {

SignalOscillationDetector::SignalOscillationDetector(Config cfg)
    : cfg_(std::move(cfg))
{
    int w = std::max(4, cfg_.window);
    sign_ring_.assign(static_cast<std::size_t>(w), 0);
}

void SignalOscillationDetector::record(double bias) {
    std::function<void(double)> osc_cb, stab_cb;
    bool fire_osc = false, fire_stab = false;
    double zcr_val = 0.0;

    {
        std::lock_guard<std::mutex> lk(mutex_);
        osc_cb  = cfg_.on_oscillating;
        stab_cb = cfg_.on_stabilized;

        total_.fetch_add(1, std::memory_order_relaxed);

        int8_t new_sign = 0;
        if (!seeded_) {
            prev_bias_ = bias;
            seeded_    = true;
        } else {
            double delta = bias - prev_bias_;
            new_sign  = (delta > 0.0) ? 1 : ((delta < 0.0) ? -1 : 0);
            prev_bias_ = bias;

            sign_ring_[static_cast<std::size_t>(head_)] = new_sign;
            head_ = (head_ + 1) % static_cast<int>(sign_ring_.size());
            if (fill_ < static_cast<int>(sign_ring_.size())) ++fill_;

            int min_obs = std::max(2, cfg_.min_samples);
            if (fill_ >= min_obs) {
                // Count sign changes in the filled window.
                int n    = fill_;
                int size = static_cast<int>(sign_ring_.size());
                int crossings = 0;
                int8_t prev_s = sign_ring_[static_cast<std::size_t>((head_ - n + size * 2) % size)];
                for (int i = 1; i < n; ++i) {
                    int8_t cur_s = sign_ring_[static_cast<std::size_t>((head_ - n + i + size * 2) % size)];
                    if (cur_s != 0 && prev_s != 0 && cur_s != prev_s) ++crossings;
                    if (cur_s != 0) prev_s = cur_s;
                }
                zcr_val = (n > 1) ? static_cast<double>(crossings) / (n - 1) : 0.0;
                zcr_.store(zcr_val, std::memory_order_relaxed);

                bool now_osc = zcr_val > cfg_.oscillation_threshold;
                oscillating_.store(now_osc, std::memory_order_relaxed);

                if (now_osc && !prev_oscillating_) {
                    fire_osc = true;
                    events_.fetch_add(1, std::memory_order_relaxed);
                } else if (!now_osc && prev_oscillating_ &&
                           zcr_val < cfg_.oscillation_threshold * cfg_.clear_hysteresis) {
                    fire_stab = true;
                }
                prev_oscillating_ = now_osc;
            }
        }
    }

    if (fire_osc  && osc_cb)  osc_cb(zcr_val);
    if (fire_stab && stab_cb) stab_cb(zcr_val);
}

std::string SignalOscillationDetector::to_stats_json() const {
    double zcr_val;
    bool   osc;
    uint64_t events, total;
    {
        std::lock_guard<std::mutex> lk(mutex_);
        zcr_val = zcr_.load(std::memory_order_relaxed);
        osc     = oscillating_.load(std::memory_order_relaxed);
        events  = events_.load(std::memory_order_relaxed);
        total   = total_.load(std::memory_order_relaxed);
    }
    std::ostringstream ss;
    ss << std::fixed << std::setprecision(6);
    ss << "{"
       << "\"zcr\":"                  << zcr_val << ","
       << "\"is_oscillating\":"       << (osc ? "true" : "false") << ","
       << "\"oscillation_events\":"   << events  << ","
       << "\"total_records\":"        << total
       << "}";
    return ss.str();
}

void SignalOscillationDetector::reset() {
    std::lock_guard<std::mutex> lk(mutex_);
    std::fill(sign_ring_.begin(), sign_ring_.end(), int8_t{0});
    head_ = fill_ = 0;
    prev_bias_        = 0.0;
    seeded_           = false;
    prev_oscillating_ = false;
    zcr_.store(0.0,         std::memory_order_relaxed);
    oscillating_.store(false, std::memory_order_relaxed);
    events_.store(0,        std::memory_order_relaxed);
    total_.store(0,         std::memory_order_relaxed);
}

void SignalOscillationDetector::update_config(const Config& cfg) {
    std::lock_guard<std::mutex> lk(mutex_);
    cfg_ = cfg;
    int w = std::max(4, cfg_.window);
    sign_ring_.assign(static_cast<std::size_t>(w), int8_t{0});
    head_ = fill_ = 0;
    prev_bias_        = 0.0;
    seeded_           = false;
    prev_oscillating_ = false;
    zcr_.store(0.0,         std::memory_order_relaxed);
    oscillating_.store(false, std::memory_order_relaxed);
    events_.store(0,        std::memory_order_relaxed);
    total_.store(0,         std::memory_order_relaxed);
}

} // namespace llmquant
