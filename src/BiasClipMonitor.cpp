#include "BiasClipMonitor.h"

#include <algorithm>
#include <cmath>
#include <sstream>
#include <iomanip>

namespace llmquant {

BiasClipMonitor::BiasClipMonitor(Config cfg)
    : cfg_(std::move(cfg)) {
    win_.resize(static_cast<size_t>(std::max(cfg_.window, 2)), 0);
}

void BiasClipMonitor::record(double bias) {
    double fire_rate = 0.0;
    bool   do_fire   = false;

    {
        std::lock_guard<std::mutex> lk(mutex_);

        obs_count_.fetch_add(1, std::memory_order_relaxed);

        int sz    = static_cast<int>(win_.size());
        int clipped = (std::abs(bias) >= cfg_.clip_threshold) ? 1 : 0;

        if (clipped) clip_count_.fetch_add(1, std::memory_order_relaxed);

        // Subtract the old value being evicted from the window.
        if (fill_ >= sz) {
            window_clips_ -= win_[static_cast<size_t>(head_)];
        }
        win_[static_cast<size_t>(head_)] = clipped;
        window_clips_ += clipped;
        head_ = (head_ + 1) % sz;
        if (fill_ < sz) ++fill_;

        double rate = static_cast<double>(window_clips_) / static_cast<double>(fill_);
        clip_rate_.store(rate, std::memory_order_relaxed);

        bool now_spiking = (rate >= cfg_.spike_threshold);
        spiking_.store(now_spiking, std::memory_order_relaxed);

        if (now_spiking && !prev_spiking_ && cfg_.on_clip_spike) {
            fire_rate = rate;
            do_fire   = true;
        }
        prev_spiking_ = now_spiking;
    }

    if (do_fire) cfg_.on_clip_spike(fire_rate);
}

std::string BiasClipMonitor::to_stats_json() const {
    std::lock_guard<std::mutex> lk(mutex_);
    std::ostringstream oss;
    oss << std::fixed << std::setprecision(4);
    oss << "{\"clip_rate\":" << clip_rate_.load(std::memory_order_relaxed)
        << ",\"clip_count\":" << clip_count_.load(std::memory_order_relaxed)
        << ",\"is_spiking\":" << (spiking_.load(std::memory_order_relaxed) ? "true" : "false")
        << ",\"obs_count\":" << obs_count_.load(std::memory_order_relaxed)
        << "}";
    return oss.str();
}

void BiasClipMonitor::reset() {
    std::lock_guard<std::mutex> lk(mutex_);
    std::fill(win_.begin(), win_.end(), 0);
    head_ = 0; fill_ = 0; window_clips_ = 0; prev_spiking_ = false;
    clip_rate_.store(0.0, std::memory_order_relaxed);
    clip_count_.store(0, std::memory_order_relaxed);
    spiking_.store(false, std::memory_order_relaxed);
    obs_count_.store(0, std::memory_order_relaxed);
}

void BiasClipMonitor::update_config(const Config& cfg) {
    std::lock_guard<std::mutex> lk(mutex_);
    cfg_ = cfg;
    win_.resize(static_cast<size_t>(std::max(cfg_.window, 2)), 0);
}

} // namespace llmquant
