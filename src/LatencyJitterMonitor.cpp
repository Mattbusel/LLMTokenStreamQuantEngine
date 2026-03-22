#include "LatencyJitterMonitor.h"

#include <algorithm>
#include <cmath>
#include <sstream>
#include <iomanip>

namespace llmquant {

LatencyJitterMonitor::LatencyJitterMonitor(Config cfg)
    : cfg_(std::move(cfg))
{
    buf_.resize(cfg_.window_size, 0.0);
}

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

void LatencyJitterMonitor::recompute_locked() {
    if (fill_ < cfg_.min_samples) {
        mad_.store(0.0, std::memory_order_relaxed);
        median_.store(0.0, std::memory_order_relaxed);
        return;
    }

    // Copy active samples into a scratch buffer for nth_element.
    std::vector<double> scratch(buf_.begin(), buf_.begin() + fill_);

    // Compute median via nth_element (O(N) average).
    int mid = fill_ / 2;
    std::nth_element(scratch.begin(), scratch.begin() + mid, scratch.end());
    double med = scratch[mid];
    // For even-length windows use the average of the two middle elements.
    if (fill_ % 2 == 0) {
        auto it = std::max_element(scratch.begin(), scratch.begin() + mid);
        med = (med + *it) * 0.5;
    }
    median_.store(med, std::memory_order_relaxed);

    // Compute absolute deviations from the median.
    std::vector<double> devs(fill_);
    for (int i = 0; i < fill_; ++i)
        devs[i] = std::abs(buf_[i] - med);

    std::nth_element(devs.begin(), devs.begin() + mid, devs.end());
    double mad = devs[mid];
    if (fill_ % 2 == 0) {
        auto it = std::max_element(devs.begin(), devs.begin() + mid);
        mad = (mad + *it) * 0.5;
    }
    mad_.store(mad, std::memory_order_relaxed);
}

// ---------------------------------------------------------------------------
// Core
// ---------------------------------------------------------------------------

void LatencyJitterMonitor::record(double latency_us) {
    std::function<void(double)> spike_cb;
    std::function<void(double)> clear_cb;
    bool fire_spike = false;
    bool fire_clear = false;
    double mad_val  = 0.0;

    {
        std::lock_guard<std::mutex> lk(mutex_);

        // Ring-buffer insert.
        buf_[head_] = latency_us;
        head_ = (head_ + 1) % cfg_.window_size;
        if (fill_ < cfg_.window_size) ++fill_;

        recompute_locked();
        mad_val = mad_.load(std::memory_order_relaxed);

        bool now_jittery = (mad_val > cfg_.jitter_threshold_us);

        if (now_jittery && !prev_jittery_) {
            events_.fetch_add(1, std::memory_order_relaxed);
            fire_spike = true;
            spike_cb   = cfg_.on_jitter_spike;
        } else if (!now_jittery &&
                   mad_val < cfg_.jitter_threshold_us * cfg_.clear_hysteresis &&
                   prev_jittery_) {
            fire_clear = true;
            clear_cb   = cfg_.on_jitter_clear;
        }
        prev_jittery_ = now_jittery;
        jittery_.store(now_jittery, std::memory_order_relaxed);
    }

    total_.fetch_add(1, std::memory_order_relaxed);
    if (fire_spike && spike_cb) spike_cb(mad_val);
    if (fire_clear && clear_cb) clear_cb(mad_val);
}

// ---------------------------------------------------------------------------
// Accessors
// ---------------------------------------------------------------------------

std::string LatencyJitterMonitor::to_stats_json() const {
    double m   = mad_us();
    double med = median_us();
    bool   j   = is_jittery();
    auto   tot = total_samples();
    auto   evt = jitter_events();

    std::ostringstream os;
    os << std::fixed << std::setprecision(3);
    os << "{"
       << "\"mad_us\":"        << m   << ","
       << "\"median_us\":"     << med << ","
       << "\"is_jittery\":"    << (j ? "true" : "false") << ","
       << "\"jitter_events\":" << evt << ","
       << "\"total_samples\":" << tot
       << "}";
    return os.str();
}

// ---------------------------------------------------------------------------
// Mutation
// ---------------------------------------------------------------------------

void LatencyJitterMonitor::reset() {
    std::lock_guard<std::mutex> lk(mutex_);
    std::fill(buf_.begin(), buf_.end(), 0.0);
    head_         = 0;
    fill_         = 0;
    prev_jittery_ = false;

    mad_.store(0.0, std::memory_order_relaxed);
    median_.store(0.0, std::memory_order_relaxed);
    jittery_.store(false, std::memory_order_relaxed);
    total_.store(0, std::memory_order_relaxed);
    events_.store(0, std::memory_order_relaxed);
}

void LatencyJitterMonitor::update_config(const Config& cfg) {
    std::lock_guard<std::mutex> lk(mutex_);
    cfg_  = cfg;
    buf_.assign(cfg_.window_size, 0.0);
    head_         = 0;
    fill_         = 0;
    prev_jittery_ = false;

    mad_.store(0.0, std::memory_order_relaxed);
    median_.store(0.0, std::memory_order_relaxed);
    jittery_.store(false, std::memory_order_relaxed);
    total_.store(0, std::memory_order_relaxed);
    events_.store(0, std::memory_order_relaxed);
}

} // namespace llmquant
