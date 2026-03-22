#include "SignalEntropyRatchet.h"

#include <algorithm>
#include <cmath>
#include <sstream>
#include <iomanip>

namespace llmquant {

SignalEntropyRatchet::SignalEntropyRatchet(Config cfg)
    : cfg_(std::move(cfg))
{
    int w = std::max(2, cfg_.window);
    int b = std::max(2, cfg_.n_buckets);
    win_buf_.assign(static_cast<std::size_t>(w), 0);
    hist_.assign(static_cast<std::size_t>(b), 0);
}

int SignalEntropyRatchet::bucket_for(double bias) const {
    int b = static_cast<int>(hist_.size());
    double rng = cfg_.range_max - cfg_.range_min;
    if (rng <= 0.0) return 0;
    double frac = (bias - cfg_.range_min) / rng;
    frac = std::max(0.0, std::min(1.0 - 1e-12, frac));
    return static_cast<int>(frac * b);
}

double SignalEntropyRatchet::compute_entropy_locked() const {
    double H = 0.0;
    double inv_n = (win_fill_ > 0) ? 1.0 / win_fill_ : 0.0;
    for (int cnt : hist_) {
        if (cnt > 0) {
            double p = cnt * inv_n;
            H -= p * std::log2(p);
        }
    }
    return H;
}

void SignalEntropyRatchet::record(double bias) {
    bool   fire_spike = false;
    bool   fire_floor = false;
    double cur_H      = 0.0;
    double new_floor  = 0.0;
    std::function<void(double)> spike_cb, floor_cb;

    {
        std::lock_guard<std::mutex> lk(mutex_);
        spike_cb = cfg_.on_entropy_spike;
        floor_cb = cfg_.on_floor_drop;
        total_.fetch_add(1, std::memory_order_relaxed);

        int w = static_cast<int>(win_buf_.size());
        int bkt = bucket_for(bias);

        if (win_fill_ == w) {
            // Evict oldest bucket from histogram
            int old_bkt = win_buf_[static_cast<std::size_t>(win_head_)];
            if (hist_[static_cast<std::size_t>(old_bkt)] > 0)
                --hist_[static_cast<std::size_t>(old_bkt)];
        } else {
            ++win_fill_;
        }

        win_buf_[static_cast<std::size_t>(win_head_)] = bkt;
        win_head_ = (win_head_ + 1) % w;
        ++hist_[static_cast<std::size_t>(bkt)];

        cur_H = compute_entropy_locked();
        entropy_.store(cur_H, std::memory_order_relaxed);

        // Spike detection
        bool now_spiked = cur_H > cfg_.spike_threshold;
        spiked_.store(now_spiked, std::memory_order_relaxed);
        if (now_spiked && !prev_spiked_) {
            fire_spike = true;
            spike_events_.fetch_add(1, std::memory_order_relaxed);
        }
        prev_spiked_ = now_spiked;

        // Ratchet floor logic
        if (current_floor_ < 0.0) {
            // First observation — seed floor
            current_floor_ = cur_H;
            floor_.store(cur_H, std::memory_order_relaxed);
        } else if (cur_H > current_floor_) {
            // Entropy rose — raise floor eagerly
            current_floor_ = cur_H;
            floor_.store(cur_H, std::memory_order_relaxed);
            sub_floor_streak_ = 0;
        } else {
            // Entropy at or below floor
            ++sub_floor_streak_;
            int hold = std::max(1, cfg_.floor_hold_ticks);
            if (sub_floor_streak_ >= hold) {
                // Sustained below floor — allow floor to drop
                if (cur_H < current_floor_) {
                    new_floor = cur_H;
                    current_floor_ = cur_H;
                    floor_.store(cur_H, std::memory_order_relaxed);
                    floor_drop_events_.fetch_add(1, std::memory_order_relaxed);
                    fire_floor = true;
                    sub_floor_streak_ = 0;
                }
            }
        }
    }

    if (fire_spike && spike_cb) spike_cb(cur_H);
    if (fire_floor && floor_cb) floor_cb(new_floor);
}

std::string SignalEntropyRatchet::to_stats_json() const {
    std::lock_guard<std::mutex> lk(mutex_);
    std::ostringstream ss;
    ss << std::fixed << std::setprecision(6);
    ss << "{"
       << "\"entropy\":"           << entropy_.load(std::memory_order_relaxed) << ","
       << "\"entropy_floor\":"     << floor_.load(std::memory_order_relaxed) << ","
       << "\"is_spiked\":"         << (spiked_.load(std::memory_order_relaxed) ? "true" : "false") << ","
       << "\"spike_events\":"      << spike_events_.load(std::memory_order_relaxed) << ","
       << "\"floor_drop_events\":" << floor_drop_events_.load(std::memory_order_relaxed) << ","
       << "\"total_records\":"     << total_.load(std::memory_order_relaxed)
       << "}";
    return ss.str();
}

void SignalEntropyRatchet::reset() {
    std::lock_guard<std::mutex> lk(mutex_);
    std::fill(win_buf_.begin(), win_buf_.end(), 0);
    std::fill(hist_.begin(), hist_.end(), 0);
    win_head_ = win_fill_ = 0;
    current_floor_ = -1.0;
    sub_floor_streak_ = 0;
    prev_spiked_ = false;
    entropy_.store(0.0, std::memory_order_relaxed);
    floor_.store(0.0, std::memory_order_relaxed);
    spiked_.store(false, std::memory_order_relaxed);
    spike_events_.store(0, std::memory_order_relaxed);
    floor_drop_events_.store(0, std::memory_order_relaxed);
    total_.store(0, std::memory_order_relaxed);
}

void SignalEntropyRatchet::update_config(const Config& cfg) {
    std::lock_guard<std::mutex> lk(mutex_);
    cfg_ = cfg;
    int w = std::max(2, cfg_.window);
    int b = std::max(2, cfg_.n_buckets);
    win_buf_.assign(static_cast<std::size_t>(w), 0);
    hist_.assign(static_cast<std::size_t>(b), 0);
    win_head_ = win_fill_ = 0;
    current_floor_ = -1.0;
    sub_floor_streak_ = 0;
    prev_spiked_ = false;
}

} // namespace llmquant
