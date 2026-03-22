#include "SignalKurtosisTracker.h"
#include <cmath>
#include <sstream>
#include <iomanip>

namespace llmquant {

SignalKurtosisTracker::SignalKurtosisTracker(Config cfg)
    : cfg_(std::move(cfg))
    , buf_(cfg_.window, 0.0)
{}

void SignalKurtosisTracker::record(double bias) {
    double new_kurt = 0.0;
    bool fat_event    = false;
    bool normal_event = false;

    {
        std::lock_guard<std::mutex> lk(mutex_);

        buf_[head_] = bias;
        head_ = (head_ + 1) % cfg_.window;
        if (fill_ < cfg_.window) ++fill_;

        if (fill_ >= cfg_.min_samples) {
            int n = fill_;
            double sum = 0.0;
            for (int i = 0; i < n; ++i) sum += buf_[i];
            double mean = sum / n;

            double m2 = 0.0, m4 = 0.0;
            for (int i = 0; i < n; ++i) {
                double d = buf_[i] - mean;
                double d2 = d * d;
                m2 += d2;
                m4 += d2 * d2;
            }
            double var = m2 / n;

            if (var > 1e-14) {
                new_kurt = (m4 / n) / (var * var) - 3.0; // excess kurtosis
            }

            bool fat = (new_kurt > cfg_.fat_tail_threshold);
            if (fat && !prev_fat_tail_) {
                fat_event = true;
                fat_events_.fetch_add(1, std::memory_order_relaxed);
            } else if (!fat && prev_fat_tail_) {
                normal_event = true;
                normal_events_.fetch_add(1, std::memory_order_relaxed);
            }
            prev_fat_tail_ = fat;
        }
    }

    kurtosis_.store(new_kurt, std::memory_order_relaxed);
    total_.fetch_add(1, std::memory_order_relaxed);

    if (fat_event    && cfg_.on_fat_tail)    cfg_.on_fat_tail(new_kurt);
    if (normal_event && cfg_.on_normal_tail) cfg_.on_normal_tail(new_kurt);
}

double SignalKurtosisTracker::kurtosis() const noexcept { return kurtosis_.load(std::memory_order_relaxed); }

double SignalKurtosisTracker::mean() const noexcept {
    std::lock_guard<std::mutex> lk(mutex_);
    if (fill_ == 0) return 0.0;
    double s = 0.0;
    for (int i = 0; i < fill_; ++i) s += buf_[i];
    return s / fill_;
}

std::size_t SignalKurtosisTracker::fat_tail_events()    const noexcept { return fat_events_.load(std::memory_order_relaxed); }
std::size_t SignalKurtosisTracker::normal_tail_events() const noexcept { return normal_events_.load(std::memory_order_relaxed); }
std::size_t SignalKurtosisTracker::total_records()      const noexcept { return total_.load(std::memory_order_relaxed); }

std::string SignalKurtosisTracker::to_stats_json() const {
    std::lock_guard<std::mutex> lk(mutex_);
    std::ostringstream ss;
    ss << std::fixed << std::setprecision(4);
    ss << "{\"kurtosis\":" << kurtosis_.load(std::memory_order_relaxed)
       << ",\"fat_tail_events\":" << fat_events_.load(std::memory_order_relaxed)
       << ",\"normal_tail_events\":" << normal_events_.load(std::memory_order_relaxed)
       << ",\"total_records\":" << total_.load(std::memory_order_relaxed)
       << "}";
    return ss.str();
}

void SignalKurtosisTracker::reset() {
    std::lock_guard<std::mutex> lk(mutex_);
    std::fill(buf_.begin(), buf_.end(), 0.0);
    head_ = 0; fill_ = 0; prev_fat_tail_ = false;
    kurtosis_.store(0.0, std::memory_order_relaxed);
    fat_events_.store(0, std::memory_order_relaxed);
    normal_events_.store(0, std::memory_order_relaxed);
    total_.store(0, std::memory_order_relaxed);
}

void SignalKurtosisTracker::update_config(const Config& cfg) {
    std::lock_guard<std::mutex> lk(mutex_);
    cfg_ = cfg;
    buf_.assign(cfg_.window, 0.0);
    head_ = 0; fill_ = 0;
}

} // namespace llmquant
