#include "BiasSkewnessTracker.h"
#include <cmath>
#include <numeric>
#include <sstream>
#include <iomanip>

namespace llmquant {

BiasSkewnessTracker::BiasSkewnessTracker(Config cfg)
    : cfg_(std::move(cfg))
    , buf_(cfg_.window, 0.0)
{}

void BiasSkewnessTracker::record(double bias) {
    double new_skew = 0.0;
    bool pos_event  = false;
    bool neg_event  = false;

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

            double m2 = 0.0, m3 = 0.0;
            for (int i = 0; i < n; ++i) {
                double d = buf_[i] - mean;
                m2 += d * d;
                m3 += d * d * d;
            }
            double var = m2 / n;
            double std_dev = std::sqrt(var);

            if (std_dev > 1e-12) {
                new_skew = (m3 / n) / (std_dev * std_dev * std_dev);
            }

            bool pos = (new_skew > 0.0);
            bool prev_pos = prev_positive_;
            if (pos && !prev_pos) {
                pos_event = true;
                pos_events_.fetch_add(1, std::memory_order_relaxed);
            } else if (!pos && prev_pos) {
                neg_event = true;
                neg_events_.fetch_add(1, std::memory_order_relaxed);
            }
            prev_positive_ = pos;
        }
    }

    skewness_.store(new_skew, std::memory_order_relaxed);
    total_.fetch_add(1, std::memory_order_relaxed);

    if (pos_event && cfg_.on_positive_skew) cfg_.on_positive_skew(new_skew);
    if (neg_event && cfg_.on_negative_skew) cfg_.on_negative_skew(new_skew);
}

double BiasSkewnessTracker::skewness() const noexcept { return skewness_.load(std::memory_order_relaxed); }

double BiasSkewnessTracker::mean() const noexcept {
    std::lock_guard<std::mutex> lk(mutex_);
    if (fill_ == 0) return 0.0;
    double s = 0.0;
    for (int i = 0; i < fill_; ++i) s += buf_[i];
    return s / fill_;
}

double BiasSkewnessTracker::variance() const noexcept {
    std::lock_guard<std::mutex> lk(mutex_);
    if (fill_ < 2) return 0.0;
    double s = 0.0;
    for (int i = 0; i < fill_; ++i) s += buf_[i];
    double m = s / fill_;
    double v = 0.0;
    for (int i = 0; i < fill_; ++i) { double d = buf_[i] - m; v += d * d; }
    return v / fill_;
}

std::size_t BiasSkewnessTracker::positive_skew_events() const noexcept { return pos_events_.load(std::memory_order_relaxed); }
std::size_t BiasSkewnessTracker::negative_skew_events() const noexcept { return neg_events_.load(std::memory_order_relaxed); }
std::size_t BiasSkewnessTracker::total_records()        const noexcept { return total_.load(std::memory_order_relaxed); }

std::string BiasSkewnessTracker::to_stats_json() const {
    std::lock_guard<std::mutex> lk(mutex_);
    std::ostringstream ss;
    ss << std::fixed << std::setprecision(6);
    ss << "{\"skewness\":" << skewness_.load(std::memory_order_relaxed)
       << ",\"positive_skew_events\":" << pos_events_.load(std::memory_order_relaxed)
       << ",\"negative_skew_events\":" << neg_events_.load(std::memory_order_relaxed)
       << ",\"total_records\":" << total_.load(std::memory_order_relaxed)
       << "}";
    return ss.str();
}

void BiasSkewnessTracker::reset() {
    std::lock_guard<std::mutex> lk(mutex_);
    std::fill(buf_.begin(), buf_.end(), 0.0);
    head_ = 0; fill_ = 0; prev_positive_ = false;
    skewness_.store(0.0, std::memory_order_relaxed);
    pos_events_.store(0, std::memory_order_relaxed);
    neg_events_.store(0, std::memory_order_relaxed);
    total_.store(0, std::memory_order_relaxed);
}

void BiasSkewnessTracker::update_config(const Config& cfg) {
    std::lock_guard<std::mutex> lk(mutex_);
    cfg_ = cfg;
    buf_.assign(cfg_.window, 0.0);
    head_ = 0; fill_ = 0;
}

} // namespace llmquant
