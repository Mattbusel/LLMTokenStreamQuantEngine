#include "BiasRollingQuantileTracker.h"

#include <algorithm>
#include <cmath>
#include <sstream>
#include <iomanip>

namespace llmquant {

BiasRollingQuantileTracker::BiasRollingQuantileTracker(Config cfg)
    : cfg_(std::move(cfg))
{
    int w = std::max(8, cfg_.window);
    win_buf_.assign(static_cast<std::size_t>(w), 0.0);
}

void BiasRollingQuantileTracker::compute_quantiles_locked() {
    int n = win_fill_;
    if (n < std::max(2, cfg_.min_samples)) return;

    int w = static_cast<int>(win_buf_.size());
    // Build chronological snapshot
    std::vector<double> snap(static_cast<std::size_t>(n));
    for (int i = 0; i < n; ++i) {
        int idx = (win_head_ - n + i + w * 2) % w;
        snap[static_cast<std::size_t>(i)] = win_buf_[static_cast<std::size_t>(idx)];
    }

    // nth_element for each quantile
    auto qval = [&](double q) -> double {
        int pos = static_cast<int>(q * (n - 1));
        pos = std::max(0, std::min(n - 1, pos));
        std::vector<double> tmp = snap;
        std::nth_element(tmp.begin(), tmp.begin() + pos, tmp.end());
        return tmp[static_cast<std::size_t>(pos)];
    };

    double q10 = qval(0.10);
    double q25 = qval(0.25);
    double q50 = qval(0.50);
    double q75 = qval(0.75);
    double q90 = qval(0.90);

    p10_.store(q10, std::memory_order_relaxed);
    p25_.store(q25, std::memory_order_relaxed);
    p50_.store(q50, std::memory_order_relaxed);
    p75_.store(q75, std::memory_order_relaxed);
    p90_.store(q90, std::memory_order_relaxed);

    constexpr double eps = 1e-12;
    double upper = q75 - q50;
    double lower = q50 - q25;
    double skew  = upper / (lower + eps);
    skew_.store(skew, std::memory_order_relaxed);
}

void BiasRollingQuantileTracker::record(double bias) {
    bool   fire_median = false, fire_skew = false;
    double old_p50 = 0.0, new_p50 = 0.0, new_skew = 0.0;
    std::function<void(double, double)> median_cb;
    std::function<void(double)>         skew_cb;

    {
        std::lock_guard<std::mutex> lk(mutex_);
        median_cb = cfg_.on_median_shift;
        skew_cb   = cfg_.on_skew_change;
        total_.fetch_add(1, std::memory_order_relaxed);

        int w = static_cast<int>(win_buf_.size());
        if (win_fill_ < w) ++win_fill_;
        win_buf_[static_cast<std::size_t>(win_head_)] = bias;
        win_head_ = (win_head_ + 1) % w;

        compute_quantiles_locked();

        double cur_p50  = p50_.load(std::memory_order_relaxed);
        double cur_skew = skew_.load(std::memory_order_relaxed);

        if (!seeded_) {
            prev_p50_  = cur_p50;
            prev_skew_ = cur_skew;
            seeded_    = true;
        } else {
            if (std::abs(cur_p50 - prev_p50_) > cfg_.median_change_threshold) {
                old_p50   = prev_p50_;
                new_p50   = cur_p50;
                fire_median = true;
                prev_p50_ = cur_p50;
                median_events_.fetch_add(1, std::memory_order_relaxed);
            }
            if (std::abs(cur_skew - prev_skew_) > cfg_.skew_threshold) {
                new_skew   = cur_skew;
                fire_skew  = true;
                prev_skew_ = cur_skew;
            }
        }
    }

    if (fire_median && median_cb) median_cb(old_p50, new_p50);
    if (fire_skew   && skew_cb)   skew_cb(new_skew);
}

std::string BiasRollingQuantileTracker::to_stats_json() const {
    std::lock_guard<std::mutex> lk(mutex_);
    std::ostringstream ss;
    ss << std::fixed << std::setprecision(6);
    ss << "{"
       << "\"p10\":"               << p10_.load(std::memory_order_relaxed) << ","
       << "\"p25\":"               << p25_.load(std::memory_order_relaxed) << ","
       << "\"p50\":"               << p50_.load(std::memory_order_relaxed) << ","
       << "\"p75\":"               << p75_.load(std::memory_order_relaxed) << ","
       << "\"p90\":"               << p90_.load(std::memory_order_relaxed) << ","
       << "\"skew_ratio\":"        << skew_.load(std::memory_order_relaxed) << ","
       << "\"median_shift_events\":" << median_events_.load(std::memory_order_relaxed) << ","
       << "\"total_records\":"     << total_.load(std::memory_order_relaxed)
       << "}";
    return ss.str();
}

void BiasRollingQuantileTracker::reset() {
    std::lock_guard<std::mutex> lk(mutex_);
    std::fill(win_buf_.begin(), win_buf_.end(), 0.0);
    win_head_ = win_fill_ = 0;
    seeded_    = false;
    prev_p50_  = 0.0;
    prev_skew_ = 1.0;
    p10_.store(0.0, std::memory_order_relaxed);
    p25_.store(0.0, std::memory_order_relaxed);
    p50_.store(0.0, std::memory_order_relaxed);
    p75_.store(0.0, std::memory_order_relaxed);
    p90_.store(0.0, std::memory_order_relaxed);
    skew_.store(1.0, std::memory_order_relaxed);
    median_events_.store(0, std::memory_order_relaxed);
    total_.store(0, std::memory_order_relaxed);
}

void BiasRollingQuantileTracker::update_config(const Config& cfg) {
    std::lock_guard<std::mutex> lk(mutex_);
    cfg_ = cfg;
    int w = std::max(8, cfg_.window);
    win_buf_.assign(static_cast<std::size_t>(w), 0.0);
    win_head_ = win_fill_ = 0;
    seeded_   = false;
    prev_p50_ = 0.0;
    prev_skew_= 1.0;
}

} // namespace llmquant
