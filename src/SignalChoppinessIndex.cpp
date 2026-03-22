#include "SignalChoppinessIndex.h"
#include <cmath>
#include <sstream>
#include <iomanip>
#include <algorithm>

namespace llmquant {

SignalChoppinessIndex::SignalChoppinessIndex(Config cfg)
    : cfg_(std::move(cfg))
    , win_buf_(cfg_.window, 0.0)
{}

void SignalChoppinessIndex::record(double bias) {
    double new_ci{61.8}; // default neutral
    bool trending = false;
    bool choppy   = false;

    {
        std::lock_guard<std::mutex> lk(mutex_);

        win_buf_[win_head_] = bias;
        win_head_ = (win_head_ + 1) % cfg_.window;
        if (win_fill_ < cfg_.window) ++win_fill_;

        if (win_fill_ >= cfg_.min_samples) {
            int n = win_fill_;
            double range_max = win_buf_[0], range_min = win_buf_[0];
            double atr_sum = 0.0;
            for (int i = 0; i < n; ++i) {
                range_max = std::max(range_max, win_buf_[i]);
                range_min = std::min(range_min, win_buf_[i]);
            }
            // ATR for bias: |x_t - x_{t-1}|
            for (int i = 1; i < n; ++i) {
                atr_sum += std::fabs(win_buf_[i] - win_buf_[i-1]);
            }

            double range = range_max - range_min;
            if (range > 1e-12 && atr_sum > 1e-12 && n > 1) {
                new_ci = 100.0 * std::log10(atr_sum / range) / std::log10(static_cast<double>(n));
                // Clamp to [0, 100]
                if (new_ci < 0.0) new_ci = 0.0;
                if (new_ci > 100.0) new_ci = 100.0;
            }

            bool t_now = (new_ci < cfg_.trending_threshold);
            bool c_now = (new_ci > cfg_.choppy_threshold);
            if (t_now && !prev_trending_) {
                trending = true;
                trend_events_.fetch_add(1, std::memory_order_relaxed);
            }
            if (c_now && !prev_choppy_) {
                choppy = true;
                choppy_events_.fetch_add(1, std::memory_order_relaxed);
            }
            prev_trending_ = t_now;
            prev_choppy_   = c_now;
        }
    }

    ci_.store(new_ci, std::memory_order_relaxed);
    total_.fetch_add(1, std::memory_order_relaxed);

    if (trending && cfg_.on_trending) cfg_.on_trending(new_ci);
    if (choppy   && cfg_.on_choppy)   cfg_.on_choppy(new_ci);
}

std::string SignalChoppinessIndex::to_stats_json() const {
    std::lock_guard<std::mutex> lk(mutex_);
    std::ostringstream ss;
    ss << std::fixed << std::setprecision(4);
    ss << "{\"choppiness\":" << ci_.load(std::memory_order_relaxed)
       << ",\"trending_events\":" << trend_events_.load(std::memory_order_relaxed)
       << ",\"choppy_events\":" << choppy_events_.load(std::memory_order_relaxed)
       << ",\"total_records\":" << total_.load(std::memory_order_relaxed)
       << "}";
    return ss.str();
}

void SignalChoppinessIndex::reset() {
    std::lock_guard<std::mutex> lk(mutex_);
    std::fill(win_buf_.begin(), win_buf_.end(), 0.0);
    win_head_  = 0; win_fill_ = 0;
    seeded_    = false;
    prev_bias_ = 0.0;
    prev_trending_ = false;
    prev_choppy_   = false;
    ci_.store(61.8, std::memory_order_relaxed);
    trend_events_.store(0, std::memory_order_relaxed);
    choppy_events_.store(0, std::memory_order_relaxed);
    total_.store(0, std::memory_order_relaxed);
}

void SignalChoppinessIndex::update_config(const Config& cfg) {
    std::lock_guard<std::mutex> lk(mutex_);
    cfg_ = cfg;
    win_buf_.assign(cfg_.window, 0.0);
    win_head_ = 0; win_fill_ = 0;
}

} // namespace llmquant
