#include "SignalBollingerBands.h"
#include <cmath>
#include <numeric>
#include <sstream>
#include <iomanip>

namespace llmquant {

SignalBollingerBands::SignalBollingerBands(Config cfg)
    : cfg_(std::move(cfg))
    , win_buf_(cfg_.window, 0.0)
{}

void SignalBollingerBands::record(double bias) {
    double new_mid{};
    double new_up{};
    double new_lo{};
    double new_pctb{0.5};
    bool touch_up  = false;
    bool touch_lo  = false;
    bool squeeze   = false;

    {
        std::lock_guard<std::mutex> lk(mutex_);
        win_buf_[win_head_] = bias;
        win_head_ = (win_head_ + 1) % cfg_.window;
        if (win_fill_ < cfg_.window) ++win_fill_;

        if (win_fill_ >= cfg_.min_samples) {
            int n = win_fill_;
            double sum = 0.0;
            for (int i = 0; i < n; ++i) sum += win_buf_[i];
            new_mid = sum / n;

            double var = 0.0;
            for (int i = 0; i < n; ++i) {
                double d = win_buf_[i] - new_mid;
                var += d * d;
            }
            double sigma = std::sqrt(var / n);

            new_up = new_mid + cfg_.k * sigma;
            new_lo = new_mid - cfg_.k * sigma;

            double bw = new_up - new_lo;
            new_pctb = (bw > 1e-12) ? (bias - new_lo) / bw : 0.5;

            if (bias > new_up) {
                touch_up = true;
                upper_events_.fetch_add(1, std::memory_order_relaxed);
            }
            if (bias < new_lo) {
                touch_lo = true;
                lower_events_.fetch_add(1, std::memory_order_relaxed);
            }
            if (bw < cfg_.squeeze_threshold) {
                squeeze = true;
                squeeze_events_.fetch_add(1, std::memory_order_relaxed);
            }
        }
    }

    mid_.store(new_mid, std::memory_order_relaxed);
    up_.store(new_up, std::memory_order_relaxed);
    lo_.store(new_lo, std::memory_order_relaxed);
    pct_b_.store(new_pctb, std::memory_order_relaxed);
    total_.fetch_add(1, std::memory_order_relaxed);

    if (touch_up  && cfg_.on_upper_touch) cfg_.on_upper_touch(bias);
    if (touch_lo  && cfg_.on_lower_touch) cfg_.on_lower_touch(bias);
    if (squeeze   && cfg_.on_squeeze)     cfg_.on_squeeze(new_up - new_lo);
}

std::string SignalBollingerBands::to_stats_json() const {
    std::lock_guard<std::mutex> lk(mutex_);
    std::ostringstream ss;
    ss << std::fixed << std::setprecision(4);
    ss << "{\"middle\":"  << mid_.load(std::memory_order_relaxed)
       << ",\"upper\":"   << up_.load(std::memory_order_relaxed)
       << ",\"lower\":"   << lo_.load(std::memory_order_relaxed)
       << ",\"pct_b\":"   << pct_b_.load(std::memory_order_relaxed)
       << ",\"upper_events\":" << upper_events_.load(std::memory_order_relaxed)
       << ",\"lower_events\":" << lower_events_.load(std::memory_order_relaxed)
       << ",\"squeeze_events\":" << squeeze_events_.load(std::memory_order_relaxed)
       << ",\"total_records\":" << total_.load(std::memory_order_relaxed)
       << "}";
    return ss.str();
}

void SignalBollingerBands::reset() {
    std::lock_guard<std::mutex> lk(mutex_);
    std::fill(win_buf_.begin(), win_buf_.end(), 0.0);
    win_head_ = 0; win_fill_ = 0;
    mid_.store(0.0, std::memory_order_relaxed);
    up_.store(0.0, std::memory_order_relaxed);
    lo_.store(0.0, std::memory_order_relaxed);
    pct_b_.store(0.5, std::memory_order_relaxed);
    upper_events_.store(0, std::memory_order_relaxed);
    lower_events_.store(0, std::memory_order_relaxed);
    squeeze_events_.store(0, std::memory_order_relaxed);
    total_.store(0, std::memory_order_relaxed);
}

void SignalBollingerBands::update_config(const Config& cfg) {
    std::lock_guard<std::mutex> lk(mutex_);
    cfg_ = cfg;
    win_buf_.assign(cfg_.window, 0.0);
    win_head_ = 0; win_fill_ = 0;
}

} // namespace llmquant
