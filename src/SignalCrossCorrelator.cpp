#include "SignalCrossCorrelator.h"
#include <cmath>
#include <sstream>
#include <iomanip>
#include <numeric>

namespace llmquant {

SignalCrossCorrelator::SignalCrossCorrelator(Config cfg)
    : cfg_(std::move(cfg))
    , raw_buf_(cfg_.window, 0.0)
    , ema_buf_(cfg_.window, 0.0)
{}

void SignalCrossCorrelator::record(double bias) {
    double new_r0{};
    double new_r1{};
    int    new_dom{0};
    bool   changed = false;

    {
        std::lock_guard<std::mutex> lk(mutex_);

        // Update EMA
        ema_ = ema_ * (1.0 - cfg_.ema_alpha) + bias * cfg_.ema_alpha;

        // Store in ring buffer
        raw_buf_[buf_head_] = bias;
        ema_buf_[buf_head_] = ema_;
        buf_head_ = (buf_head_ + 1) % cfg_.window;
        if (buf_fill_ < cfg_.window) ++buf_fill_;

        if (buf_fill_ >= cfg_.min_samples) {
            int n = buf_fill_;
            // Compute means
            double raw_mean = 0.0, ema_mean = 0.0;
            for (int i = 0; i < n; ++i) {
                raw_mean += raw_buf_[i];
                ema_mean += ema_buf_[i];
            }
            raw_mean /= n;
            ema_mean /= n;

            // Compute variances
            double var_raw = 0.0, var_ema = 0.0;
            for (int i = 0; i < n; ++i) {
                double dr = raw_buf_[i] - raw_mean;
                double de = ema_buf_[i] - ema_mean;
                var_raw += dr * dr;
                var_ema += de * de;
            }
            double denom = std::sqrt(var_raw * var_ema) + 1e-10;

            // Compute R(τ) for τ in [0, max_lag]
            int max_lag = std::min(cfg_.max_lag, n - 1);
            double best_abs = -1.0;
            for (int lag = 0; lag <= max_lag; ++lag) {
                double cov = 0.0;
                for (int i = 0; i < n - lag; ++i) {
                    int raw_idx = (buf_head_ + i) % cfg_.window;
                    int ema_idx = (buf_head_ + i + lag) % cfg_.window;
                    if (i + lag < n) {
                        cov += (raw_buf_[raw_idx] - raw_mean)
                             * (ema_buf_[ema_idx] - ema_mean);
                    }
                }
                double r = cov / denom;
                if (lag == 0) new_r0 = r;
                if (lag == 1) new_r1 = r;
                if (std::fabs(r) > best_abs) { best_abs = std::fabs(r); new_dom = lag; }
            }

            if (new_dom != prev_dom_lag_ && prev_dom_lag_ >= 0) {
                changed = true;
                change_events_.fetch_add(1, std::memory_order_relaxed);
            }
            prev_dom_lag_ = new_dom;
        }
    }

    r0_.store(new_r0, std::memory_order_relaxed);
    r1_.store(new_r1, std::memory_order_relaxed);
    dom_lag_.store(new_dom, std::memory_order_relaxed);
    total_.fetch_add(1, std::memory_order_relaxed);

    if (changed && cfg_.on_dominant_lag_change) {
        cfg_.on_dominant_lag_change(prev_dom_lag_, new_dom, new_r0);
    }
}

std::string SignalCrossCorrelator::to_stats_json() const {
    std::lock_guard<std::mutex> lk(mutex_);
    std::ostringstream ss;
    ss << std::fixed << std::setprecision(4);
    ss << "{\"r_lag0\":" << r0_.load(std::memory_order_relaxed)
       << ",\"r_lag1\":" << r1_.load(std::memory_order_relaxed)
       << ",\"dominant_lag\":" << dom_lag_.load(std::memory_order_relaxed)
       << ",\"lag_change_events\":" << change_events_.load(std::memory_order_relaxed)
       << ",\"total_records\":" << total_.load(std::memory_order_relaxed)
       << "}";
    return ss.str();
}

void SignalCrossCorrelator::reset() {
    std::lock_guard<std::mutex> lk(mutex_);
    std::fill(raw_buf_.begin(), raw_buf_.end(), 0.0);
    std::fill(ema_buf_.begin(), ema_buf_.end(), 0.0);
    buf_head_     = 0;
    buf_fill_     = 0;
    ema_          = 0.0;
    prev_dom_lag_ = -1;
    r0_.store(0.0, std::memory_order_relaxed);
    r1_.store(0.0, std::memory_order_relaxed);
    dom_lag_.store(0, std::memory_order_relaxed);
    change_events_.store(0, std::memory_order_relaxed);
    total_.store(0, std::memory_order_relaxed);
}

void SignalCrossCorrelator::update_config(const Config& cfg) {
    std::lock_guard<std::mutex> lk(mutex_);
    cfg_ = cfg;
    raw_buf_.assign(cfg_.window, 0.0);
    ema_buf_.assign(cfg_.window, 0.0);
    buf_head_ = 0;
    buf_fill_ = 0;
}

} // namespace llmquant
