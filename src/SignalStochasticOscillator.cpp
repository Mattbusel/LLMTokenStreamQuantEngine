#include "SignalStochasticOscillator.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <sstream>

namespace llmquant {

static constexpr double kNaN = std::numeric_limits<double>::quiet_NaN();

SignalStochasticOscillator::SignalStochasticOscillator(Config cfg)
    : cfg_(std::move(cfg))
{
    int k = std::max(2, cfg_.k_period);
    int d = std::max(1, cfg_.d_period);
    cfg_.k_period = k;
    cfg_.d_period = d;
    if (cfg_.min_samples <= 0) cfg_.min_samples = k;
    buf_.resize(static_cast<std::size_t>(k), 0.0);
    d_buf_.resize(static_cast<std::size_t>(d), 0.0);
}

void SignalStochasticOscillator::record(double x) {
    bool   fire_ob = false, fire_os = false;
    bool   fire_bull = false, fire_bear = false;
    double k_val = kNaN, d_val = kNaN;
    std::function<void(double)>         ob_cb, os_cb;
    std::function<void(double, double)> bull_cb, bear_cb;

    {
        std::lock_guard<std::mutex> lk(mutex_);
        ob_cb   = cfg_.on_overbought;
        os_cb   = cfg_.on_oversold;
        bull_cb = cfg_.on_kd_bullish_cross;
        bear_cb = cfg_.on_kd_bearish_cross;

        total_.fetch_add(1, std::memory_order_relaxed);

        int k = cfg_.k_period;
        buf_[static_cast<std::size_t>(head_)] = x;
        head_ = (head_ + 1) % k;
        if (fill_ < k) ++fill_;

        if (fill_ < cfg_.min_samples) {
            // Not enough data yet — leave atomics at NaN/false
            return;
        }

        // Compute H_n and L_n over the k-period window
        double h = buf_[0], l = buf_[0];
        int n = fill_;
        for (int i = 1; i < n; ++i) {
            if (buf_[static_cast<std::size_t>(i)] > h) h = buf_[static_cast<std::size_t>(i)];
            if (buf_[static_cast<std::size_t>(i)] < l) l = buf_[static_cast<std::size_t>(i)];
        }

        double range = h - l;
        double k_pct = (range > 1e-12) ? 100.0 * (x - l) / range : 50.0;
        k_.store(k_pct, std::memory_order_relaxed);
        k_val = k_pct;

        // %D = SMA(%K) over d_period
        int d = cfg_.d_period;
        d_sum_ -= d_buf_[static_cast<std::size_t>(d_head_)];
        d_buf_[static_cast<std::size_t>(d_head_)] = k_pct;
        d_sum_ += k_pct;
        d_head_ = (d_head_ + 1) % d;
        if (d_fill_ < d) ++d_fill_;

        bool d_ready = (d_fill_ >= d);
        if (d_ready) {
            d_val = d_sum_ / static_cast<double>(d);
            d_.store(d_val, std::memory_order_relaxed);
            d_valid_ = true;
        }

        // Overbought / oversold edge detection
        bool now_ob = (k_pct >= cfg_.overbought_threshold);
        bool now_os = (k_pct <= cfg_.oversold_threshold);

        if (now_ob && !prev_overbought_) {
            fire_ob = true;
            ob_events_.fetch_add(1, std::memory_order_relaxed);
        }
        if (now_os && !prev_oversold_) {
            fire_os = true;
            os_events_.fetch_add(1, std::memory_order_relaxed);
        }
        prev_overbought_ = now_ob;
        prev_oversold_   = now_os;
        overbought_.store(now_ob, std::memory_order_relaxed);
        oversold_.store(now_os, std::memory_order_relaxed);

        // %K/%D crossover detection (only when D is valid)
        if (d_valid_) {
            bool now_k_above_d = (k_val > d_val);
            if (now_k_above_d && !prev_k_above_d_) {
                fire_bull = true;
            } else if (!now_k_above_d && prev_k_above_d_) {
                fire_bear = true;
            }
            prev_k_above_d_ = now_k_above_d;
        }
    }

    if (fire_ob   && ob_cb)   try { ob_cb(k_val); }          catch (...) {}
    if (fire_os   && os_cb)   try { os_cb(k_val); }          catch (...) {}
    if (fire_bull && bull_cb) try { bull_cb(k_val, d_val); }  catch (...) {}
    if (fire_bear && bear_cb) try { bear_cb(k_val, d_val); }  catch (...) {}
}

std::string SignalStochasticOscillator::to_stats_json() const {
    double kv = k_.load(std::memory_order_relaxed);
    double dv = d_.load(std::memory_order_relaxed);
    bool   ob = overbought_.load(std::memory_order_relaxed);
    bool   os = oversold_.load(std::memory_order_relaxed);
    uint64_t tot = total_.load(std::memory_order_relaxed);
    uint64_t obe = ob_events_.load(std::memory_order_relaxed);
    uint64_t ose = os_events_.load(std::memory_order_relaxed);

    auto fmt = [](double v) -> std::string {
        if (!std::isfinite(v)) return "null";
        char tmp[32]; std::snprintf(tmp, sizeof(tmp), "%.4f", v); return tmp;
    };
    char buf[256];
    std::snprintf(buf, sizeof(buf),
        "{\"k_pct\":%s,\"d_pct\":%s,\"is_overbought\":%s,\"is_oversold\":%s,"
        "\"overbought_events\":%llu,\"oversold_events\":%llu,\"total_records\":%llu}",
        fmt(kv).c_str(), fmt(dv).c_str(),
        ob ? "true" : "false", os ? "true" : "false",
        static_cast<unsigned long long>(obe),
        static_cast<unsigned long long>(ose),
        static_cast<unsigned long long>(tot));
    return buf;
}

void SignalStochasticOscillator::reset() {
    std::lock_guard<std::mutex> lk(mutex_);
    std::fill(buf_.begin(), buf_.end(), 0.0);
    std::fill(d_buf_.begin(), d_buf_.end(), 0.0);
    head_ = fill_ = 0;
    d_head_ = d_fill_ = 0;
    d_sum_ = 0.0;
    prev_overbought_ = prev_oversold_ = prev_k_above_d_ = false;
    d_valid_ = false;
    k_.store(kNaN, std::memory_order_relaxed);
    d_.store(kNaN, std::memory_order_relaxed);
    overbought_.store(false, std::memory_order_relaxed);
    oversold_.store(false, std::memory_order_relaxed);
    total_.store(0, std::memory_order_relaxed);
    ob_events_.store(0, std::memory_order_relaxed);
    os_events_.store(0, std::memory_order_relaxed);
}

void SignalStochasticOscillator::update_config(const Config& cfg) {
    std::lock_guard<std::mutex> lk(mutex_);
    cfg_ = cfg;
    int k = std::max(2, cfg_.k_period);
    int d = std::max(1, cfg_.d_period);
    cfg_.k_period = k;
    cfg_.d_period = d;
    if (cfg_.min_samples <= 0) cfg_.min_samples = k;
    buf_.assign(static_cast<std::size_t>(k), 0.0);
    d_buf_.assign(static_cast<std::size_t>(d), 0.0);
    head_ = fill_ = 0;
    d_head_ = d_fill_ = 0;
    d_sum_ = 0.0;
    prev_overbought_ = prev_oversold_ = prev_k_above_d_ = false;
    d_valid_ = false;
    k_.store(kNaN, std::memory_order_relaxed);
    d_.store(kNaN, std::memory_order_relaxed);
    overbought_.store(false, std::memory_order_relaxed);
    oversold_.store(false, std::memory_order_relaxed);
    total_.store(0, std::memory_order_relaxed);
    ob_events_.store(0, std::memory_order_relaxed);
    os_events_.store(0, std::memory_order_relaxed);
}

} // namespace llmquant
