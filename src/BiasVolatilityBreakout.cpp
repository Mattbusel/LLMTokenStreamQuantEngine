#include "BiasVolatilityBreakout.h"
#include <cmath>
#include <sstream>
#include <iomanip>

namespace llmquant {

// static helper
double BiasVolatilityBreakout::rolling_std(const std::vector<double>& buf, int head, int fill) {
    if (fill < 2) return 0.0;
    int n = fill;
    int sz = static_cast<int>(buf.size());
    double sum = 0.0;
    for (int i = 0; i < n; ++i) sum += buf[(head - n + i + sz) % sz];
    double mean = sum / n;
    double var  = 0.0;
    for (int i = 0; i < n; ++i) {
        double d = buf[(head - n + i + sz) % sz] - mean;
        var += d * d;
    }
    return std::sqrt(var / n);
}

BiasVolatilityBreakout::BiasVolatilityBreakout(Config cfg)
    : cfg_(std::move(cfg))
    , short_buf_(cfg_.short_window, 0.0)
    , long_buf_(cfg_.long_window, 0.0)
{}

void BiasVolatilityBreakout::record(double bias) {
    double new_sv = 0.0, new_lv = 0.0, new_ratio = 1.0;
    bool exp_ev = false, con_ev = false;

    {
        std::lock_guard<std::mutex> lk(mutex_);

        short_buf_[s_head_] = bias;
        s_head_ = (s_head_ + 1) % cfg_.short_window;
        if (s_fill_ < cfg_.short_window) ++s_fill_;

        long_buf_[l_head_] = bias;
        l_head_ = (l_head_ + 1) % cfg_.long_window;
        if (l_fill_ < cfg_.long_window) ++l_fill_;

        if (l_fill_ >= cfg_.min_samples) {
            new_sv = rolling_std(short_buf_, s_head_, s_fill_);
            new_lv = rolling_std(long_buf_,  l_head_, l_fill_);
            new_ratio = (new_lv > 1e-12) ? new_sv / new_lv : 1.0;

            bool expansion   = (new_ratio > cfg_.breakout_ratio);
            bool contraction = (new_ratio < cfg_.contraction_ratio);

            if (expansion && !prev_expansion_) {
                exp_ev = true;
                expansion_events_.fetch_add(1, std::memory_order_relaxed);
            }
            if (contraction && !prev_contraction_) {
                con_ev = true;
                contraction_events_.fetch_add(1, std::memory_order_relaxed);
            }
            prev_expansion_   = expansion;
            prev_contraction_ = contraction;
        }
    }

    short_vol_.store(new_sv, std::memory_order_relaxed);
    long_vol_.store(new_lv, std::memory_order_relaxed);
    vol_ratio_.store(new_ratio, std::memory_order_relaxed);
    total_.fetch_add(1, std::memory_order_relaxed);

    if (exp_ev && cfg_.on_expansion)   cfg_.on_expansion(new_ratio);
    if (con_ev && cfg_.on_contraction) cfg_.on_contraction(new_ratio);
}

double BiasVolatilityBreakout::short_vol()  const noexcept { return short_vol_.load(std::memory_order_relaxed); }
double BiasVolatilityBreakout::long_vol()   const noexcept { return long_vol_.load(std::memory_order_relaxed); }
double BiasVolatilityBreakout::vol_ratio()  const noexcept { return vol_ratio_.load(std::memory_order_relaxed); }

std::size_t BiasVolatilityBreakout::expansion_events()   const noexcept { return expansion_events_.load(std::memory_order_relaxed); }
std::size_t BiasVolatilityBreakout::contraction_events() const noexcept { return contraction_events_.load(std::memory_order_relaxed); }
std::size_t BiasVolatilityBreakout::total_records()      const noexcept { return total_.load(std::memory_order_relaxed); }

std::string BiasVolatilityBreakout::to_stats_json() const {
    std::lock_guard<std::mutex> lk(mutex_);
    std::ostringstream ss;
    ss << std::fixed << std::setprecision(4);
    ss << "{\"short_vol\":" << short_vol_.load(std::memory_order_relaxed)
       << ",\"long_vol\":" << long_vol_.load(std::memory_order_relaxed)
       << ",\"vol_ratio\":" << vol_ratio_.load(std::memory_order_relaxed)
       << ",\"expansion_events\":" << expansion_events_.load(std::memory_order_relaxed)
       << ",\"contraction_events\":" << contraction_events_.load(std::memory_order_relaxed)
       << ",\"total_records\":" << total_.load(std::memory_order_relaxed)
       << "}";
    return ss.str();
}

void BiasVolatilityBreakout::reset() {
    std::lock_guard<std::mutex> lk(mutex_);
    std::fill(short_buf_.begin(), short_buf_.end(), 0.0);
    std::fill(long_buf_.begin(), long_buf_.end(), 0.0);
    s_head_ = 0; s_fill_ = 0; l_head_ = 0; l_fill_ = 0;
    prev_expansion_ = false; prev_contraction_ = false;
    short_vol_.store(0.0, std::memory_order_relaxed);
    long_vol_.store(0.0, std::memory_order_relaxed);
    vol_ratio_.store(1.0, std::memory_order_relaxed);
    expansion_events_.store(0, std::memory_order_relaxed);
    contraction_events_.store(0, std::memory_order_relaxed);
    total_.store(0, std::memory_order_relaxed);
}

void BiasVolatilityBreakout::update_config(const Config& cfg) {
    std::lock_guard<std::mutex> lk(mutex_);
    cfg_ = cfg;
    short_buf_.assign(cfg_.short_window, 0.0);
    long_buf_.assign(cfg_.long_window, 0.0);
    s_head_ = 0; s_fill_ = 0; l_head_ = 0; l_fill_ = 0;
}

} // namespace llmquant
