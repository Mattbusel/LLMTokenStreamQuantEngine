#include "SignalWilliamsR.h"
#include <algorithm>
#include <sstream>
#include <iomanip>

namespace llmquant {

SignalWilliamsR::SignalWilliamsR(Config cfg)
    : cfg_(std::move(cfg))
    , buf_(cfg_.period, 0.0)
{}

void SignalWilliamsR::record(double bias) {
    double new_wr = -50.0;
    bool ob_ev = false, os_ev = false;
    bool new_ob = false, new_os = false;

    {
        std::lock_guard<std::mutex> lk(mutex_);

        buf_[head_] = bias;
        head_ = (head_ + 1) % cfg_.period;
        if (fill_ < cfg_.period) ++fill_;

        if (fill_ >= cfg_.min_samples) {
            int n = fill_;
            int sz = cfg_.period;
            double hh = buf_[0], ll = buf_[0];
            for (int i = 0; i < n; ++i) {
                double v = buf_[(head_ - n + i + sz) % sz];
                if (v > hh) hh = v;
                if (v < ll) ll = v;
            }
            double range = hh - ll;
            if (range > 1e-12) {
                new_wr = -100.0 * (hh - bias) / range;
            } else {
                new_wr = -50.0;
            }

            new_ob = (new_wr > cfg_.overbought_threshold);
            new_os = (new_wr < cfg_.oversold_threshold);

            if (new_ob && !prev_overbought_) {
                ob_ev = true;
                ob_events_.fetch_add(1, std::memory_order_relaxed);
            }
            if (new_os && !prev_oversold_) {
                os_ev = true;
                os_events_.fetch_add(1, std::memory_order_relaxed);
            }
            prev_overbought_ = new_ob;
            prev_oversold_   = new_os;
        }
    }

    wr_.store(new_wr, std::memory_order_relaxed);
    is_overbought_.store(new_ob, std::memory_order_relaxed);
    is_oversold_.store(new_os, std::memory_order_relaxed);
    total_.fetch_add(1, std::memory_order_relaxed);

    if (ob_ev && cfg_.on_overbought) cfg_.on_overbought(new_wr);
    if (os_ev && cfg_.on_oversold)   cfg_.on_oversold(new_wr);
}

double SignalWilliamsR::williams_r()    const noexcept { return wr_.load(std::memory_order_relaxed); }
bool   SignalWilliamsR::is_overbought() const noexcept { return is_overbought_.load(std::memory_order_relaxed); }
bool   SignalWilliamsR::is_oversold()   const noexcept { return is_oversold_.load(std::memory_order_relaxed); }

std::size_t SignalWilliamsR::overbought_events() const noexcept { return ob_events_.load(std::memory_order_relaxed); }
std::size_t SignalWilliamsR::oversold_events()   const noexcept { return os_events_.load(std::memory_order_relaxed); }
std::size_t SignalWilliamsR::total_records()     const noexcept { return total_.load(std::memory_order_relaxed); }

std::string SignalWilliamsR::to_stats_json() const {
    std::lock_guard<std::mutex> lk(mutex_);
    std::ostringstream ss;
    ss << std::fixed << std::setprecision(2);
    ss << "{\"williams_r\":" << wr_.load(std::memory_order_relaxed)
       << ",\"is_overbought\":" << (is_overbought_.load(std::memory_order_relaxed) ? "true" : "false")
       << ",\"is_oversold\":" << (is_oversold_.load(std::memory_order_relaxed) ? "true" : "false")
       << ",\"overbought_events\":" << ob_events_.load(std::memory_order_relaxed)
       << ",\"oversold_events\":" << os_events_.load(std::memory_order_relaxed)
       << ",\"total_records\":" << total_.load(std::memory_order_relaxed)
       << "}";
    return ss.str();
}

void SignalWilliamsR::reset() {
    std::lock_guard<std::mutex> lk(mutex_);
    std::fill(buf_.begin(), buf_.end(), 0.0);
    head_ = 0; fill_ = 0; prev_overbought_ = false; prev_oversold_ = false;
    wr_.store(-50.0, std::memory_order_relaxed);
    is_overbought_.store(false, std::memory_order_relaxed);
    is_oversold_.store(false, std::memory_order_relaxed);
    ob_events_.store(0, std::memory_order_relaxed);
    os_events_.store(0, std::memory_order_relaxed);
    total_.store(0, std::memory_order_relaxed);
}

void SignalWilliamsR::update_config(const Config& cfg) {
    std::lock_guard<std::mutex> lk(mutex_);
    cfg_ = cfg;
    buf_.assign(cfg_.period, 0.0);
    head_ = 0; fill_ = 0;
}

} // namespace llmquant
