#include "SignalChandeOscillator.h"
#include <algorithm>
#include <cmath>
#include <sstream>
#include <iomanip>

namespace llmquant {

SignalChandeOscillator::SignalChandeOscillator(Config cfg)
    : cfg_(std::move(cfg))
{
    int w = std::max(2, cfg_.window);
    win_buf_.assign(static_cast<std::size_t>(w + 1), 0.0);
}

void SignalChandeOscillator::record(double bias) {
    bool   fire_ob = false, fire_os = false;
    double new_cmo = 0.0;
    std::function<void(double)> ob_cb, os_cb;

    {
        std::lock_guard<std::mutex> lk(mutex_);
        ob_cb = cfg_.on_overbought;
        os_cb = cfg_.on_oversold;

        int w = static_cast<int>(win_buf_.size());
        if (win_fill_ < w) ++win_fill_;
        win_buf_[static_cast<std::size_t>(win_head_)] = bias;
        win_head_ = (win_head_ + 1) % w;

        // Need at least min_samples+1 values to compute differences
        if (win_fill_ >= std::max(2, cfg_.min_samples + 1)) {
            double sum_up = 0.0, sum_dn = 0.0;
            int n = std::min(win_fill_, cfg_.window + 1);
            // Iterate over pairs of consecutive values
            for (int i = 1; i < n; ++i) {
                int cur_idx  = (win_head_ - i     + w * 2) % w;
                int prev_idx = (win_head_ - i - 1 + w * 2) % w;
                double diff = win_buf_[static_cast<std::size_t>(cur_idx)]
                            - win_buf_[static_cast<std::size_t>(prev_idx)];
                if (diff > 0.0) sum_up += diff;
                else            sum_dn -= diff; // sum_dn accumulates absolute values
            }
            double denom = sum_up + sum_dn;
            new_cmo = (denom > 1e-12) ? 100.0 * (sum_up - sum_dn) / denom : 0.0;

            bool cur_ob = new_cmo >  cfg_.overbought_threshold;
            bool cur_os = new_cmo <  cfg_.oversold_threshold;
            bool was_ob = overbought_.load(std::memory_order_relaxed);
            bool was_os = oversold_.load(std::memory_order_relaxed);

            if (cur_ob && !was_ob) {
                fire_ob = true;
                ob_events_.fetch_add(1, std::memory_order_relaxed);
            }
            if (cur_os && !was_os) {
                fire_os = true;
                os_events_.fetch_add(1, std::memory_order_relaxed);
            }
            overbought_.store(cur_ob, std::memory_order_relaxed);
            oversold_.store(cur_os,   std::memory_order_relaxed);
            cmo_.store(new_cmo,       std::memory_order_relaxed);
        }
    }

    total_.fetch_add(1, std::memory_order_relaxed);
    if (fire_ob && ob_cb) ob_cb(new_cmo);
    if (fire_os && os_cb) os_cb(new_cmo);
}

double SignalChandeOscillator::value()         const noexcept { return cmo_.load(std::memory_order_relaxed); }
bool   SignalChandeOscillator::is_overbought() const noexcept { return overbought_.load(std::memory_order_relaxed); }
bool   SignalChandeOscillator::is_oversold()   const noexcept { return oversold_.load(std::memory_order_relaxed); }

std::size_t SignalChandeOscillator::overbought_events() const noexcept { return ob_events_.load(std::memory_order_relaxed); }
std::size_t SignalChandeOscillator::oversold_events()   const noexcept { return os_events_.load(std::memory_order_relaxed); }
std::size_t SignalChandeOscillator::total_records()     const noexcept { return total_.load(std::memory_order_relaxed); }

std::string SignalChandeOscillator::to_stats_json() const {
    std::ostringstream ss;
    ss << std::fixed << std::setprecision(4);
    ss << "{\"cmo\":"              << cmo_.load(std::memory_order_relaxed)
       << ",\"is_overbought\":"    << (overbought_.load(std::memory_order_relaxed) ? "true" : "false")
       << ",\"is_oversold\":"      << (oversold_.load(std::memory_order_relaxed)   ? "true" : "false")
       << ",\"overbought_events\":" << ob_events_.load(std::memory_order_relaxed)
       << ",\"oversold_events\":"  << os_events_.load(std::memory_order_relaxed)
       << ",\"total_records\":"    << total_.load(std::memory_order_relaxed)
       << "}";
    return ss.str();
}

void SignalChandeOscillator::reset() {
    std::lock_guard<std::mutex> lk(mutex_);
    std::fill(win_buf_.begin(), win_buf_.end(), 0.0);
    win_head_ = win_fill_ = 0;
    cmo_.store(0.0,    std::memory_order_relaxed);
    overbought_.store(false, std::memory_order_relaxed);
    oversold_.store(false,   std::memory_order_relaxed);
    ob_events_.store(0,      std::memory_order_relaxed);
    os_events_.store(0,      std::memory_order_relaxed);
    total_.store(0,          std::memory_order_relaxed);
}

void SignalChandeOscillator::update_config(const Config& cfg) {
    std::lock_guard<std::mutex> lk(mutex_);
    cfg_ = cfg;
    int w = std::max(2, cfg_.window);
    win_buf_.assign(static_cast<std::size_t>(w + 1), 0.0);
    win_head_ = win_fill_ = 0;
    cmo_.store(0.0,    std::memory_order_relaxed);
    overbought_.store(false, std::memory_order_relaxed);
    oversold_.store(false,   std::memory_order_relaxed);
    ob_events_.store(0,      std::memory_order_relaxed);
    os_events_.store(0,      std::memory_order_relaxed);
    total_.store(0,          std::memory_order_relaxed);
}

} // namespace llmquant
