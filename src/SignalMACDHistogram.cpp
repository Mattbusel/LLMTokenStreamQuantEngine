#include "SignalMACDHistogram.h"
#include <cmath>
#include <sstream>
#include <iomanip>

namespace llmquant {

SignalMACDHistogram::SignalMACDHistogram(Config cfg)
    : cfg_(std::move(cfg))
{}

void SignalMACDHistogram::record(double bias) {
    double new_fast = 0.0, new_slow = 0.0, new_macd = 0.0, new_sig = 0.0, new_hist = 0.0;
    bool cross_ev = false, div_ev = false;

    {
        std::lock_guard<std::mutex> lk(mutex_);

        if (!seeded_) {
            ema_fast_ = ema_slow_ = ema_signal_ = bias;
            seeded_ = true;
        } else {
            ema_fast_   += cfg_.alpha_fast   * (bias - ema_fast_);
            ema_slow_   += cfg_.alpha_slow   * (bias - ema_slow_);
            double macd  = ema_fast_ - ema_slow_;
            ema_signal_ += cfg_.alpha_signal * (macd - ema_signal_);
            new_fast = ema_fast_; new_slow = ema_slow_;
            new_macd = macd;
            new_sig  = ema_signal_;
            new_hist = macd - ema_signal_;

            if (total_.load(std::memory_order_relaxed) + 1 >= static_cast<std::size_t>(cfg_.min_samples)) {
                bool pos = (new_hist >= 0.0);
                if (pos != prev_pos_) {
                    cross_ev = true;
                    zero_cross_.fetch_add(1, std::memory_order_relaxed);
                }
                prev_pos_ = pos;

                if (std::fabs(new_hist) > cfg_.divergence_threshold) {
                    div_ev = true;
                    divergence_.fetch_add(1, std::memory_order_relaxed);
                }
            }
        }
    }

    fast_ema_.store(new_fast, std::memory_order_relaxed);
    slow_ema_.store(new_slow, std::memory_order_relaxed);
    macd_.store(new_macd, std::memory_order_relaxed);
    signal_.store(new_sig, std::memory_order_relaxed);
    hist_.store(new_hist, std::memory_order_relaxed);
    total_.fetch_add(1, std::memory_order_relaxed);

    if (cross_ev && cfg_.on_zero_cross) cfg_.on_zero_cross(new_hist);
    if (div_ev   && cfg_.on_divergence) cfg_.on_divergence(new_hist);
}

double SignalMACDHistogram::fast_ema()    const noexcept { return fast_ema_.load(std::memory_order_relaxed); }
double SignalMACDHistogram::slow_ema()    const noexcept { return slow_ema_.load(std::memory_order_relaxed); }
double SignalMACDHistogram::macd()        const noexcept { return macd_.load(std::memory_order_relaxed); }
double SignalMACDHistogram::signal_line() const noexcept { return signal_.load(std::memory_order_relaxed); }
double SignalMACDHistogram::histogram()   const noexcept { return hist_.load(std::memory_order_relaxed); }

std::size_t SignalMACDHistogram::zero_cross_events() const noexcept { return zero_cross_.load(std::memory_order_relaxed); }
std::size_t SignalMACDHistogram::divergence_events() const noexcept { return divergence_.load(std::memory_order_relaxed); }
std::size_t SignalMACDHistogram::total_records()     const noexcept { return total_.load(std::memory_order_relaxed); }

std::string SignalMACDHistogram::to_stats_json() const {
    std::lock_guard<std::mutex> lk(mutex_);
    std::ostringstream ss;
    ss << std::fixed << std::setprecision(6);
    ss << "{\"macd\":" << macd_.load(std::memory_order_relaxed)
       << ",\"signal_line\":" << signal_.load(std::memory_order_relaxed)
       << ",\"histogram\":" << hist_.load(std::memory_order_relaxed)
       << ",\"zero_cross_events\":" << zero_cross_.load(std::memory_order_relaxed)
       << ",\"divergence_events\":" << divergence_.load(std::memory_order_relaxed)
       << ",\"total_records\":" << total_.load(std::memory_order_relaxed)
       << "}";
    return ss.str();
}

void SignalMACDHistogram::reset() {
    std::lock_guard<std::mutex> lk(mutex_);
    ema_fast_ = 0.0; ema_slow_ = 0.0; ema_signal_ = 0.0;
    seeded_ = false; prev_pos_ = false;
    fast_ema_.store(0.0, std::memory_order_relaxed);
    slow_ema_.store(0.0, std::memory_order_relaxed);
    macd_.store(0.0, std::memory_order_relaxed);
    signal_.store(0.0, std::memory_order_relaxed);
    hist_.store(0.0, std::memory_order_relaxed);
    zero_cross_.store(0, std::memory_order_relaxed);
    divergence_.store(0, std::memory_order_relaxed);
    total_.store(0, std::memory_order_relaxed);
}

void SignalMACDHistogram::update_config(const Config& cfg) {
    std::lock_guard<std::mutex> lk(mutex_);
    cfg_ = cfg;
}

} // namespace llmquant
