#include "TokenBurstIntensity.h"
#include <cmath>
#include <numeric>
#include <sstream>
#include <iomanip>

namespace llmquant {

TokenBurstIntensity::TokenBurstIntensity(Config cfg)
    : cfg_(std::move(cfg))
    , short_buf_(cfg_.short_window, 0.0)
    , long_buf_(cfg_.long_window, 0.0)
{}

double TokenBurstIntensity::window_variance(const std::vector<double>& buf, int fill, int sz) noexcept {
    if (fill < 2) return 0.0;
    double sum = 0.0;
    for (int i = 0; i < fill; ++i) sum += buf[i];
    double mean = sum / fill;
    double sq = 0.0;
    for (int i = 0; i < fill; ++i) { double d = buf[i] - mean; sq += d * d; }
    return sq / fill;
    (void)sz;
}

void TokenBurstIntensity::record(double bias) {
    double new_sv = 0.0, new_lv = 0.0, new_ratio = 0.0;
    bool burst_start = false, burst_end = false;
    bool new_burst = false;

    {
        std::lock_guard<std::mutex> lk(mutex_);

        short_buf_[short_head_] = bias;
        short_head_ = (short_head_ + 1) % cfg_.short_window;
        if (short_fill_ < cfg_.short_window) ++short_fill_;

        long_buf_[long_head_] = bias;
        long_head_ = (long_head_ + 1) % cfg_.long_window;
        if (long_fill_ < cfg_.long_window) ++long_fill_;

        if (total_.load(std::memory_order_relaxed) + 1 >= static_cast<uint64_t>(cfg_.min_samples)) {
            new_sv    = window_variance(short_buf_, short_fill_, cfg_.short_window);
            new_lv    = window_variance(long_buf_,  long_fill_,  cfg_.long_window);
            new_ratio = (new_lv > 1e-12) ? new_sv / new_lv : 0.0;
            new_burst = (new_ratio >= cfg_.burst_ratio);

            if (new_burst && !prev_bursting_) {
                burst_start = true;
                burst_ev_.fetch_add(1, std::memory_order_relaxed);
            } else if (!new_burst && prev_bursting_) {
                burst_end = true;
            }
            prev_bursting_ = new_burst;
        }
    }

    short_var_.store(new_sv,    std::memory_order_relaxed);
    long_var_.store(new_lv,     std::memory_order_relaxed);
    ratio_.store(new_ratio,     std::memory_order_relaxed);
    bursting_.store(new_burst,  std::memory_order_relaxed);
    total_.fetch_add(1, std::memory_order_relaxed);

    if (burst_start && cfg_.on_burst_start) cfg_.on_burst_start(new_ratio);
    if (burst_end   && cfg_.on_burst_end)   cfg_.on_burst_end(new_ratio);
}

double TokenBurstIntensity::short_variance() const noexcept { return short_var_.load(std::memory_order_relaxed); }
double TokenBurstIntensity::long_variance()  const noexcept { return long_var_.load(std::memory_order_relaxed); }
double TokenBurstIntensity::burst_ratio()    const noexcept { return ratio_.load(std::memory_order_relaxed); }
bool   TokenBurstIntensity::is_bursting()    const noexcept { return bursting_.load(std::memory_order_relaxed); }

std::size_t TokenBurstIntensity::burst_events()  const noexcept { return burst_ev_.load(std::memory_order_relaxed); }
std::size_t TokenBurstIntensity::total_records() const noexcept { return total_.load(std::memory_order_relaxed); }

std::string TokenBurstIntensity::to_stats_json() const {
    std::lock_guard<std::mutex> lk(mutex_);
    std::ostringstream ss;
    ss << std::fixed << std::setprecision(6);
    ss << "{\"short_variance\":" << short_var_.load(std::memory_order_relaxed)
       << ",\"long_variance\":"  << long_var_.load(std::memory_order_relaxed)
       << ",\"burst_ratio\":"    << ratio_.load(std::memory_order_relaxed)
       << ",\"is_bursting\":"    << (bursting_.load(std::memory_order_relaxed) ? "true" : "false")
       << ",\"burst_events\":"   << burst_ev_.load(std::memory_order_relaxed)
       << ",\"total_records\":"  << total_.load(std::memory_order_relaxed)
       << "}";
    return ss.str();
}

void TokenBurstIntensity::reset() {
    std::lock_guard<std::mutex> lk(mutex_);
    std::fill(short_buf_.begin(), short_buf_.end(), 0.0);
    std::fill(long_buf_.begin(),  long_buf_.end(),  0.0);
    short_head_ = 0; short_fill_ = 0;
    long_head_  = 0; long_fill_  = 0;
    prev_bursting_ = false;
    short_var_.store(0.0, std::memory_order_relaxed);
    long_var_.store(0.0, std::memory_order_relaxed);
    ratio_.store(0.0, std::memory_order_relaxed);
    bursting_.store(false, std::memory_order_relaxed);
    burst_ev_.store(0, std::memory_order_relaxed);
    total_.store(0, std::memory_order_relaxed);
}

void TokenBurstIntensity::update_config(const Config& cfg) {
    std::lock_guard<std::mutex> lk(mutex_);
    cfg_ = cfg;
    short_buf_.assign(cfg_.short_window, 0.0);
    long_buf_.assign(cfg_.long_window,   0.0);
    short_head_ = 0; short_fill_ = 0;
    long_head_  = 0; long_fill_  = 0;
}

} // namespace llmquant
