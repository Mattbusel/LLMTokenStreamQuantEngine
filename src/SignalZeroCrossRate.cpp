#include "SignalZeroCrossRate.h"
#include <sstream>
#include <iomanip>

namespace llmquant {

SignalZeroCrossRate::SignalZeroCrossRate(Config cfg)
    : cfg_(std::move(cfg))
    , buf_(cfg_.window, 0.0)
{}

void SignalZeroCrossRate::record(double bias) {
    double new_zcr   = 0.0;
    bool high_event  = false;
    bool low_event   = false;

    {
        std::lock_guard<std::mutex> lk(mutex_);

        buf_[head_] = bias;
        head_ = (head_ + 1) % cfg_.window;
        if (fill_ < cfg_.window) ++fill_;

        if (fill_ >= cfg_.min_samples) {
            int n = fill_;
            int crossings = 0;
            for (int i = 1; i < n; ++i) {
                // Count sign changes (treat zero as positive)
                double a = buf_[(head_ - n + i - 1 + cfg_.window) % cfg_.window];
                double b = buf_[(head_ - n + i     + cfg_.window) % cfg_.window];
                if ((a >= 0.0) != (b >= 0.0)) ++crossings;
            }
            new_zcr = static_cast<double>(crossings) / (n - 1);

            bool h = (new_zcr > cfg_.high_threshold);
            bool l = (new_zcr < cfg_.low_threshold);

            if (h && !prev_high_) {
                high_event = true;
                high_events_.fetch_add(1, std::memory_order_relaxed);
            }
            if (l && !prev_low_) {
                low_event = true;
                low_events_.fetch_add(1, std::memory_order_relaxed);
            }
            prev_high_ = h;
            prev_low_  = l;
        }
    }

    zcr_.store(new_zcr, std::memory_order_relaxed);
    total_.fetch_add(1, std::memory_order_relaxed);

    if (high_event && cfg_.on_high_zcr) cfg_.on_high_zcr(new_zcr);
    if (low_event  && cfg_.on_low_zcr)  cfg_.on_low_zcr(new_zcr);
}

double SignalZeroCrossRate::zcr() const noexcept { return zcr_.load(std::memory_order_relaxed); }

std::size_t SignalZeroCrossRate::high_zcr_events() const noexcept { return high_events_.load(std::memory_order_relaxed); }
std::size_t SignalZeroCrossRate::low_zcr_events()  const noexcept { return low_events_.load(std::memory_order_relaxed); }
std::size_t SignalZeroCrossRate::total_records()   const noexcept { return total_.load(std::memory_order_relaxed); }

std::string SignalZeroCrossRate::to_stats_json() const {
    std::lock_guard<std::mutex> lk(mutex_);
    std::ostringstream ss;
    ss << std::fixed << std::setprecision(4);
    ss << "{\"zcr\":" << zcr_.load(std::memory_order_relaxed)
       << ",\"high_zcr_events\":" << high_events_.load(std::memory_order_relaxed)
       << ",\"low_zcr_events\":" << low_events_.load(std::memory_order_relaxed)
       << ",\"total_records\":" << total_.load(std::memory_order_relaxed)
       << "}";
    return ss.str();
}

void SignalZeroCrossRate::reset() {
    std::lock_guard<std::mutex> lk(mutex_);
    std::fill(buf_.begin(), buf_.end(), 0.0);
    head_ = 0; fill_ = 0; prev_high_ = false; prev_low_ = false;
    zcr_.store(0.0, std::memory_order_relaxed);
    high_events_.store(0, std::memory_order_relaxed);
    low_events_.store(0, std::memory_order_relaxed);
    total_.store(0, std::memory_order_relaxed);
}

void SignalZeroCrossRate::update_config(const Config& cfg) {
    std::lock_guard<std::mutex> lk(mutex_);
    cfg_ = cfg;
    buf_.assign(cfg_.window, 0.0);
    head_ = 0; fill_ = 0;
}

} // namespace llmquant
