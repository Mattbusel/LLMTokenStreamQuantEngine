#include "SignalDonchianChannel.h"
#include <algorithm>
#include <limits>
#include <sstream>
#include <iomanip>

namespace llmquant {

SignalDonchianChannel::SignalDonchianChannel(Config cfg)
    : cfg_(std::move(cfg))
{
    int w = std::max(4, cfg_.window);
    win_buf_.assign(static_cast<std::size_t>(w), 0.0);
}

void SignalDonchianChannel::record(double bias) {
    bool   fire_upper = false, fire_lower = false;
    double new_upper = 0.0, new_lower = 0.0, new_mid = 0.0;
    double old_upper = 0.0, old_lower = 0.0;
    std::function<void(double, double)> upper_cb, lower_cb;

    {
        std::lock_guard<std::mutex> lk(mutex_);
        upper_cb = cfg_.on_upper_breakout;
        lower_cb = cfg_.on_lower_breakout;

        int w = static_cast<int>(win_buf_.size());
        if (win_fill_ < w) ++win_fill_;
        win_buf_[static_cast<std::size_t>(win_head_)] = bias;
        win_head_ = (win_head_ + 1) % w;

        if (win_fill_ >= cfg_.min_samples) {
            double hi = std::numeric_limits<double>::lowest();
            double lo = std::numeric_limits<double>::max();
            for (int i = 0; i < win_fill_; ++i) {
                double v = win_buf_[static_cast<std::size_t>(i)];
                if (v > hi) hi = v;
                if (v < lo) lo = v;
            }
            new_upper = hi;
            new_lower = lo;
            new_mid   = (hi + lo) * 0.5;

            old_upper = upper_.load(std::memory_order_relaxed);
            old_lower = lower_.load(std::memory_order_relaxed);

            if (seeded_ && hi > old_upper) {
                fire_upper = true;
                upper_events_.fetch_add(1, std::memory_order_relaxed);
            }
            if (seeded_ && lo < old_lower) {
                fire_lower = true;
                lower_events_.fetch_add(1, std::memory_order_relaxed);
            }
            seeded_ = true;

            upper_.store(hi,      std::memory_order_relaxed);
            lower_.store(lo,      std::memory_order_relaxed);
            mid_.store(new_mid,   std::memory_order_relaxed);
            at_high_.store(bias >= hi, std::memory_order_relaxed);
            at_low_.store(bias  <= lo, std::memory_order_relaxed);
        }
    }

    total_.fetch_add(1, std::memory_order_relaxed);
    if (fire_upper && upper_cb) upper_cb(new_upper, old_upper);
    if (fire_lower && lower_cb) lower_cb(new_lower, old_lower);
}

double SignalDonchianChannel::upper_band()  const noexcept { return upper_.load(std::memory_order_relaxed); }
double SignalDonchianChannel::lower_band()  const noexcept { return lower_.load(std::memory_order_relaxed); }
double SignalDonchianChannel::midline()     const noexcept { return mid_.load(std::memory_order_relaxed); }
bool   SignalDonchianChannel::is_at_high()  const noexcept { return at_high_.load(std::memory_order_relaxed); }
bool   SignalDonchianChannel::is_at_low()   const noexcept { return at_low_.load(std::memory_order_relaxed); }

std::size_t SignalDonchianChannel::upper_breakouts() const noexcept { return upper_events_.load(std::memory_order_relaxed); }
std::size_t SignalDonchianChannel::lower_breakouts() const noexcept { return lower_events_.load(std::memory_order_relaxed); }
std::size_t SignalDonchianChannel::total_records()   const noexcept { return total_.load(std::memory_order_relaxed); }

std::string SignalDonchianChannel::to_stats_json() const {
    std::ostringstream ss;
    ss << std::fixed << std::setprecision(4);
    ss << "{\"upper_band\":"      << upper_.load(std::memory_order_relaxed)
       << ",\"lower_band\":"      << lower_.load(std::memory_order_relaxed)
       << ",\"midline\":"         << mid_.load(std::memory_order_relaxed)
       << ",\"upper_breakouts\":" << upper_events_.load(std::memory_order_relaxed)
       << ",\"lower_breakouts\":" << lower_events_.load(std::memory_order_relaxed)
       << ",\"total_records\":"   << total_.load(std::memory_order_relaxed)
       << "}";
    return ss.str();
}

void SignalDonchianChannel::reset() {
    std::lock_guard<std::mutex> lk(mutex_);
    std::fill(win_buf_.begin(), win_buf_.end(), 0.0);
    win_head_ = win_fill_ = 0;
    seeded_ = false;
    upper_.store(0.0,   std::memory_order_relaxed);
    lower_.store(0.0,   std::memory_order_relaxed);
    mid_.store(0.0,     std::memory_order_relaxed);
    at_high_.store(false, std::memory_order_relaxed);
    at_low_.store(false,  std::memory_order_relaxed);
    upper_events_.store(0, std::memory_order_relaxed);
    lower_events_.store(0, std::memory_order_relaxed);
    total_.store(0,        std::memory_order_relaxed);
}

void SignalDonchianChannel::update_config(const Config& cfg) {
    std::lock_guard<std::mutex> lk(mutex_);
    cfg_ = cfg;
    int w = std::max(4, cfg_.window);
    win_buf_.assign(static_cast<std::size_t>(w), 0.0);
    win_head_ = win_fill_ = 0;
    seeded_ = false;
    upper_.store(0.0,   std::memory_order_relaxed);
    lower_.store(0.0,   std::memory_order_relaxed);
    mid_.store(0.0,     std::memory_order_relaxed);
    at_high_.store(false, std::memory_order_relaxed);
    at_low_.store(false,  std::memory_order_relaxed);
    upper_events_.store(0, std::memory_order_relaxed);
    lower_events_.store(0, std::memory_order_relaxed);
    total_.store(0,        std::memory_order_relaxed);
}

} // namespace llmquant
