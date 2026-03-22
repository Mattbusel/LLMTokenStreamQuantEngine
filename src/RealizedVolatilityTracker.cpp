#include "RealizedVolatilityTracker.h"

#include <cmath>
#include <sstream>
#include <iomanip>

namespace llmquant {

RealizedVolatilityTracker::RealizedVolatilityTracker(Config cfg)
    : cfg_(std::move(cfg))
{
    int w = std::max(2, cfg_.window);
    sq_returns_.resize(static_cast<std::size_t>(w), 0.0);
}

void RealizedVolatilityTracker::record(double bias) {
    std::function<void(double)> cb;
    bool fire_high = false;
    bool fire_low  = false;
    double rv_val  = 0.0;

    {
        std::lock_guard<std::mutex> lk(mutex_);

        if (!seeded_) {
            prev_bias_ = bias;
            seeded_    = true;
        } else {
            double ret = bias - prev_bias_;
            prev_bias_ = bias;
            double sq  = ret * ret;

            int sz = static_cast<int>(sq_returns_.size());
            // Subtract the value being evicted from sq_sum, add new value.
            sq_sum_ -= sq_returns_[static_cast<std::size_t>(head_)];
            sq_returns_[static_cast<std::size_t>(head_)] = sq;
            sq_sum_ += sq;
            if (sq_sum_ < 0.0) sq_sum_ = 0.0;  // guard floating-point underflow

            head_ = (head_ + 1) % sz;
            if (fill_ < sz) ++fill_;

            if (fill_ > 0) {
                rv_val = std::sqrt(sq_sum_ / static_cast<double>(fill_));
            }

            rv_.store(rv_val, std::memory_order_relaxed);

            double threshold = cfg_.alert_threshold;
            double clear_th  = threshold * cfg_.clear_hysteresis;

            if (!high_vol_state_ && rv_val > threshold) {
                high_vol_state_ = true;
                high_vol_flag_.store(true, std::memory_order_relaxed);
                alert_events_.fetch_add(1, std::memory_order_relaxed);
                fire_high = true;
                cb = cfg_.on_high_volatility;
            } else if (high_vol_state_ && rv_val < clear_th) {
                high_vol_state_ = false;
                high_vol_flag_.store(false, std::memory_order_relaxed);
                fire_low = true;
                cb = cfg_.on_low_volatility;
            }
        }
    }

    total_.fetch_add(1, std::memory_order_relaxed);

    if (fire_high && cb) cb(rv_val);
    if (fire_low  && cb) cb(rv_val);
}

void RealizedVolatilityTracker::reset() {
    std::lock_guard<std::mutex> lk(mutex_);
    int w = static_cast<int>(sq_returns_.size());
    sq_returns_.assign(static_cast<std::size_t>(w), 0.0);
    head_           = 0;
    fill_           = 0;
    seeded_         = false;
    prev_bias_      = 0.0;
    sq_sum_         = 0.0;
    high_vol_state_ = false;

    rv_.store(0.0, std::memory_order_relaxed);
    high_vol_flag_.store(false, std::memory_order_relaxed);
    total_.store(0, std::memory_order_relaxed);
    alert_events_.store(0, std::memory_order_relaxed);
}

void RealizedVolatilityTracker::update_config(const Config& cfg) {
    std::lock_guard<std::mutex> lk(mutex_);
    cfg_ = cfg;
    int w = std::max(2, cfg_.window);
    sq_returns_.assign(static_cast<std::size_t>(w), 0.0);
    head_           = 0;
    fill_           = 0;
    seeded_         = false;
    prev_bias_      = 0.0;
    sq_sum_         = 0.0;
    high_vol_state_ = false;

    rv_.store(0.0, std::memory_order_relaxed);
    high_vol_flag_.store(false, std::memory_order_relaxed);
    total_.store(0, std::memory_order_relaxed);
    alert_events_.store(0, std::memory_order_relaxed);
}

std::string RealizedVolatilityTracker::to_stats_json() const {
    double   rv    = rv_.load(std::memory_order_relaxed);
    bool     high  = high_vol_flag_.load(std::memory_order_relaxed);
    uint64_t tot   = total_.load(std::memory_order_relaxed);
    uint64_t evts  = alert_events_.load(std::memory_order_relaxed);

    std::ostringstream ss;
    ss << std::fixed << std::setprecision(6);
    ss << "{"
       << "\"realized_vol\":"       << rv   << ","
       << "\"is_high_volatility\":" << (high ? "true" : "false") << ","
       << "\"total_records\":"      << tot  << ","
       << "\"alert_events\":"       << evts
       << "}";
    return ss.str();
}

} // namespace llmquant
