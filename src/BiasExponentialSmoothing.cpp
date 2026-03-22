#include "BiasExponentialSmoothing.h"
#include <cmath>
#include <sstream>
#include <iomanip>

namespace llmquant {

BiasExponentialSmoothing::BiasExponentialSmoothing(Config cfg)
    : cfg_(std::move(cfg))
{}

void BiasExponentialSmoothing::record(double bias) {
    double new_level = 0.0, new_trend = 0.0, new_fc = 0.0, new_err = 0.0;
    bool err_ev = false;

    {
        std::lock_guard<std::mutex> lk(mutex_);

        if (n_ == 0) {
            level_raw_ = bias;
            trend_raw_ = 0.0;
        } else {
            double prev_level = level_raw_;
            level_raw_ = cfg_.alpha * bias + (1.0 - cfg_.alpha) * (level_raw_ + trend_raw_);
            trend_raw_ = cfg_.beta  * (level_raw_ - prev_level) + (1.0 - cfg_.beta) * trend_raw_;
        }
        ++n_;

        new_level = level_raw_;
        new_trend = trend_raw_;
        new_fc    = level_raw_ + trend_raw_;

        if (forecasting_) {
            new_err = std::fabs(bias - prev_forecast_);
            if (new_err > cfg_.error_threshold) {
                err_ev = true;
                error_events_.fetch_add(1, std::memory_order_relaxed);
            }
        }
        forecasting_ = (n_ >= cfg_.min_samples);
        prev_forecast_ = new_fc;
    }

    level_.store(new_level, std::memory_order_relaxed);
    trend_.store(new_trend, std::memory_order_relaxed);
    forecast_.store(new_fc,  std::memory_order_relaxed);
    last_error_.store(new_err, std::memory_order_relaxed);
    total_.fetch_add(1, std::memory_order_relaxed);

    if (err_ev && cfg_.on_large_error) cfg_.on_large_error(bias, prev_forecast_);
}

double BiasExponentialSmoothing::level()      const noexcept { return level_.load(std::memory_order_relaxed); }
double BiasExponentialSmoothing::trend()      const noexcept { return trend_.load(std::memory_order_relaxed); }
double BiasExponentialSmoothing::forecast()   const noexcept { return forecast_.load(std::memory_order_relaxed); }
double BiasExponentialSmoothing::last_error() const noexcept { return last_error_.load(std::memory_order_relaxed); }

std::size_t BiasExponentialSmoothing::error_events()  const noexcept { return error_events_.load(std::memory_order_relaxed); }
std::size_t BiasExponentialSmoothing::total_records() const noexcept { return total_.load(std::memory_order_relaxed); }

std::string BiasExponentialSmoothing::to_stats_json() const {
    std::lock_guard<std::mutex> lk(mutex_);
    std::ostringstream ss;
    ss << std::fixed << std::setprecision(6);
    ss << "{\"level\":"        << level_.load(std::memory_order_relaxed)
       << ",\"trend\":"        << trend_.load(std::memory_order_relaxed)
       << ",\"forecast\":"     << forecast_.load(std::memory_order_relaxed)
       << ",\"last_error\":"   << last_error_.load(std::memory_order_relaxed)
       << ",\"error_events\":" << error_events_.load(std::memory_order_relaxed)
       << ",\"total_records\":" << total_.load(std::memory_order_relaxed)
       << "}";
    return ss.str();
}

void BiasExponentialSmoothing::reset() {
    std::lock_guard<std::mutex> lk(mutex_);
    n_ = 0; level_raw_ = 0.0; trend_raw_ = 0.0;
    prev_forecast_ = 0.0; forecasting_ = false;
    level_.store(0.0, std::memory_order_relaxed);
    trend_.store(0.0, std::memory_order_relaxed);
    forecast_.store(0.0, std::memory_order_relaxed);
    last_error_.store(0.0, std::memory_order_relaxed);
    error_events_.store(0, std::memory_order_relaxed);
    total_.store(0, std::memory_order_relaxed);
}

void BiasExponentialSmoothing::update_config(const Config& cfg) {
    std::lock_guard<std::mutex> lk(mutex_);
    cfg_ = cfg;
    n_ = 0; level_raw_ = 0.0; trend_raw_ = 0.0;
    prev_forecast_ = 0.0; forecasting_ = false;
    level_.store(0.0,      std::memory_order_relaxed);
    trend_.store(0.0,      std::memory_order_relaxed);
    forecast_.store(0.0,   std::memory_order_relaxed);
    last_error_.store(0.0, std::memory_order_relaxed);
    error_events_.store(0, std::memory_order_relaxed);
    total_.store(0,        std::memory_order_relaxed);
}

} // namespace llmquant
