#include "VolatilityForecaster.h"

#include <cmath>
#include <spdlog/spdlog.h>
#include <sstream>

namespace llmquant {

// ---------------------------------------------------------------------------
// Construction
// ---------------------------------------------------------------------------

VolatilityForecaster::VolatilityForecaster(Config config)
    : config_(std::move(config))
    , sigma2_(config_.initial_variance) {}

// ---------------------------------------------------------------------------
// record
// ---------------------------------------------------------------------------

void VolatilityForecaster::record(double bias_value) {
    total_records_.fetch_add(1, std::memory_order_relaxed);

    std::function<void(double, double)> cb;
    bool fire_cb = false;
    double cur_vol = 0.0;
    {
        std::lock_guard<std::mutex> lk(mutex_);
        double r = 0.0;
        if (has_prev_) {
            r = bias_value - prev_value_;
        }
        prev_value_ = bias_value;
        has_prev_   = true;

        // GARCH(1,1): σ²_t = ω + α * r²_{t-1} + β * σ²_{t-1}
        sigma2_ = config_.omega + config_.alpha * r * r + config_.beta * sigma2_;
        cur_vol = std::sqrt(sigma2_);

        bool now_high = (config_.high_vol_threshold > 0.0 &&
                         cur_vol > config_.high_vol_threshold);
        bool was_high = high_vol_.exchange(now_high, std::memory_order_relaxed);
        if (now_high && !was_high) {
            cb      = config_.on_high_vol;
            fire_cb = true;
        }
    }
    if (fire_cb) {
        double var = 0.0;
        {
            std::lock_guard<std::mutex> lk(mutex_);
            var = sigma2_;
        }
        high_vol_events_.fetch_add(1, std::memory_order_relaxed);
        spdlog::warn("[vol_fcst] HIGH vol={:.4f} var={:.6f}", cur_vol, var);
        if (cb) cb(cur_vol, var);
    }
}

// ---------------------------------------------------------------------------
// reset
// ---------------------------------------------------------------------------

void VolatilityForecaster::reset() {
    std::lock_guard<std::mutex> lk(mutex_);
    sigma2_   = config_.initial_variance;
    prev_value_ = 0.0;
    has_prev_   = false;
    high_vol_.store(false, std::memory_order_relaxed);
}

// ---------------------------------------------------------------------------
// conditional_variance / conditional_vol
// ---------------------------------------------------------------------------

double VolatilityForecaster::conditional_variance() const noexcept {
    std::lock_guard<std::mutex> lk(mutex_);
    return sigma2_;
}

double VolatilityForecaster::conditional_vol() const noexcept {
    std::lock_guard<std::mutex> lk(mutex_);
    return std::sqrt(sigma2_);
}

// ---------------------------------------------------------------------------
// long_run_variance
// ---------------------------------------------------------------------------

double VolatilityForecaster::long_run_variance() const noexcept {
    std::lock_guard<std::mutex> lk(mutex_);
    double persistence = config_.alpha + config_.beta;
    if (persistence >= 1.0) return sigma2_;  // non-stationary
    return config_.omega / (1.0 - persistence);
}

// ---------------------------------------------------------------------------
// update_config
// ---------------------------------------------------------------------------

void VolatilityForecaster::update_config(const Config& config) {
    std::lock_guard<std::mutex> lk(mutex_);
    config_    = config;
    sigma2_    = config_.initial_variance;
    has_prev_  = false;
}

// ---------------------------------------------------------------------------
// to_stats_json
// ---------------------------------------------------------------------------

std::string VolatilityForecaster::to_stats_json() const {
    double var = 0.0, vol = 0.0, lrv = 0.0;
    {
        std::lock_guard<std::mutex> lk(mutex_);
        var = sigma2_;
        vol = std::sqrt(sigma2_);
        double persistence = config_.alpha + config_.beta;
        lrv = (persistence < 1.0) ? config_.omega / (1.0 - persistence) : sigma2_;
    }
    std::ostringstream s;
    s << "{\"conditional_variance\":"  << var
      << ",\"conditional_vol\":"       << vol
      << ",\"long_run_variance\":"     << lrv
      << ",\"is_high_vol\":"           << (is_high_vol() ? "true" : "false")
      << ",\"total_records\":"         << total_records()
      << ",\"high_vol_events\":"       << high_vol_events()
      << "}";
    return s.str();
}

} // namespace llmquant
