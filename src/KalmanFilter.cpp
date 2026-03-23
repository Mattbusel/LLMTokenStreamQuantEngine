#include "KalmanFilter.h"

#include <algorithm>
#include <cstdio>
#include <mutex>

namespace llmquant {

// ---------------------------------------------------------------------------
// Construction
// ---------------------------------------------------------------------------

KalmanFilter::KalmanFilter(Config cfg)
    : cfg_(std::move(cfg)),
      state_(cfg_.initial_state),
      covariance_(cfg_.initial_covariance),
      last_gain_(0.0) {}

// ---------------------------------------------------------------------------
// update
// ---------------------------------------------------------------------------

double KalmanFilter::update(double observation) {
    std::lock_guard<std::mutex> lk(mtx_);

    // --- Predict step ---
    // State prediction is identity (constant-velocity model with zero drift).
    // x_pred = x  (unchanged)
    // P_pred = P + Q
    double p_pred = covariance_ + cfg_.process_noise;

    // --- Update step ---
    // Kalman gain: K = P_pred / (P_pred + R)
    double denom = p_pred + cfg_.observation_noise;
    double gain  = (denom > 0.0) ? (p_pred / denom) : 0.0;

    // State update: x = x + K * (z - x)
    double innovation = observation - state_;
    state_     += gain * innovation;

    // Covariance update: P = (1 - K) * P_pred
    covariance_ = (1.0 - gain) * p_pred;

    last_gain_ = gain;
    updates_.fetch_add(1, std::memory_order_relaxed);

    return state_;
}

// ---------------------------------------------------------------------------
// Accessors
// ---------------------------------------------------------------------------

double KalmanFilter::state() const noexcept {
    std::lock_guard<std::mutex> lk(mtx_);
    return state_;
}

double KalmanFilter::covariance() const noexcept {
    std::lock_guard<std::mutex> lk(mtx_);
    return covariance_;
}

double KalmanFilter::last_gain() const noexcept {
    std::lock_guard<std::mutex> lk(mtx_);
    return last_gain_;
}

// ---------------------------------------------------------------------------
// reset / update_config
// ---------------------------------------------------------------------------

void KalmanFilter::reset() {
    std::lock_guard<std::mutex> lk(mtx_);
    state_      = cfg_.initial_state;
    covariance_ = cfg_.initial_covariance;
    last_gain_  = 0.0;
    updates_.store(0, std::memory_order_relaxed);
}

void KalmanFilter::update_config(const Config& cfg) {
    std::lock_guard<std::mutex> lk(mtx_);
    cfg_        = cfg;
    state_      = cfg_.initial_state;
    covariance_ = cfg_.initial_covariance;
    last_gain_  = 0.0;
    updates_.store(0, std::memory_order_relaxed);
}

// ---------------------------------------------------------------------------
// to_stats_json
// ---------------------------------------------------------------------------

std::string KalmanFilter::to_stats_json() const {
    std::lock_guard<std::mutex> lk(mtx_);
    char buf[512];
    std::snprintf(buf, sizeof(buf),
        "{\"state\":%.8f,\"covariance\":%.8f,\"last_gain\":%.6f,"
        "\"process_noise\":%.2e,\"observation_noise\":%.2e,"
        "\"total_updates\":%llu}",
        state_,
        covariance_,
        last_gain_,
        cfg_.process_noise,
        cfg_.observation_noise,
        static_cast<unsigned long long>(updates_.load(std::memory_order_relaxed)));
    return buf;
}

} // namespace llmquant
