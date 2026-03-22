#include "BiasKalmanFilter.h"

#include <algorithm>
#include <cmath>
#include <sstream>
#include <iomanip>

namespace llmquant {

BiasKalmanFilter::BiasKalmanFilter(Config cfg)
    : cfg_(std::move(cfg))
{
    x_hat_  = cfg_.x0;
    P_      = cfg_.P0;
    R_est_  = cfg_.R;
    x_hat_a_.store(x_hat_, std::memory_order_relaxed);
}

void BiasKalmanFilter::record(double bias) {
    bool   fire   = false;
    double nis_v  = 0.0;
    std::function<void(double)> cb;

    {
        std::lock_guard<std::mutex> lk(mutex_);
        cb = cfg_.on_model_mismatch;
        total_.fetch_add(1, std::memory_order_relaxed);

        // --- Predict ---
        // x̂_{t|t-1} = x̂_{t-1|t-1}  (random walk: A=1)
        // P_{t|t-1}  = P_{t-1|t-1} + Q
        double P_pred = P_ + cfg_.Q;

        // Innovation: ν = z - x̂_{t|t-1}
        double nu = bias - x_hat_;

        // Innovation variance: S = P_{t|t-1} + R
        double R_use = cfg_.estimate_R ? R_est_ : cfg_.R;
        double S     = P_pred + R_use;

        // Kalman gain: K = P_{t|t-1} / S
        double K = (S > 1e-14) ? P_pred / S : 0.0;

        // --- Update ---
        x_hat_ = x_hat_ + K * nu;
        P_      = (1.0 - K) * P_pred;

        // Normalised Innovation Squared
        nis_v = (S > 1e-14) ? (nu * nu / S) : 0.0;

        // Online R estimation: EMA of ν²
        if (cfg_.estimate_R) {
            double alpha = 1.0 / static_cast<double>(std::max(1, cfg_.R_estimation_period));
            innov_ema_ += alpha * (nu * nu - innov_ema_);
            // R = E[ν²] - P_{t|t-1} (innovation covariance consistency)
            R_est_ = std::max(1e-6, innov_ema_ - P_pred);
        }

        x_hat_a_.store(x_hat_, std::memory_order_relaxed);
        innovation_.store(nu,  std::memory_order_relaxed);
        nis_.store(nis_v,      std::memory_order_relaxed);
        K_.store(K,            std::memory_order_relaxed);

        if (nis_v > cfg_.nis_threshold) {
            fire = true;
            mismatch_events_.fetch_add(1, std::memory_order_relaxed);
        }
    }

    if (fire && cb) cb(nis_v);
}

std::string BiasKalmanFilter::to_stats_json() const {
    std::lock_guard<std::mutex> lk(mutex_);
    std::ostringstream ss;
    ss << std::fixed << std::setprecision(6);
    ss << "{"
       << "\"filtered_bias\":"    << x_hat_a_.load(std::memory_order_relaxed) << ","
       << "\"innovation\":"       << innovation_.load(std::memory_order_relaxed) << ","
       << "\"nis\":"              << nis_.load(std::memory_order_relaxed) << ","
       << "\"kalman_gain\":"      << K_.load(std::memory_order_relaxed) << ","
       << "\"mismatch_events\":"  << mismatch_events_.load(std::memory_order_relaxed) << ","
       << "\"total_records\":"    << total_.load(std::memory_order_relaxed)
       << "}";
    return ss.str();
}

void BiasKalmanFilter::reset() {
    std::lock_guard<std::mutex> lk(mutex_);
    x_hat_     = cfg_.x0;
    P_         = cfg_.P0;
    R_est_     = cfg_.R;
    innov_ema_ = 0.0;
    x_hat_a_.store(x_hat_, std::memory_order_relaxed);
    innovation_.store(0.0, std::memory_order_relaxed);
    nis_.store(0.0,        std::memory_order_relaxed);
    K_.store(0.0,          std::memory_order_relaxed);
    mismatch_events_.store(0, std::memory_order_relaxed);
    total_.store(0,           std::memory_order_relaxed);
}

void BiasKalmanFilter::update_config(const Config& cfg) {
    std::lock_guard<std::mutex> lk(mutex_);
    cfg_       = cfg;
    x_hat_     = cfg_.x0;
    P_         = cfg_.P0;
    R_est_     = cfg_.R;
    innov_ema_ = 0.0;
    x_hat_a_.store(x_hat_, std::memory_order_relaxed);
}

} // namespace llmquant
