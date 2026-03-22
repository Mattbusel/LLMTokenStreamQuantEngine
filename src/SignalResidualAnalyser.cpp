#include "SignalResidualAnalyser.h"
#include <cmath>
#include <sstream>
#include <iomanip>
#include <numeric>

namespace llmquant {

SignalResidualAnalyser::SignalResidualAnalyser(Config cfg)
    : cfg_(std::move(cfg))
    , resid_buf_(cfg_.window, 0.0)
{}

void SignalResidualAnalyser::record(double bias) {
    double new_dw  = 2.0;
    double new_ar1{};
    double resid_val{};
    bool   alarm = false;

    {
        std::lock_guard<std::mutex> lk(mutex_);

        if (!seeded_) {
            prev_bias_ = bias;
            seeded_    = true;
            total_.fetch_add(1, std::memory_order_relaxed);
            return;
        }

        // Update AR(1) coefficient estimate via EMA: â = â + α*(bias*prev - â*prev²)/(prev²+ε)
        double prev2 = prev_bias_ * prev_bias_;
        if (prev2 > 1e-12) {
            ar1_est_ += cfg_.ar_alpha * (bias * prev_bias_ - ar1_est_ * prev2) / prev2;
        }

        // Compute residual
        double eps = bias - ar1_est_ * prev_bias_;
        resid_val  = eps;
        prev_bias_ = bias;

        // Store in ring buffer
        int idx = resid_head_ % cfg_.window;
        resid_buf_[idx] = eps;
        resid_head_     = (resid_head_ + 1) % cfg_.window;
        if (resid_fill_ < cfg_.window) ++resid_fill_;

        new_ar1 = ar1_est_;

        if (resid_fill_ >= cfg_.min_samples) {
            // Durbin-Watson: DW = Σ(e_t - e_{t-1})² / Σ e_t²
            int n = resid_fill_;
            double sum_sq   = 0.0;
            double sum_diff = 0.0;
            for (int i = 0; i < n; ++i) sum_sq += resid_buf_[i] * resid_buf_[i];
            for (int i = 1; i < n; ++i) {
                double d = resid_buf_[i] - resid_buf_[(i - 1 + cfg_.window) % cfg_.window];
                // Use sequential order in ring buffer (approximate)
                (void)d;
            }
            // Simple approximation: use adjacent pairs in fill order
            sum_diff = 0.0;
            for (int i = 0; i < n - 1; ++i) {
                double d = resid_buf_[(resid_head_ + i + 1) % cfg_.window]
                         - resid_buf_[(resid_head_ + i    ) % cfg_.window];
                sum_diff += d * d;
            }
            new_dw = (sum_sq > 1e-12) ? (sum_diff / sum_sq) : 2.0;

            bool ok = (new_dw >= cfg_.dw_lo && new_dw <= cfg_.dw_hi);
            if (!ok && prev_dw_ok_) {
                alarm = true;
                alarms_.fetch_add(1, std::memory_order_relaxed);
            }
            prev_dw_ok_ = ok;
        }
    }

    dw_.store(new_dw, std::memory_order_relaxed);
    ar1_.store(new_ar1, std::memory_order_relaxed);
    last_resid_.store(resid_val, std::memory_order_relaxed);
    total_.fetch_add(1, std::memory_order_relaxed);

    if (alarm && cfg_.on_residual_autocorr) cfg_.on_residual_autocorr(new_dw);
}

std::string SignalResidualAnalyser::to_stats_json() const {
    std::lock_guard<std::mutex> lk(mutex_);
    std::ostringstream ss;
    ss << std::fixed << std::setprecision(4);
    ss << "{\"durbin_watson\":" << dw_.load(std::memory_order_relaxed)
       << ",\"ar1_coeff\":"     << ar1_.load(std::memory_order_relaxed)
       << ",\"last_residual\":" << last_resid_.load(std::memory_order_relaxed)
       << ",\"alarm_events\":"  << alarms_.load(std::memory_order_relaxed)
       << ",\"total_records\":" << total_.load(std::memory_order_relaxed)
       << "}";
    return ss.str();
}

void SignalResidualAnalyser::reset() {
    std::lock_guard<std::mutex> lk(mutex_);
    ar1_est_    = 0.0;
    prev_bias_  = 0.0;
    seeded_     = false;
    resid_head_ = 0;
    resid_fill_ = 0;
    std::fill(resid_buf_.begin(), resid_buf_.end(), 0.0);
    prev_dw_ok_ = true;
    dw_.store(2.0, std::memory_order_relaxed);
    ar1_.store(0.0, std::memory_order_relaxed);
    last_resid_.store(0.0, std::memory_order_relaxed);
    alarms_.store(0, std::memory_order_relaxed);
    total_.store(0, std::memory_order_relaxed);
}

void SignalResidualAnalyser::update_config(const Config& cfg) {
    std::lock_guard<std::mutex> lk(mutex_);
    cfg_ = cfg;
    resid_buf_.assign(cfg_.window, 0.0);
    resid_head_ = 0;
    resid_fill_ = 0;
}

} // namespace llmquant
