#include "BiasGarchEstimator.h"
#include <cmath>
#include <sstream>
#include <iomanip>

namespace llmquant {

BiasGarchEstimator::BiasGarchEstimator(Config cfg)
    : cfg_(std::move(cfg))
    , sigma2_(cfg_.sigma2_init)
    , prev_sigma_(std::sqrt(cfg_.sigma2_init))
{}

void BiasGarchEstimator::record(double bias) {
    double new_sigma2{};
    double new_sigma{};
    double eps_val{};
    bool spike = false;

    {
        std::lock_guard<std::mutex> lk(mutex_);

        // Update running mean (slow EMA)
        mu_ = mu_ * 0.95 + bias * 0.05;

        // Innovation
        double eps = bias - mu_;
        eps_val = eps;

        // GARCH(1,1): σ²_t = ω + α ε²_{t-1} + β σ²_{t-1}
        sigma2_ = cfg_.omega
                + cfg_.alpha * eps * eps
                + cfg_.beta  * sigma2_;

        // Clamp to prevent runaway
        if (sigma2_ < 1e-10) sigma2_ = 1e-10;
        if (sigma2_ > 1e6)   sigma2_ = 1e6;

        new_sigma2 = sigma2_;
        new_sigma  = std::sqrt(sigma2_);

        if (total_.load(std::memory_order_relaxed) >= static_cast<uint64_t>(cfg_.min_samples)) {
            if (prev_sigma_ > 1e-10 && new_sigma / prev_sigma_ > cfg_.vol_spike_ratio) {
                spike = true;
                spike_events_.fetch_add(1, std::memory_order_relaxed);
            }
        }
        prev_sigma_ = new_sigma;
    }

    sigma_.store(new_sigma, std::memory_order_relaxed);
    sigma2_a_.store(new_sigma2, std::memory_order_relaxed);
    eps_.store(eps_val, std::memory_order_relaxed);
    total_.fetch_add(1, std::memory_order_relaxed);

    if (spike && cfg_.on_vol_spike) cfg_.on_vol_spike(new_sigma);
}

std::string BiasGarchEstimator::to_stats_json() const {
    std::lock_guard<std::mutex> lk(mutex_);
    std::ostringstream ss;
    ss << std::fixed << std::setprecision(6);
    ss << "{\"sigma\":" << sigma_.load(std::memory_order_relaxed)
       << ",\"sigma2\":" << sigma2_a_.load(std::memory_order_relaxed)
       << ",\"last_eps\":" << eps_.load(std::memory_order_relaxed)
       << ",\"persistence\":" << (cfg_.alpha + cfg_.beta)
       << ",\"spike_events\":" << spike_events_.load(std::memory_order_relaxed)
       << ",\"total_records\":" << total_.load(std::memory_order_relaxed)
       << "}";
    return ss.str();
}

void BiasGarchEstimator::reset() {
    std::lock_guard<std::mutex> lk(mutex_);
    sigma2_    = cfg_.sigma2_init;
    mu_        = 0.0;
    prev_sigma_ = std::sqrt(cfg_.sigma2_init);
    sigma_.store(std::sqrt(cfg_.sigma2_init), std::memory_order_relaxed);
    sigma2_a_.store(cfg_.sigma2_init, std::memory_order_relaxed);
    eps_.store(0.0, std::memory_order_relaxed);
    spike_events_.store(0, std::memory_order_relaxed);
    total_.store(0, std::memory_order_relaxed);
}

void BiasGarchEstimator::update_config(const Config& cfg) {
    std::lock_guard<std::mutex> lk(mutex_);
    cfg_ = cfg;
}

} // namespace llmquant
