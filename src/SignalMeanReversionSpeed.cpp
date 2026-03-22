#include "SignalMeanReversionSpeed.h"
#include <cmath>
#include <numeric>
#include <sstream>
#include <iomanip>

namespace llmquant {

SignalMeanReversionSpeed::SignalMeanReversionSpeed(Config cfg)
    : cfg_(std::move(cfg))
    , buf_(cfg_.window, 0.0)
{}

void SignalMeanReversionSpeed::record(double bias) {
    double new_theta    = 0.0;
    double new_kappa    = 0.0;
    double new_hl       = 0.0;
    bool fast_ev  = false;
    bool expl_ev  = false;

    {
        std::lock_guard<std::mutex> lk(mutex_);

        buf_[head_] = bias;
        head_ = (head_ + 1) % cfg_.window;
        if (fill_ < cfg_.window) ++fill_;

        if (fill_ >= cfg_.min_samples) {
            int n = fill_;
            // OLS: regress x[t] on x[t-1] → θ = Σ(x[t]·x[t-1]) / Σ(x[t-1]²)
            double sum_xy = 0.0, sum_xx = 0.0;
            for (int i = 1; i < n; ++i) {
                int idx_t   = (head_ - n + i     + cfg_.window) % cfg_.window;
                int idx_tm1 = (head_ - n + i - 1 + cfg_.window) % cfg_.window;
                sum_xy += buf_[idx_t] * buf_[idx_tm1];
                sum_xx += buf_[idx_tm1] * buf_[idx_tm1];
            }
            if (std::fabs(sum_xx) > 1e-14) {
                new_theta = sum_xy / sum_xx;
                // κ = −log(|θ|); meaningful when |θ| < 1
                if (new_theta > 0.0 && new_theta < 1.0) {
                    new_kappa = -std::log(new_theta);
                    new_hl    = std::log(2.0) / new_kappa;
                } else {
                    new_kappa = 0.0;
                    new_hl    = 0.0;
                }
            }

            bool fast = (new_kappa > cfg_.fast_kappa_threshold);
            bool expl = (std::fabs(new_theta) > cfg_.explosive_theta);
            if (fast && !prev_fast_) { fast_ev = true; fast_events_.fetch_add(1, std::memory_order_relaxed); }
            if (expl && !prev_expl_) { expl_ev = true; explosive_events_.fetch_add(1, std::memory_order_relaxed); }
            prev_fast_ = fast;
            prev_expl_ = expl;
        }
    }

    theta_.store(new_theta, std::memory_order_relaxed);
    kappa_.store(new_kappa, std::memory_order_relaxed);
    half_life_.store(new_hl, std::memory_order_relaxed);
    total_.fetch_add(1, std::memory_order_relaxed);

    if (fast_ev && cfg_.on_fast_reversion) cfg_.on_fast_reversion(new_kappa);
    if (expl_ev && cfg_.on_explosive)      cfg_.on_explosive(new_theta);
}

double SignalMeanReversionSpeed::theta()     const noexcept { return theta_.load(std::memory_order_relaxed); }
double SignalMeanReversionSpeed::kappa()     const noexcept { return kappa_.load(std::memory_order_relaxed); }
double SignalMeanReversionSpeed::half_life() const noexcept { return half_life_.load(std::memory_order_relaxed); }

std::size_t SignalMeanReversionSpeed::fast_reversion_events() const noexcept { return fast_events_.load(std::memory_order_relaxed); }
std::size_t SignalMeanReversionSpeed::explosive_events()      const noexcept { return explosive_events_.load(std::memory_order_relaxed); }
std::size_t SignalMeanReversionSpeed::total_records()         const noexcept { return total_.load(std::memory_order_relaxed); }

std::string SignalMeanReversionSpeed::to_stats_json() const {
    std::lock_guard<std::mutex> lk(mutex_);
    std::ostringstream ss;
    ss << std::fixed << std::setprecision(4);
    ss << "{\"theta\":" << theta_.load(std::memory_order_relaxed)
       << ",\"kappa\":" << kappa_.load(std::memory_order_relaxed)
       << ",\"half_life\":" << half_life_.load(std::memory_order_relaxed)
       << ",\"fast_reversion_events\":" << fast_events_.load(std::memory_order_relaxed)
       << ",\"explosive_events\":" << explosive_events_.load(std::memory_order_relaxed)
       << ",\"total_records\":" << total_.load(std::memory_order_relaxed)
       << "}";
    return ss.str();
}

void SignalMeanReversionSpeed::reset() {
    std::lock_guard<std::mutex> lk(mutex_);
    std::fill(buf_.begin(), buf_.end(), 0.0);
    head_ = 0; fill_ = 0; prev_fast_ = false; prev_expl_ = false;
    theta_.store(0.0, std::memory_order_relaxed);
    kappa_.store(0.0, std::memory_order_relaxed);
    half_life_.store(0.0, std::memory_order_relaxed);
    fast_events_.store(0, std::memory_order_relaxed);
    explosive_events_.store(0, std::memory_order_relaxed);
    total_.store(0, std::memory_order_relaxed);
}

void SignalMeanReversionSpeed::update_config(const Config& cfg) {
    std::lock_guard<std::mutex> lk(mutex_);
    cfg_ = cfg;
    buf_.assign(cfg_.window, 0.0);
    head_ = 0; fill_ = 0;
}

} // namespace llmquant
