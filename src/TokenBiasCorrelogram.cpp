#include "TokenBiasCorrelogram.h"
#include <cmath>
#include <numeric>
#include <sstream>
#include <iomanip>
#include <algorithm>

namespace llmquant {

TokenBiasCorrelogram::TokenBiasCorrelogram(Config cfg)
    : cfg_(std::move(cfg))
    , buf_(cfg_.window, 0.0)
    , acf_values_(cfg_.max_lag, 0.0)
{}

void TokenBiasCorrelogram::record(double bias) {
    int    new_dom_lag = 0;
    double new_peak    = 0.0;
    bool   cycle_fired = false;

    {
        std::lock_guard<std::mutex> lk(mutex_);

        buf_[head_] = bias;
        head_ = (head_ + 1) % cfg_.window;
        if (fill_ < cfg_.window) ++fill_;

        if (fill_ >= cfg_.min_samples) {
            int n = fill_;
            // Compute mean over window
            double sum = 0.0;
            for (int i = 0; i < n; ++i) sum += buf_[i];
            double mu = sum / n;

            // Variance (lag-0 autocovariance)
            double var = 0.0;
            for (int i = 0; i < n; ++i) {
                double d = buf_[i] - mu;
                var += d * d;
            }
            var /= n;

            if (var > 1e-14) {
                int effective_lag = std::min(cfg_.max_lag, n - 1);
                new_peak = -1.0; new_dom_lag = 1;

                for (int lag = 1; lag <= effective_lag; ++lag) {
                    double cov = 0.0;
                    for (int i = lag; i < n; ++i) {
                        // Ordered access via circular buffer
                        int idx_i   = (head_ - n + i     + cfg_.window) % cfg_.window;
                        int idx_lag = (head_ - n + i - lag + cfg_.window) % cfg_.window;
                        cov += (buf_[idx_i] - mu) * (buf_[idx_lag] - mu);
                    }
                    cov /= (n - lag);
                    double acf = cov / var;
                    acf_values_[lag - 1] = acf;

                    if (acf > new_peak) {
                        new_peak    = acf;
                        new_dom_lag = lag;
                    }
                }

                if (new_peak > cfg_.cycle_threshold) {
                    cycle_fired = true;
                    cycle_events_.fetch_add(1, std::memory_order_relaxed);
                }
            }
        }
    }

    dominant_lag_.store(new_dom_lag, std::memory_order_relaxed);
    peak_acf_.store(new_peak, std::memory_order_relaxed);
    total_.fetch_add(1, std::memory_order_relaxed);

    if (cycle_fired && cfg_.on_cycle_detected) cfg_.on_cycle_detected(new_dom_lag, new_peak);
}

double TokenBiasCorrelogram::acf_at(int lag) const {
    std::lock_guard<std::mutex> lk(mutex_);
    if (lag < 1 || lag > cfg_.max_lag) return 0.0;
    return acf_values_[lag - 1];
}

int    TokenBiasCorrelogram::dominant_lag() const noexcept { return dominant_lag_.load(std::memory_order_relaxed); }
double TokenBiasCorrelogram::peak_acf()     const noexcept { return peak_acf_.load(std::memory_order_relaxed); }

std::size_t TokenBiasCorrelogram::cycle_events()  const noexcept { return cycle_events_.load(std::memory_order_relaxed); }
std::size_t TokenBiasCorrelogram::total_records() const noexcept { return total_.load(std::memory_order_relaxed); }

std::string TokenBiasCorrelogram::to_stats_json() const {
    std::lock_guard<std::mutex> lk(mutex_);
    std::ostringstream ss;
    ss << std::fixed << std::setprecision(4);
    ss << "{\"dominant_lag\":" << dominant_lag_.load(std::memory_order_relaxed)
       << ",\"peak_acf\":" << peak_acf_.load(std::memory_order_relaxed)
       << ",\"cycle_events\":" << cycle_events_.load(std::memory_order_relaxed)
       << ",\"total_records\":" << total_.load(std::memory_order_relaxed)
       << "}";
    return ss.str();
}

void TokenBiasCorrelogram::reset() {
    std::lock_guard<std::mutex> lk(mutex_);
    std::fill(buf_.begin(), buf_.end(), 0.0);
    std::fill(acf_values_.begin(), acf_values_.end(), 0.0);
    head_ = 0; fill_ = 0;
    dominant_lag_.store(0, std::memory_order_relaxed);
    peak_acf_.store(0.0, std::memory_order_relaxed);
    cycle_events_.store(0, std::memory_order_relaxed);
    total_.store(0, std::memory_order_relaxed);
}

void TokenBiasCorrelogram::update_config(const Config& cfg) {
    std::lock_guard<std::mutex> lk(mutex_);
    cfg_ = cfg;
    buf_.assign(cfg_.window, 0.0);
    acf_values_.assign(cfg_.max_lag, 0.0);
    head_ = 0; fill_ = 0;
}

} // namespace llmquant
