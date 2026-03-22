#include "BiasFrequencyAnalyser.h"

#include <algorithm>
#include <cmath>
#include <sstream>
#include <iomanip>
#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

namespace llmquant {

BiasFrequencyAnalyser::BiasFrequencyAnalyser(Config cfg)
    : cfg_(std::move(cfg))
{
    int w = std::max(8, cfg_.window);
    buf_.assign(static_cast<std::size_t>(w), 0.0);
}

void BiasFrequencyAnalyser::record(double bias) {
    int    new_k    = 0;
    double new_freq = 0.0;
    double new_pow  = 0.0;
    bool   changed  = false;
    std::function<void(int, int, double)> cb;

    {
        std::lock_guard<std::mutex> lk(mutex_);
        total_.fetch_add(1, std::memory_order_relaxed);
        cb = cfg_.on_dominant_frequency_change;

        int w = static_cast<int>(buf_.size());
        buf_[static_cast<std::size_t>(head_)] = bias;
        head_ = (head_ + 1) % w;
        if (fill_ < w) ++fill_;

        int min_samp = std::max(4, cfg_.min_samples);
        if (fill_ >= min_samp) {
            recompute_locked();
            new_k    = dominant_k_.load(std::memory_order_relaxed);
            new_freq = dominant_freq_.load(std::memory_order_relaxed);
            new_pow  = dominant_power_.load(std::memory_order_relaxed);
            if (new_k != prev_dominant_k_) {
                changed = true;
                prev_dominant_k_ = new_k;
            }
        }
    }

    if (changed && cb) cb(prev_dominant_k_ == new_k ? 0 : prev_dominant_k_, new_k, new_freq);
    (void)new_pow;
}

void BiasFrequencyAnalyser::recompute_locked() {
    int n      = fill_;
    int w      = static_cast<int>(buf_.size());
    int max_k  = std::min({cfg_.max_k, kMaxK, n / 2});
    if (max_k < 1) return;

    // Build ordered array (oldest first).
    std::vector<double> x(static_cast<std::size_t>(n));
    for (int i = 0; i < n; ++i) {
        int idx = (head_ - n + i + w * 2) % w;
        x[static_cast<std::size_t>(i)] = buf_[static_cast<std::size_t>(idx)];
    }

    // DCT-II: X[k] = Σ_{j=0}^{N-1} x[j] * cos(π/N * (j+0.5) * k)
    int    best_k   = 1;
    double best_pow = -1.0;
    double inv_n    = 1.0 / static_cast<double>(n);

    for (int k = 1; k <= max_k; ++k) {
        double coef = 0.0;
        double pi_over_n = M_PI * inv_n;
        for (int j = 0; j < n; ++j) {
            coef += x[static_cast<std::size_t>(j)] *
                    std::cos(pi_over_n * (j + 0.5) * k);
        }
        double power = coef * coef;
        if (power > best_pow) {
            best_pow = power;
            best_k   = k;
        }
    }

    dominant_k_.store(best_k, std::memory_order_relaxed);
    dominant_freq_.store(static_cast<double>(best_k) / static_cast<double>(n),
                         std::memory_order_relaxed);
    dominant_power_.store(best_pow, std::memory_order_relaxed);
}

std::string BiasFrequencyAnalyser::to_stats_json() const {
    std::lock_guard<std::mutex> lk(mutex_);
    std::ostringstream ss;
    ss << std::fixed << std::setprecision(6);
    ss << "{"
       << "\"dominant_k\":"        << dominant_k_.load(std::memory_order_relaxed) << ","
       << "\"dominant_frequency\":" << dominant_freq_.load(std::memory_order_relaxed) << ","
       << "\"dominant_power\":"    << dominant_power_.load(std::memory_order_relaxed) << ","
       << "\"total_records\":"     << total_.load(std::memory_order_relaxed)
       << "}";
    return ss.str();
}

void BiasFrequencyAnalyser::reset() {
    std::lock_guard<std::mutex> lk(mutex_);
    std::fill(buf_.begin(), buf_.end(), 0.0);
    head_ = fill_ = 0;
    prev_dominant_k_ = 0;
    dominant_k_.store(0, std::memory_order_relaxed);
    dominant_freq_.store(0.0, std::memory_order_relaxed);
    dominant_power_.store(0.0, std::memory_order_relaxed);
    total_.store(0, std::memory_order_relaxed);
}

void BiasFrequencyAnalyser::update_config(const Config& cfg) {
    std::lock_guard<std::mutex> lk(mutex_);
    cfg_ = cfg;
    int w = std::max(8, cfg_.window);
    buf_.assign(static_cast<std::size_t>(w), 0.0);
    head_ = fill_ = 0;
    prev_dominant_k_ = 0;
    dominant_k_.store(0, std::memory_order_relaxed);
    dominant_freq_.store(0.0, std::memory_order_relaxed);
    dominant_power_.store(0.0, std::memory_order_relaxed);
}

} // namespace llmquant
