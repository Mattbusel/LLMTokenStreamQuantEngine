#include "SentimentCycleDetector.h"

#include <algorithm>
#include <cmath>
#include <numeric>
#include <sstream>

namespace llmquant {

SentimentCycleDetector::SentimentCycleDetector()
    : config_{} {
    cyclic_threshold_cache_.store(config_.cyclic_threshold, std::memory_order_relaxed);
    ring_.assign(static_cast<size_t>(config_.window_size), 0.0);
    acf_.assign(static_cast<size_t>(config_.max_lag), 0.0);
}

SentimentCycleDetector::SentimentCycleDetector(Config cfg)
    : config_(std::move(cfg)) {
    cyclic_threshold_cache_.store(config_.cyclic_threshold, std::memory_order_relaxed);
    ring_.assign(static_cast<size_t>(config_.window_size), 0.0);
    acf_.assign(static_cast<size_t>(config_.max_lag), 0.0);
}

void SentimentCycleDetector::recompute_locked() {
    // Snapshot N samples from ring buffer in chronological order.
    int N = ring_fill_;
    if (N < config_.min_samples) return;
    N = std::min(N, config_.window_size);

    std::vector<double> data(static_cast<size_t>(N));
    for (int i = 0; i < N; ++i) {
        int idx = (ring_head_ - N + i + config_.window_size * 2) % config_.window_size;
        data[static_cast<size_t>(i)] = ring_[static_cast<size_t>(idx)];
    }

    // Compute mean and variance.
    double mean = std::accumulate(data.begin(), data.end(), 0.0) / N;
    double var  = 0.0;
    for (double v : data) var += (v - mean) * (v - mean);
    var /= N;
    if (var < 1e-12) {
        // Constant series — no cycle.
        acf_.assign(static_cast<size_t>(config_.max_lag), 0.0);
        current_period_ = 0;
        dominant_period_.store(0, std::memory_order_relaxed);
        cycle_strength_.store(0.0, std::memory_order_relaxed);
        return;
    }

    int max_lag = std::min(config_.max_lag, N / 2);
    acf_.resize(static_cast<size_t>(config_.max_lag), 0.0);

    for (int lag = 1; lag <= max_lag; ++lag) {
        double cov = 0.0;
        int    cnt = N - lag;
        for (int i = 0; i < cnt; ++i)
            cov += (data[static_cast<size_t>(i)] - mean)
                 * (data[static_cast<size_t>(i + lag)] - mean);
        acf_[static_cast<size_t>(lag - 1)] = (cov / cnt) / var;
    }

    // Find lag with the highest positive ACF (dominant period).
    int    best_lag = 0;
    double best_acf = 0.0;
    for (int lag = 1; lag <= max_lag; ++lag) {
        double a = acf_[static_cast<size_t>(lag - 1)];
        if (a > best_acf) {
            best_acf = a;
            best_lag = lag;
        }
    }

    current_period_ = best_lag;
    dominant_period_.store(best_lag, std::memory_order_relaxed);
    cycle_strength_.store(best_acf, std::memory_order_relaxed);
}

void SentimentCycleDetector::record(double bias) noexcept {
    total_.fetch_add(1, std::memory_order_relaxed);

    bool   fire       = false;
    int    old_period = 0;
    int    new_period = 0;
    double strength   = 0.0;
    Config cfg_copy;

    {
        std::lock_guard<std::mutex> lk(mutex_);
        cfg_copy = config_;

        ring_[static_cast<size_t>(ring_head_)] = bias;
        ring_head_ = (ring_head_ + 1) % config_.window_size;
        if (ring_fill_ < config_.window_size) ++ring_fill_;

        int prev_period = current_period_;
        recompute_locked();

        if (current_period_ != prev_period) {
            old_period = prev_period;
            new_period = current_period_;
            strength   = cycle_strength_.load(std::memory_order_relaxed);
            fire       = true;
            period_changes_.fetch_add(1, std::memory_order_relaxed);
        }
    }

    if (fire && cfg_copy.on_period_change)
        cfg_copy.on_period_change(new_period, old_period, strength);
}

void SentimentCycleDetector::update_config(Config cfg) {
    std::lock_guard<std::mutex> lk(mutex_);
    config_ = std::move(cfg);
    cyclic_threshold_cache_.store(config_.cyclic_threshold, std::memory_order_relaxed);
    ring_.assign(static_cast<size_t>(config_.window_size), 0.0);
    acf_.assign(static_cast<size_t>(config_.max_lag), 0.0);
    ring_head_ = 0;
    ring_fill_ = 0;
    current_period_ = 0;
    dominant_period_.store(0, std::memory_order_relaxed);
    cycle_strength_.store(0.0, std::memory_order_relaxed);
    total_.store(0, std::memory_order_relaxed);
    period_changes_.store(0, std::memory_order_relaxed);
}

SentimentCycleDetector::Config SentimentCycleDetector::get_config() const {
    std::lock_guard<std::mutex> lk(mutex_);
    return config_;
}

bool SentimentCycleDetector::is_cyclic() const noexcept {
    double cs = cycle_strength_.load(std::memory_order_relaxed);
    double th = cyclic_threshold_cache_.load(std::memory_order_relaxed);
    return cs >= th;
}

void SentimentCycleDetector::reset() {
    std::lock_guard<std::mutex> lk(mutex_);
    std::fill(ring_.begin(), ring_.end(), 0.0);
    ring_head_ = 0;
    ring_fill_ = 0;
    current_period_ = 0;
    dominant_period_.store(0, std::memory_order_relaxed);
    cycle_strength_.store(0.0, std::memory_order_relaxed);
    total_.store(0, std::memory_order_relaxed);
    period_changes_.store(0, std::memory_order_relaxed);
}

std::vector<double> SentimentCycleDetector::acf_snapshot() const {
    std::lock_guard<std::mutex> lk(mutex_);
    return acf_;
}

std::string SentimentCycleDetector::to_stats_json() const {
    int    p  = dominant_period_.load(std::memory_order_relaxed);
    double cs = cycle_strength_.load(std::memory_order_relaxed);
    uint64_t t = total_.load(std::memory_order_relaxed);
    uint64_t c = period_changes_.load(std::memory_order_relaxed);

    std::ostringstream o;
    o << std::fixed;
    o.precision(4);
    o << "{\"dominant_period\":" << p
      << ",\"cycle_strength\":"  << cs
      << ",\"is_cyclic\":"       << (is_cyclic() ? "true" : "false")
      << ",\"period_changes\":"  << c
      << ",\"total_records\":"   << t << "}";
    return o.str();
}

} // namespace llmquant
