#include "SignalSurpriseIndex.h"

#include <algorithm>
#include <cmath>
#include <numeric>
#include <sstream>

namespace llmquant {

SignalSurpriseIndex::SignalSurpriseIndex()
    : config_{} {
    hist_.assign(static_cast<size_t>(config_.n_bins), 0ULL);
    threshold_cache_.store(config_.high_surprise_threshold, std::memory_order_relaxed);
}

SignalSurpriseIndex::SignalSurpriseIndex(Config cfg)
    : config_(std::move(cfg)) {
    hist_.assign(static_cast<size_t>(config_.n_bins), 0ULL);
    threshold_cache_.store(config_.high_surprise_threshold, std::memory_order_relaxed);
}

void SignalSurpriseIndex::record(double bias) noexcept {
    total_.fetch_add(1, std::memory_order_relaxed);

    double norm_surprise = 1.0;
    bool   fire_high     = false;
    Config cfg_copy;

    {
        std::lock_guard<std::mutex> lk(mutex_);
        cfg_copy = config_;

        // Map bias into a histogram bin.
        double span = config_.range_max - config_.range_min;
        double frac = (span > 1e-12)
                    ? (bias - config_.range_min) / span
                    : 0.5;
        frac = std::max(0.0, std::min(1.0 - 1e-12, frac));
        int bin = static_cast<int>(frac * config_.n_bins);
        bin = std::max(0, std::min(config_.n_bins - 1, bin));

        // Update histogram AFTER computing surprise from the prior distribution.
        // This way the surprise reflects "how expected was this given what I saw
        // before this observation" (proper self-information).
        uint64_t n = total_.load(std::memory_order_relaxed);  // includes this record
        if (static_cast<int>(n) >= config_.min_samples) {
            uint64_t bin_count = hist_[static_cast<size_t>(bin)];
            double   p_bin     = static_cast<double>(bin_count) / static_cast<double>(n - 1);
            if (p_bin > 0.0) {
                double bits_max   = std::log2(static_cast<double>(config_.n_bins));
                double self_info  = -std::log2(p_bin);
                norm_surprise = (bits_max > 1e-12) ? std::min(1.0, self_info / bits_max) : 1.0;
            }
            // If p_bin == 0 (first time in this bin), surprise = 1.0 (maximally surprising).
        }

        // Now update histogram with this observation.
        hist_[static_cast<size_t>(bin)]++;

        // EMA of normalized surprise (alpha = 0.05 → ≈ 20-observation smoothing).
        double prev_mean = mean_surprise_.load(std::memory_order_relaxed);
        mean_surprise_.store(prev_mean + 0.05 * (norm_surprise - prev_mean),
                             std::memory_order_relaxed);

        fire_high = (norm_surprise >= config_.high_surprise_threshold);
        if (fire_high)
            high_count_.fetch_add(1, std::memory_order_relaxed);
    }

    surprise_.store(norm_surprise, std::memory_order_relaxed);

    if (fire_high && cfg_copy.on_high_surprise)
        cfg_copy.on_high_surprise(bias, norm_surprise);
}

void SignalSurpriseIndex::update_config(Config cfg) {
    std::lock_guard<std::mutex> lk(mutex_);
    config_ = std::move(cfg);
    hist_.assign(static_cast<size_t>(config_.n_bins), 0ULL);
    threshold_cache_.store(config_.high_surprise_threshold, std::memory_order_relaxed);
    surprise_.store(1.0, std::memory_order_relaxed);
    mean_surprise_.store(1.0, std::memory_order_relaxed);
    total_.store(0, std::memory_order_relaxed);
    high_count_.store(0, std::memory_order_relaxed);
}

SignalSurpriseIndex::Config SignalSurpriseIndex::get_config() const {
    std::lock_guard<std::mutex> lk(mutex_);
    return config_;
}

void SignalSurpriseIndex::reset() {
    std::lock_guard<std::mutex> lk(mutex_);
    std::fill(hist_.begin(), hist_.end(), 0ULL);
    surprise_.store(1.0, std::memory_order_relaxed);
    mean_surprise_.store(1.0, std::memory_order_relaxed);
    total_.store(0, std::memory_order_relaxed);
    high_count_.store(0, std::memory_order_relaxed);
}

std::string SignalSurpriseIndex::to_stats_json() const {
    double s   = surprise_.load(std::memory_order_relaxed);
    double ms  = mean_surprise_.load(std::memory_order_relaxed);
    uint64_t t = total_.load(std::memory_order_relaxed);
    uint64_t h = high_count_.load(std::memory_order_relaxed);

    std::ostringstream o;
    o << std::fixed;
    o.precision(4);
    o << "{\"surprise\":"        << s
      << ",\"mean_surprise\":"   << ms
      << ",\"is_high_surprise\":" << (is_high_surprise() ? "true" : "false")
      << ",\"high_count\":"      << h
      << ",\"total_records\":"   << t << "}";
    return o.str();
}

} // namespace llmquant
