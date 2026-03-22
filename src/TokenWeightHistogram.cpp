#include "TokenWeightHistogram.h"

#include <algorithm>
#include <cmath>
#include <sstream>
#include <iomanip>

namespace llmquant {

TokenWeightHistogram::TokenWeightHistogram(Config cfg)
    : cfg_(std::move(cfg)) {
    counts_.fill(0);
}

void TokenWeightHistogram::record(double bias) {
    int  fire_mode = -1;
    bool do_fire   = false;

    {
        std::lock_guard<std::mutex> lk(mutex_);

        total_.fetch_add(1, std::memory_order_relaxed);

        double range = cfg_.range_max - cfg_.range_min;
        if (range <= 0.0) range = 2.0;

        double clamped = std::clamp(bias, cfg_.range_min, cfg_.range_max);
        double normalized = (clamped - cfg_.range_min) / range;
        int bucket = static_cast<int>(normalized * kBuckets);
        bucket = std::clamp(bucket, 0, kBuckets - 1);
        ++counts_[static_cast<size_t>(bucket)];

        // Find mode.
        int mode = 0;
        for (int i = 1; i < kBuckets; ++i) {
            if (counts_[static_cast<size_t>(i)] > counts_[static_cast<size_t>(mode)])
                mode = i;
        }
        mode_bucket_.store(mode, std::memory_order_relaxed);

        // Compute entropy.
        uint64_t tot = total_.load(std::memory_order_relaxed);
        double H = 0.0;
        if (tot > 0) {
            for (int i = 0; i < kBuckets; ++i) {
                double p = static_cast<double>(counts_[static_cast<size_t>(i)]) /
                           static_cast<double>(tot);
                if (p > 0.0) H -= p * std::log2(p);
            }
        }
        entropy_.store(H, std::memory_order_relaxed);

        if (mode != prev_mode_ && prev_mode_ >= 0 && cfg_.on_distribution_shift) {
            fire_mode = mode;
            do_fire   = true;
        }
        prev_mode_ = mode;
    }

    if (do_fire) cfg_.on_distribution_shift(fire_mode);
}

double TokenWeightHistogram::mode_value() const noexcept {
    std::lock_guard<std::mutex> lk(mutex_);
    int m = mode_bucket_.load(std::memory_order_relaxed);
    double bucket_width = (cfg_.range_max - cfg_.range_min) / kBuckets;
    return cfg_.range_min + (m + 0.5) * bucket_width;
}

uint64_t TokenWeightHistogram::bucket_count(int idx) const noexcept {
    if (idx < 0 || idx >= kBuckets) return 0;
    std::lock_guard<std::mutex> lk(mutex_);
    return counts_[static_cast<size_t>(idx)];
}

std::string TokenWeightHistogram::to_stats_json() const {
    std::lock_guard<std::mutex> lk(mutex_);
    std::ostringstream oss;
    oss << std::fixed << std::setprecision(4);
    oss << "{\"mode_bucket\":" << mode_bucket_.load(std::memory_order_relaxed)
        << ",\"entropy\":" << entropy_.load(std::memory_order_relaxed)
        << ",\"total\":" << total_.load(std::memory_order_relaxed)
        << ",\"buckets\":[";
    for (int i = 0; i < kBuckets; ++i) {
        if (i > 0) oss << ",";
        oss << counts_[static_cast<size_t>(i)];
    }
    oss << "]}";
    return oss.str();
}

void TokenWeightHistogram::reset() {
    std::lock_guard<std::mutex> lk(mutex_);
    counts_.fill(0);
    prev_mode_ = -1;
    mode_bucket_.store(0, std::memory_order_relaxed);
    entropy_.store(0.0, std::memory_order_relaxed);
    total_.store(0, std::memory_order_relaxed);
}

void TokenWeightHistogram::update_config(const Config& cfg) {
    std::lock_guard<std::mutex> lk(mutex_);
    cfg_ = cfg;
}

} // namespace llmquant
