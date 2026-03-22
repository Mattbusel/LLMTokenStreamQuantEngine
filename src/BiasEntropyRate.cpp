#include "BiasEntropyRate.h"
#include <cmath>
#include <algorithm>
#include <sstream>
#include <iomanip>

namespace llmquant {

BiasEntropyRate::BiasEntropyRate(Config cfg)
    : cfg_(std::move(cfg))
    , buf_(cfg_.window, 0.0)
{}

void BiasEntropyRate::record(double bias) {
    double new_entropy = 0.0;
    bool high_ev = false;
    bool low_ev  = false;

    {
        std::lock_guard<std::mutex> lk(mutex_);

        buf_[head_] = bias;
        head_ = (head_ + 1) % cfg_.window;
        if (fill_ < cfg_.window) ++fill_;

        if (fill_ >= cfg_.min_samples) {
            int n = fill_;
            // Find range for bucketing
            double vmin = buf_[0], vmax = buf_[0];
            for (int i = 1; i < n; ++i) {
                if (buf_[i] < vmin) vmin = buf_[i];
                if (buf_[i] > vmax) vmax = buf_[i];
            }
            double range = vmax - vmin;

            // Compute histogram
            std::vector<int> hist(cfg_.n_buckets, 0);
            if (range > 1e-12) {
                for (int i = 0; i < n; ++i) {
                    int b = static_cast<int>((buf_[i] - vmin) / range * cfg_.n_buckets);
                    if (b >= cfg_.n_buckets) b = cfg_.n_buckets - 1;
                    ++hist[b];
                }
            } else {
                hist[0] = n; // all values identical → one bucket
            }

            // Shannon entropy
            double H = 0.0;
            double log2_k = std::log2(static_cast<double>(cfg_.n_buckets));
            for (int b = 0; b < cfg_.n_buckets; ++b) {
                if (hist[b] > 0) {
                    double p = static_cast<double>(hist[b]) / n;
                    H -= p * std::log2(p);
                }
            }
            new_entropy = (log2_k > 1e-12) ? H / log2_k : 0.0; // normalise to [0,1]

            bool h = (new_entropy > cfg_.high_threshold);
            bool l = (new_entropy < cfg_.low_threshold);
            if (h && !prev_high_) { high_ev = true; high_events_.fetch_add(1, std::memory_order_relaxed); }
            if (l && !prev_low_)  { low_ev  = true; low_events_.fetch_add(1, std::memory_order_relaxed); }
            prev_high_ = h;
            prev_low_  = l;
        }
    }

    entropy_.store(new_entropy, std::memory_order_relaxed);
    total_.fetch_add(1, std::memory_order_relaxed);

    if (high_ev && cfg_.on_high_entropy) cfg_.on_high_entropy(new_entropy);
    if (low_ev  && cfg_.on_low_entropy)  cfg_.on_low_entropy(new_entropy);
}

double BiasEntropyRate::entropy() const noexcept { return entropy_.load(std::memory_order_relaxed); }

std::size_t BiasEntropyRate::high_entropy_events() const noexcept { return high_events_.load(std::memory_order_relaxed); }
std::size_t BiasEntropyRate::low_entropy_events()  const noexcept { return low_events_.load(std::memory_order_relaxed); }
std::size_t BiasEntropyRate::total_records()       const noexcept { return total_.load(std::memory_order_relaxed); }

std::string BiasEntropyRate::to_stats_json() const {
    std::lock_guard<std::mutex> lk(mutex_);
    std::ostringstream ss;
    ss << std::fixed << std::setprecision(4);
    ss << "{\"entropy\":" << entropy_.load(std::memory_order_relaxed)
       << ",\"high_entropy_events\":" << high_events_.load(std::memory_order_relaxed)
       << ",\"low_entropy_events\":" << low_events_.load(std::memory_order_relaxed)
       << ",\"total_records\":" << total_.load(std::memory_order_relaxed)
       << "}";
    return ss.str();
}

void BiasEntropyRate::reset() {
    std::lock_guard<std::mutex> lk(mutex_);
    std::fill(buf_.begin(), buf_.end(), 0.0);
    head_ = 0; fill_ = 0; prev_high_ = false; prev_low_ = false;
    entropy_.store(0.0, std::memory_order_relaxed);
    high_events_.store(0, std::memory_order_relaxed);
    low_events_.store(0, std::memory_order_relaxed);
    total_.store(0, std::memory_order_relaxed);
}

void BiasEntropyRate::update_config(const Config& cfg) {
    std::lock_guard<std::mutex> lk(mutex_);
    cfg_ = cfg;
    buf_.assign(cfg_.window, 0.0);
    head_ = 0; fill_ = 0;
}

} // namespace llmquant
