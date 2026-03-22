#include "TokenEntropyMonitor.h"

#include <algorithm>
#include <cmath>
#include <sstream>
#include <unordered_map>

namespace llmquant {

static constexpr int kMaxWindowSize = 4096;

TokenEntropyMonitor::TokenEntropyMonitor(Config config) : config_(config) {
    int sz = std::clamp(config_.window_size, 2, kMaxWindowSize);
    config_.window_size = sz;
    ring_.assign(sz, 0);
    config_snapshot_focused_threshold_.store(config_.focused_threshold,
                                             std::memory_order_relaxed);
}

void TokenEntropyMonitor::record(uint64_t token_hash) noexcept {
    total_.fetch_add(1, std::memory_order_relaxed);
    std::lock_guard<std::mutex> lk(mutex_);
    ring_[head_] = token_hash;
    head_ = (head_ + 1) % static_cast<int>(ring_.size());
    if (fill_ < static_cast<int>(ring_.size())) ++fill_;
}

double TokenEntropyMonitor::compute_entropy_locked() const {
    if (fill_ < 2) return 0.0;

    // Build frequency table over the filled portion of the ring.
    std::unordered_map<uint64_t, int> freq;
    freq.reserve(fill_);
    for (int i = 0; i < fill_; ++i) {
        ++freq[ring_[i]];
    }

    double h     = 0.0;
    double inv_n = 1.0 / static_cast<double>(fill_);
    for (auto& [tok, cnt] : freq) {
        double p = static_cast<double>(cnt) * inv_n;
        h -= p * std::log2(p);
    }

    // Normalise to [0, 1] by dividing by the maximum possible entropy log2(W).
    double h_max = std::log2(static_cast<double>(fill_));
    return (h_max > 1e-12) ? h / h_max : 0.0;
}

double TokenEntropyMonitor::entropy() {
    std::lock_guard<std::mutex> lk(mutex_);
    double h = compute_entropy_locked();

    // Update EMA.
    double alpha = config_.ema_alpha;
    double prev  = entropy_ema_.load(std::memory_order_relaxed);
    double next  = prev + alpha * (h - prev);
    entropy_ema_.store(next, std::memory_order_relaxed);
    return h;
}

int TokenEntropyMonitor::unique_tokens() {
    std::lock_guard<std::mutex> lk(mutex_);
    if (fill_ == 0) return 0;
    std::unordered_map<uint64_t, int> freq;
    freq.reserve(fill_);
    for (int i = 0; i < fill_; ++i) ++freq[ring_[i]];
    return static_cast<int>(freq.size());
}

void TokenEntropyMonitor::update_config(const Config& config) {
    std::lock_guard<std::mutex> lk(mutex_);
    int new_sz  = std::clamp(config.window_size, 2, kMaxWindowSize);
    config_     = config;
    config_.window_size = new_sz;
    config_snapshot_focused_threshold_.store(config_.focused_threshold,
                                             std::memory_order_relaxed);
    if (static_cast<int>(ring_.size()) != new_sz) {
        ring_.assign(new_sz, 0);
        head_ = 0;
        fill_ = 0;
    }
}

Config TokenEntropyMonitor::get_config() const {
    std::lock_guard<std::mutex> lk(mutex_);
    return config_;
}

void TokenEntropyMonitor::reset() {
    std::lock_guard<std::mutex> lk(mutex_);
    std::fill(ring_.begin(), ring_.end(), 0);
    head_ = 0;
    fill_ = 0;
    total_.store(0, std::memory_order_relaxed);
    entropy_ema_.store(0.5, std::memory_order_relaxed);
}

std::string TokenEntropyMonitor::to_stats_json() {
    double h_ema = entropy_ema_.load(std::memory_order_relaxed);
    bool   foc   = is_focused();
    uint64_t tot = total_.load(std::memory_order_relaxed);
    int uniq;
    double fill_frac;
    {
        std::lock_guard<std::mutex> lk(mutex_);
        if (fill_ == 0) {
            uniq      = 0;
            fill_frac = 0.0;
        } else {
            std::unordered_map<uint64_t, int> freq;
            freq.reserve(fill_);
            for (int i = 0; i < fill_; ++i) ++freq[ring_[i]];
            uniq      = static_cast<int>(freq.size());
            fill_frac = static_cast<double>(fill_) / static_cast<double>(ring_.size());
        }
    }
    std::ostringstream o;
    o << std::fixed;
    o.precision(4);
    o << "{\"entropy_ema\":" << h_ema
      << ",\"is_focused\":"  << (foc ? "true" : "false")
      << ",\"unique_tokens\":" << uniq
      << ",\"window_fill\":" << fill_frac
      << ",\"total_recorded\":" << tot << "}";
    return o.str();
}

} // namespace llmquant
