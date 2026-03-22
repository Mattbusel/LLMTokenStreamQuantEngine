#include "TokenBiasHistogram.h"
#include <algorithm>
#include <cmath>
#include <sstream>
#include <iomanip>

namespace llmquant {

TokenBiasHistogram::TokenBiasHistogram(Config cfg)
    : cfg_(std::move(cfg))
{
    int nb = std::max(2, cfg_.n_bins);
    counts_.assign(static_cast<std::size_t>(nb), 0u);
}

void TokenBiasHistogram::recompute_locked() {
    uint64_t total_count = total_.load(std::memory_order_relaxed);
    if (total_count == 0) return;

    int nb = static_cast<int>(counts_.size());
    // Find peak bin
    int    best_bin  = 0;
    uint64_t best_cnt = 0;
    for (int b = 0; b < nb; ++b) {
        if (counts_[static_cast<std::size_t>(b)] > best_cnt) {
            best_cnt = counts_[static_cast<std::size_t>(b)];
            best_bin = b;
        }
    }

    // Shannon entropy
    double Hs = 0.0;
    for (int b = 0; b < nb; ++b) {
        double p = static_cast<double>(counts_[static_cast<std::size_t>(b)])
                 / static_cast<double>(total_count);
        if (p > 1e-14) Hs -= p * std::log2(p);
    }

    peak_bin_.store(best_bin, std::memory_order_relaxed);
    entropy_.store(Hs, std::memory_order_relaxed);

    if (static_cast<int>(total_count) >= cfg_.min_samples && prev_mode_ >= 0 && best_bin != prev_mode_) {
        mode_events_.fetch_add(1, std::memory_order_relaxed);
        if (cfg_.on_mode_change) cfg_.on_mode_change(best_bin, prev_mode_);
    }
    prev_mode_ = best_bin;
}

void TokenBiasHistogram::record(double bias) {
    {
        std::lock_guard<std::mutex> lk(mutex_);
        int nb = static_cast<int>(counts_.size());
        double range = cfg_.hi - cfg_.lo;
        if (range <= 0.0) range = 1.0;
        int bin = static_cast<int>((bias - cfg_.lo) / range * nb);
        bin = std::max(0, std::min(nb - 1, bin));
        counts_[static_cast<std::size_t>(bin)]++;
        total_.fetch_add(1, std::memory_order_relaxed);
        recompute_locked();
    }
}

uint64_t TokenBiasHistogram::bin_count(int bin) const noexcept {
    std::lock_guard<std::mutex> lk(mutex_);
    int nb = static_cast<int>(counts_.size());
    if (bin < 0 || bin >= nb) return 0;
    return counts_[static_cast<std::size_t>(bin)];
}

int TokenBiasHistogram::peak_bin()    const noexcept { return peak_bin_.load(std::memory_order_relaxed); }
double TokenBiasHistogram::entropy()  const noexcept { return entropy_.load(std::memory_order_relaxed); }

double TokenBiasHistogram::peak_center() const noexcept {
    int nb = static_cast<int>(counts_.size());
    if (nb == 0) return 0.0;
    int b = peak_bin_.load(std::memory_order_relaxed);
    double bin_width = (cfg_.hi - cfg_.lo) / nb;
    return cfg_.lo + (b + 0.5) * bin_width;
}

std::size_t TokenBiasHistogram::mode_changes()  const noexcept { return mode_events_.load(std::memory_order_relaxed); }
std::size_t TokenBiasHistogram::total_records() const noexcept { return total_.load(std::memory_order_relaxed); }

std::string TokenBiasHistogram::to_stats_json() const {
    std::lock_guard<std::mutex> lk(mutex_);
    std::ostringstream ss;
    ss << std::fixed << std::setprecision(4);
    ss << "{\"peak_bin\":"    << peak_bin_.load(std::memory_order_relaxed)
       << ",\"entropy\":"     << entropy_.load(std::memory_order_relaxed)
       << ",\"mode_changes\":" << mode_events_.load(std::memory_order_relaxed)
       << ",\"total_records\":" << total_.load(std::memory_order_relaxed)
       << "}";
    return ss.str();
}

void TokenBiasHistogram::reset() {
    std::lock_guard<std::mutex> lk(mutex_);
    std::fill(counts_.begin(), counts_.end(), 0u);
    prev_mode_ = -1;
    peak_bin_.store(0,   std::memory_order_relaxed);
    entropy_.store(0.0,  std::memory_order_relaxed);
    mode_events_.store(0, std::memory_order_relaxed);
    total_.store(0,       std::memory_order_relaxed);
}

void TokenBiasHistogram::update_config(const Config& cfg) {
    std::lock_guard<std::mutex> lk(mutex_);
    cfg_ = cfg;
    int nb = std::max(2, cfg_.n_bins);
    counts_.assign(static_cast<std::size_t>(nb), 0u);
    prev_mode_ = -1;
    peak_bin_.store(0,   std::memory_order_relaxed);
    entropy_.store(0.0,  std::memory_order_relaxed);
    mode_events_.store(0, std::memory_order_relaxed);
    total_.store(0,       std::memory_order_relaxed);
}

} // namespace llmquant
