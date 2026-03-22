#include "SignalCompressor.h"

#include <algorithm>
#include <cmath>
#include <sstream>
#include <iomanip>
#include <unordered_set>
#include <string>

namespace llmquant {

SignalCompressor::SignalCompressor(Config cfg)
    : cfg_(std::move(cfg))
{
    int w = std::max(4, cfg_.window);
    win_buf_.assign(static_cast<std::size_t>(w), 0);
}

int SignalCompressor::bucket_for(double bias) const {
    int n = std::max(2, cfg_.n_symbols);
    double rng = cfg_.range_max - cfg_.range_min;
    if (rng <= 0.0) return 0;
    double frac = (bias - cfg_.range_min) / rng;
    frac = std::max(0.0, std::min(1.0 - 1e-12, frac));
    return static_cast<int>(frac * n);
}

// LZ76 phrase count on the current window (oldest-first traversal).
int SignalCompressor::lz76_locked() const {
    int n = win_fill_;
    if (n < 2) return 1;
    int w = static_cast<int>(win_buf_.size());

    // Build sequence in chronological order
    // head_ points to the slot AFTER the newest; oldest is at (head_ - fill_) mod w
    std::vector<int> seq(static_cast<std::size_t>(n));
    for (int i = 0; i < n; ++i) {
        int idx = (win_head_ - n + i + w * 2) % w;
        seq[static_cast<std::size_t>(i)] = win_buf_[static_cast<std::size_t>(idx)];
    }

    // LZ76: count distinct phrases
    int phrases = 0;
    int i = 0;
    while (i < n) {
        int k = 1;
        // Find longest prefix of seq[i..] that appears in seq[0..i+k-2]
        while (i + k <= n) {
            // Check if seq[i..i+k-1] appears in seq[0..i+k-2]
            bool found = false;
            int end = i + k - 1;
            for (int j = 0; j <= end - k; ++j) {
                bool match = true;
                for (int l = 0; l < k; ++l) {
                    if (seq[static_cast<std::size_t>(j + l)] != seq[static_cast<std::size_t>(i + l)]) {
                        match = false; break;
                    }
                }
                if (match) { found = true; break; }
            }
            if (!found) break;
            ++k;
        }
        ++phrases;
        i += k;
    }
    return phrases;
}

void SignalCompressor::record(double bias) {
    bool   fire        = false;
    double old_norm    = 0.0;
    double new_norm    = 0.0;
    std::function<void(double, double)> cb;

    {
        std::lock_guard<std::mutex> lk(mutex_);
        cb = cfg_.on_complexity_change;
        total_.fetch_add(1, std::memory_order_relaxed);

        int w   = static_cast<int>(win_buf_.size());
        int sym = bucket_for(bias);

        if (win_fill_ < w) ++win_fill_;
        win_buf_[static_cast<std::size_t>(win_head_)] = sym;
        win_head_ = (win_head_ + 1) % w;

        if (win_fill_ < std::max(2, cfg_.min_samples)) return;

        int lzc = lz76_locked();
        lz_raw_.store(lzc, std::memory_order_relaxed);

        // Normalise: expected LZC for random sequence ≈ n / log2(n) * log2(n_symbols)
        double n     = static_cast<double>(win_fill_);
        double log2n = std::log2(n);
        double norm  = (log2n > 0.0) ? static_cast<double>(lzc) * log2n / n : 0.0;
        lz_norm_.store(norm, std::memory_order_relaxed);

        if (prev_lz_norm_ >= 0.0 && std::abs(norm - prev_lz_norm_) > cfg_.change_threshold) {
            old_norm = prev_lz_norm_;
            new_norm = norm;
            fire     = true;
        }
        prev_lz_norm_ = norm;
    }

    if (fire && cb) cb(old_norm, new_norm);
}

std::string SignalCompressor::to_stats_json() const {
    std::lock_guard<std::mutex> lk(mutex_);
    std::ostringstream ss;
    ss << std::fixed << std::setprecision(6);
    ss << "{"
       << "\"lz_complexity\":"          << lz_raw_.load(std::memory_order_relaxed) << ","
       << "\"normalised_complexity\":"  << lz_norm_.load(std::memory_order_relaxed) << ","
       << "\"total_records\":"          << total_.load(std::memory_order_relaxed)
       << "}";
    return ss.str();
}

void SignalCompressor::reset() {
    std::lock_guard<std::mutex> lk(mutex_);
    std::fill(win_buf_.begin(), win_buf_.end(), 0);
    win_head_ = win_fill_ = 0;
    prev_lz_norm_ = -1.0;
    lz_raw_.store(0, std::memory_order_relaxed);
    lz_norm_.store(0.0, std::memory_order_relaxed);
    total_.store(0, std::memory_order_relaxed);
}

void SignalCompressor::update_config(const Config& cfg) {
    std::lock_guard<std::mutex> lk(mutex_);
    cfg_ = cfg;
    int w = std::max(4, cfg_.window);
    win_buf_.assign(static_cast<std::size_t>(w), 0);
    win_head_ = win_fill_ = 0;
    prev_lz_norm_ = -1.0;
    lz_raw_.store(0, std::memory_order_relaxed);
    lz_norm_.store(0.0, std::memory_order_relaxed);
    total_.store(0, std::memory_order_relaxed);
}

} // namespace llmquant
