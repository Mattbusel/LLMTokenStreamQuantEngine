#include "BiasBootstrapCI.h"

#include <algorithm>
#include <cmath>
#include <numeric>
#include <sstream>
#include <iomanip>

namespace llmquant {

BiasBootstrapCI::BiasBootstrapCI(Config cfg)
    : cfg_(std::move(cfg))
{
    int w = std::max(4, cfg_.window);
    win_buf_.assign(static_cast<std::size_t>(w), 0.0);
}

void BiasBootstrapCI::run_bootstrap_locked() {
    int n = win_fill_;
    if (n < std::max(2, cfg_.min_samples)) return;

    int w = static_cast<int>(win_buf_.size());
    // Build snapshot
    std::vector<double> snap(static_cast<std::size_t>(n));
    for (int i = 0; i < n; ++i) {
        int idx = (win_head_ - n + i + w * 2) % w;
        snap[static_cast<std::size_t>(i)] = win_buf_[static_cast<std::size_t>(idx)];
    }

    // Compute window mean
    double sum = 0.0;
    for (double v : snap) sum += v;
    double mu = sum / static_cast<double>(n);
    mean_.store(mu, std::memory_order_relaxed);

    // Bootstrap: resample n times, compute mean of each resample
    int B = std::max(10, cfg_.n_bootstrap);
    std::vector<double> boot_means(static_cast<std::size_t>(B));
    std::uniform_int_distribution<int> dist(0, n - 1);
    for (int b = 0; b < B; ++b) {
        double bsum = 0.0;
        for (int i = 0; i < n; ++i)
            bsum += snap[static_cast<std::size_t>(dist(rng_))];
        boot_means[static_cast<std::size_t>(b)] = bsum / static_cast<double>(n);
    }

    std::sort(boot_means.begin(), boot_means.end());
    int lo_idx = static_cast<int>(0.025 * B);
    int hi_idx = static_cast<int>(0.975 * B);
    lo_idx = std::max(0, std::min(B - 1, lo_idx));
    hi_idx = std::max(0, std::min(B - 1, hi_idx));

    double lo = boot_means[static_cast<std::size_t>(lo_idx)];
    double hi = boot_means[static_cast<std::size_t>(hi_idx)];

    lo_.store(lo, std::memory_order_relaxed);
    hi_.store(hi, std::memory_order_relaxed);
}

void BiasBootstrapCI::record(double bias) {
    bool   fire_wide   = false;
    bool   fire_narrow = false;
    double lo = 0.0, hi = 0.0;
    std::function<void(double, double)> wide_cb, narrow_cb;

    {
        std::lock_guard<std::mutex> lk(mutex_);
        wide_cb   = cfg_.on_ci_wide;
        narrow_cb = cfg_.on_ci_narrow;
        total_.fetch_add(1, std::memory_order_relaxed);

        int w = static_cast<int>(win_buf_.size());
        if (win_fill_ < w) ++win_fill_;
        win_buf_[static_cast<std::size_t>(win_head_)] = bias;
        win_head_ = (win_head_ + 1) % w;

        uint64_t t = total_.load(std::memory_order_relaxed);
        if (static_cast<int>(t) % std::max(1, cfg_.recompute_interval) == 0) {
            run_bootstrap_locked();

            lo = lo_.load(std::memory_order_relaxed);
            hi = hi_.load(std::memory_order_relaxed);
            double width = hi - lo;

            if (width > cfg_.width_threshold) {
                fire_wide = true;
                wide_events_.fetch_add(1, std::memory_order_relaxed);
            }
            if (width < cfg_.width_threshold * 0.3) {
                fire_narrow = true;
            }
        }
    }

    if (fire_wide   && wide_cb)   wide_cb(lo, hi);
    if (fire_narrow && narrow_cb) narrow_cb(lo, hi);
}

std::string BiasBootstrapCI::to_stats_json() const {
    std::lock_guard<std::mutex> lk(mutex_);
    std::ostringstream ss;
    ss << std::fixed << std::setprecision(6);
    ss << "{"
       << "\"ci_lo\":"        << lo_.load(std::memory_order_relaxed) << ","
       << "\"ci_hi\":"        << hi_.load(std::memory_order_relaxed) << ","
       << "\"rolling_mean\":" << mean_.load(std::memory_order_relaxed) << ","
       << "\"wide_events\":"  << wide_events_.load(std::memory_order_relaxed) << ","
       << "\"total_records\":" << total_.load(std::memory_order_relaxed)
       << "}";
    return ss.str();
}

void BiasBootstrapCI::reset() {
    std::lock_guard<std::mutex> lk(mutex_);
    std::fill(win_buf_.begin(), win_buf_.end(), 0.0);
    win_head_ = win_fill_ = 0;
    lo_.store(0.0, std::memory_order_relaxed);
    hi_.store(0.0, std::memory_order_relaxed);
    mean_.store(0.0, std::memory_order_relaxed);
    wide_events_.store(0, std::memory_order_relaxed);
    total_.store(0, std::memory_order_relaxed);
    rng_.seed(42);
}

void BiasBootstrapCI::update_config(const Config& cfg) {
    std::lock_guard<std::mutex> lk(mutex_);
    cfg_ = cfg;
    int w = std::max(4, cfg_.window);
    win_buf_.assign(static_cast<std::size_t>(w), 0.0);
    win_head_ = win_fill_ = 0;
    rng_.seed(42);
}

} // namespace llmquant
