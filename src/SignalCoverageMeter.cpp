#include "SignalCoverageMeter.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <sstream>
#include <iomanip>

namespace llmquant {

SignalCoverageMeter::SignalCoverageMeter(Config cfg)
    : cfg_(std::move(cfg)) {
    buf_.resize(static_cast<size_t>(std::max(cfg_.window, 2)), 0.0);
}

void SignalCoverageMeter::record(double bias) {
    double fire_cov = 0.0;
    bool   do_fire  = false;

    {
        std::lock_guard<std::mutex> lk(mutex_);

        obs_count_.fetch_add(1, std::memory_order_relaxed);

        int sz = static_cast<int>(buf_.size());
        buf_[static_cast<size_t>(head_)] = bias;
        head_ = (head_ + 1) % sz;
        if (fill_ < sz) ++fill_;

        // Compute rolling min/max.
        double rmin =  std::numeric_limits<double>::max();
        double rmax = -std::numeric_limits<double>::max();
        for (int i = 0; i < fill_; ++i) {
            double v = buf_[static_cast<size_t>(i)];
            if (v < rmin) rmin = v;
            if (v > rmax) rmax = v;
        }

        rolling_min_.store(rmin, std::memory_order_relaxed);
        rolling_max_.store(rmax, std::memory_order_relaxed);

        double total_range = cfg_.range_max - cfg_.range_min;
        double cov = 0.0;
        if (total_range > 1e-12 && fill_ > 0) {
            cov = (rmax - rmin) / total_range;
            cov = std::clamp(cov, 0.0, 1.0);
        }
        coverage_.store(cov, std::memory_order_relaxed);

        bool now_expanded = (cov >= cfg_.expansion_threshold);
        expanded_.store(now_expanded, std::memory_order_relaxed);

        if (now_expanded && !prev_expanded_ && cfg_.on_range_expansion) {
            fire_cov = cov;
            do_fire  = true;
        }
        prev_expanded_ = now_expanded;
    }

    if (do_fire) cfg_.on_range_expansion(fire_cov);
}

std::string SignalCoverageMeter::to_stats_json() const {
    std::lock_guard<std::mutex> lk(mutex_);
    std::ostringstream oss;
    oss << std::fixed << std::setprecision(4);
    oss << "{\"coverage\":" << coverage_.load(std::memory_order_relaxed)
        << ",\"rolling_min\":" << rolling_min_.load(std::memory_order_relaxed)
        << ",\"rolling_max\":" << rolling_max_.load(std::memory_order_relaxed)
        << ",\"is_expanded\":" << (expanded_.load(std::memory_order_relaxed) ? "true" : "false")
        << ",\"obs_count\":" << obs_count_.load(std::memory_order_relaxed)
        << "}";
    return oss.str();
}

void SignalCoverageMeter::reset() {
    std::lock_guard<std::mutex> lk(mutex_);
    std::fill(buf_.begin(), buf_.end(), 0.0);
    head_         = 0;
    fill_         = 0;
    prev_expanded_ = false;
    coverage_.store(0.0, std::memory_order_relaxed);
    rolling_min_.store(0.0, std::memory_order_relaxed);
    rolling_max_.store(0.0, std::memory_order_relaxed);
    expanded_.store(false, std::memory_order_relaxed);
    obs_count_.store(0, std::memory_order_relaxed);
}

void SignalCoverageMeter::update_config(const Config& cfg) {
    std::lock_guard<std::mutex> lk(mutex_);
    cfg_ = cfg;
    buf_.resize(static_cast<size_t>(std::max(cfg_.window, 2)), 0.0);
}

} // namespace llmquant
