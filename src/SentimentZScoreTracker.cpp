#include "SentimentZScoreTracker.h"

#include <algorithm>
#include <cmath>
#include <sstream>
#include <iomanip>

namespace llmquant {

SentimentZScoreTracker::SentimentZScoreTracker(Config cfg)
    : cfg_(std::move(cfg)) {
    buf_.resize(static_cast<size_t>(std::max(cfg_.window, 2)), 0.0);
}

void SentimentZScoreTracker::record(double bias) {
    double fire_z = 0.0;
    bool   do_fire = false;

    {
        std::lock_guard<std::mutex> lk(mutex_);

        obs_count_.fetch_add(1, std::memory_order_relaxed);

        int sz = static_cast<int>(buf_.size());
        buf_[static_cast<size_t>(head_)] = bias;
        head_ = (head_ + 1) % sz;
        if (fill_ < sz) ++fill_;

        double sum = 0.0, sum2 = 0.0;
        for (int i = 0; i < fill_; ++i) {
            double v = buf_[static_cast<size_t>(i)];
            sum  += v;
            sum2 += v * v;
        }
        double mu  = sum / fill_;
        double var = (sum2 / fill_) - mu * mu;
        double sig = (var > 0.0) ? std::sqrt(var) : 0.0;

        mean_.store(mu,  std::memory_order_relaxed);
        sigma_.store(sig, std::memory_order_relaxed);

        double z = (sig > 1e-12) ? (bias - mu) / sig : 0.0;
        z_score_.store(z, std::memory_order_relaxed);

        bool now_extreme = (std::abs(z) >= cfg_.extreme_threshold);
        extreme_.store(now_extreme, std::memory_order_relaxed);

        if (now_extreme && !prev_extreme_ && cfg_.on_extreme) {
            fire_z = z;
            do_fire = true;
        }
        prev_extreme_ = now_extreme;
    }

    if (do_fire) cfg_.on_extreme(fire_z);
}

std::string SentimentZScoreTracker::to_stats_json() const {
    std::lock_guard<std::mutex> lk(mutex_);
    std::ostringstream oss;
    oss << std::fixed << std::setprecision(4);
    oss << "{\"z_score\":" << z_score_.load(std::memory_order_relaxed)
        << ",\"mean\":" << mean_.load(std::memory_order_relaxed)
        << ",\"sigma\":" << sigma_.load(std::memory_order_relaxed)
        << ",\"is_extreme\":" << (extreme_.load(std::memory_order_relaxed) ? "true" : "false")
        << ",\"obs_count\":" << obs_count_.load(std::memory_order_relaxed)
        << "}";
    return oss.str();
}

void SentimentZScoreTracker::reset() {
    std::lock_guard<std::mutex> lk(mutex_);
    std::fill(buf_.begin(), buf_.end(), 0.0);
    head_ = 0; fill_ = 0; prev_extreme_ = false;
    z_score_.store(0.0, std::memory_order_relaxed);
    mean_.store(0.0, std::memory_order_relaxed);
    sigma_.store(0.0, std::memory_order_relaxed);
    extreme_.store(false, std::memory_order_relaxed);
    obs_count_.store(0, std::memory_order_relaxed);
}

void SentimentZScoreTracker::update_config(const Config& cfg) {
    std::lock_guard<std::mutex> lk(mutex_);
    cfg_ = cfg;
    buf_.resize(static_cast<size_t>(std::max(cfg_.window, 2)), 0.0);
}

} // namespace llmquant
