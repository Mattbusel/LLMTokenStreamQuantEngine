#include "SentimentAutocorrelationMeter.h"

#include <algorithm>
#include <cmath>
#include <numeric>
#include <sstream>
#include <iomanip>

namespace llmquant {

SentimentAutocorrelationMeter::SentimentAutocorrelationMeter(Config cfg)
    : cfg_(std::move(cfg))
{
    int w = std::max(2, cfg_.window);
    buf_.resize(static_cast<std::size_t>(w), 0.0);
    std::fill(std::begin(ac_), std::end(ac_), 0.0);
}

void SentimentAutocorrelationMeter::record(double bias) {
    std::function<void(double, bool)> cb;
    bool   fire     = false;
    double lag1_ac  = 0.0;
    bool   trending = false;

    {
        std::lock_guard<std::mutex> lk(mutex_);
        cb = cfg_.on_regime_change;
        obs_count_.fetch_add(1, std::memory_order_relaxed);

        buf_[static_cast<std::size_t>(head_)] = bias;
        head_ = (head_ + 1) % static_cast<int>(buf_.size());
        if (fill_ < static_cast<int>(buf_.size())) ++fill_;

        if (fill_ >= std::max(2, cfg_.max_lag + 1)) {
            recompute_locked();

            lag1_ac  = ac_[1];
            trending = std::abs(lag1_ac) > cfg_.trending_threshold;
            trending_.store(trending, std::memory_order_relaxed);

            if (trending_seeded_ && trending != prev_trending_) {
                fire = true;
            }
            prev_trending_   = trending;
            trending_seeded_ = true;
        }
    }

    if (fire && cb) cb(lag1_ac, trending);
}

void SentimentAutocorrelationMeter::recompute_locked() {
    int n    = fill_;
    int size = static_cast<int>(buf_.size());
    int max_lag = std::min(cfg_.max_lag, kMaxLag);

    // Compute mean of the filled portion.
    double sum = 0.0;
    for (int i = 0; i < n; ++i) {
        int idx = (head_ - 1 - i + size) % size;
        sum += buf_[static_cast<std::size_t>(idx)];
    }
    double mu = sum / n;

    // Compute variance.
    double var = 0.0;
    for (int i = 0; i < n; ++i) {
        int idx = (head_ - 1 - i + size) % size;
        double d = buf_[static_cast<std::size_t>(idx)] - mu;
        var += d * d;
    }
    var /= n;
    if (var <= 0.0) {
        std::fill(std::begin(ac_), std::end(ac_), 0.0);
        return;
    }

    // Compute each lag.
    for (int k = 1; k <= max_lag; ++k) {
        if (n <= k) { ac_[k] = 0.0; continue; }
        double cov = 0.0;
        int pairs  = n - k;
        for (int i = 0; i < pairs; ++i) {
            int i0 = (head_ - 1 - i + size) % size;
            int ik = (head_ - 1 - i - k + size * 2) % size;
            cov += (buf_[static_cast<std::size_t>(i0)] - mu) *
                   (buf_[static_cast<std::size_t>(ik)] - mu);
        }
        ac_[k] = (cov / pairs) / var;
    }
}

double SentimentAutocorrelationMeter::autocorrelation(int lag) const noexcept {
    std::lock_guard<std::mutex> lk(mutex_);
    if (lag < 1 || lag > kMaxLag) return 0.0;
    return ac_[lag];
}

std::string SentimentAutocorrelationMeter::to_stats_json() const {
    std::lock_guard<std::mutex> lk(mutex_);
    std::ostringstream ss;
    ss << std::fixed << std::setprecision(6);
    ss << "{"
       << "\"observation_count\":" << obs_count_.load(std::memory_order_relaxed) << ","
       << "\"is_trending\":"       << (trending_.load(std::memory_order_relaxed) ? "true" : "false") << ","
       << "\"autocorrelations\":[";
    int max_lag = std::min(cfg_.max_lag, kMaxLag);
    for (int k = 1; k <= max_lag; ++k) {
        if (k > 1) ss << ",";
        ss << "{\"lag\":" << k << ",\"ac\":" << ac_[k] << "}";
    }
    ss << "]}";
    return ss.str();
}

void SentimentAutocorrelationMeter::reset() {
    std::lock_guard<std::mutex> lk(mutex_);
    std::fill(buf_.begin(), buf_.end(), 0.0);
    head_ = fill_ = 0;
    std::fill(std::begin(ac_), std::end(ac_), 0.0);
    prev_trending_   = false;
    trending_seeded_ = false;
    trending_.store(false, std::memory_order_relaxed);
    obs_count_.store(0, std::memory_order_relaxed);
}

void SentimentAutocorrelationMeter::update_config(const Config& cfg) {
    std::lock_guard<std::mutex> lk(mutex_);
    cfg_ = cfg;
    int w = std::max(2, cfg_.window);
    buf_.assign(static_cast<std::size_t>(w), 0.0);
    head_ = fill_ = 0;
    std::fill(std::begin(ac_), std::end(ac_), 0.0);
    prev_trending_   = false;
    trending_seeded_ = false;
}

} // namespace llmquant
