#include "RollingSharpeBiasTracker.h"

#include <cmath>
#include <spdlog/spdlog.h>
#include <sstream>

namespace llmquant {

// ---------------------------------------------------------------------------
// Construction
// ---------------------------------------------------------------------------

RollingSharpeBiasTracker::RollingSharpeBiasTracker(Config config)
    : config_(std::move(config)) {}

// ---------------------------------------------------------------------------
// record
// ---------------------------------------------------------------------------

double RollingSharpeBiasTracker::record(double bias_value) {
    total_records_.fetch_add(1, std::memory_order_relaxed);

    double sharpe = 0.0;
    {
        std::lock_guard<std::mutex> lk(mutex_);

        window_.push_back(bias_value);
        sum_  += bias_value;
        sum2_ += bias_value * bias_value;
        if (static_cast<int>(window_.size()) > config_.window_size) {
            double evicted = window_.front();
            window_.pop_front();
            sum_  -= evicted;
            sum2_ -= evicted * evicted;
        }

        int n = static_cast<int>(window_.size());
        if (n >= config_.min_samples) {
            double mean = sum_ / n;
            double var  = (sum2_ / n) - (mean * mean);
            double sd   = (var > 0.0) ? std::sqrt(var) : 0.0;
            sharpe = (sd > 0.0) ? mean / sd : 0.0;
            last_sharpe_ = sharpe;
        } else {
            last_sharpe_ = 0.0;
        }

        bool pq = (last_sharpe_ < config_.min_sharpe);
        poor_quality_.store(pq, std::memory_order_relaxed);
    }

    return sharpe;
}

// ---------------------------------------------------------------------------
// reset
// ---------------------------------------------------------------------------

void RollingSharpeBiasTracker::reset() {
    std::lock_guard<std::mutex> lk(mutex_);
    window_.clear();
    sum_         = 0.0;
    sum2_        = 0.0;
    last_sharpe_ = 0.0;
    poor_quality_.store(false, std::memory_order_relaxed);
}

// ---------------------------------------------------------------------------
// Query
// ---------------------------------------------------------------------------

double RollingSharpeBiasTracker::last_sharpe() const noexcept {
    std::lock_guard<std::mutex> lk(mutex_);
    return last_sharpe_;
}

double RollingSharpeBiasTracker::rolling_mean() const noexcept {
    std::lock_guard<std::mutex> lk(mutex_);
    int n = static_cast<int>(window_.size());
    if (n == 0) return 0.0;
    return sum_ / n;
}

double RollingSharpeBiasTracker::rolling_stddev() const noexcept {
    std::lock_guard<std::mutex> lk(mutex_);
    int n = static_cast<int>(window_.size());
    if (n < 2) return 0.0;
    double mean = sum_ / n;
    double var  = (sum2_ / n) - (mean * mean);
    return (var > 0.0) ? std::sqrt(var) : 0.0;
}

int RollingSharpeBiasTracker::sample_count() const noexcept {
    std::lock_guard<std::mutex> lk(mutex_);
    return static_cast<int>(window_.size());
}

// ---------------------------------------------------------------------------
// update_config
// ---------------------------------------------------------------------------

void RollingSharpeBiasTracker::update_config(const Config& config) {
    std::lock_guard<std::mutex> lk(mutex_);
    config_ = config;
    window_.clear();
    sum_         = 0.0;
    sum2_        = 0.0;
    last_sharpe_ = 0.0;
    poor_quality_.store(false, std::memory_order_relaxed);
}

// ---------------------------------------------------------------------------
// to_stats_json
// ---------------------------------------------------------------------------

std::string RollingSharpeBiasTracker::to_stats_json() const {
    double sharpe = 0.0, mean = 0.0, sd = 0.0;
    int n = 0;
    {
        std::lock_guard<std::mutex> lk(mutex_);
        sharpe = last_sharpe_;
        n      = static_cast<int>(window_.size());
        if (n > 0) {
            mean = sum_ / n;
            double var = (sum2_ / n) - (mean * mean);
            sd = (var > 0.0) ? std::sqrt(var) : 0.0;
        }
    }
    std::ostringstream s;
    s << "{\"rolling_sharpe\":"   << sharpe
      << ",\"rolling_mean\":"     << mean
      << ",\"rolling_stddev\":"   << sd
      << ",\"sample_count\":"     << n
      << ",\"is_poor_quality\":"  << (is_poor_quality() ? "true" : "false")
      << ",\"total_records\":"    << total_records()
      << "}";
    return s.str();
}

} // namespace llmquant
