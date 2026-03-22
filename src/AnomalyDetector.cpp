#include "AnomalyDetector.h"

#include <cmath>
#include <limits>
#include <spdlog/spdlog.h>
#include <sstream>

namespace llmquant {

// ---------------------------------------------------------------------------
// Construction
// ---------------------------------------------------------------------------

AnomalyDetector::AnomalyDetector(Config config)
    : config_(std::move(config)) {}

// ---------------------------------------------------------------------------
// compute_z (static helper)
// ---------------------------------------------------------------------------

double AnomalyDetector::compute_z(double value, double mean,
                                  double stddev) noexcept {
    if (stddev <= 0.0)
        return 0.0;
    return (value - mean) / stddev;
}

// ---------------------------------------------------------------------------
// record
// ---------------------------------------------------------------------------

double AnomalyDetector::record(double value) {
    total_records_.fetch_add(1, std::memory_order_relaxed);

    AnomalyCallback soft_cb_copy, hard_cb_copy;
    bool fire_soft = false, fire_hard = false;
    AnomalyEvent event{};
    double z = std::numeric_limits<double>::quiet_NaN();

    {
        std::lock_guard<std::mutex> lk(mutex_);

        // Maintain rolling window.
        window_.push_back(value);
        sum_  += value;
        sum2_ += value * value;
        if (static_cast<int>(window_.size()) > config_.window_size) {
            double evicted = window_.front();
            window_.pop_front();
            sum_  -= evicted;
            sum2_ -= evicted * evicted;
        }

        int n = static_cast<int>(window_.size());
        if (n < config_.min_samples) {
            last_z_ = 0.0;
            // Return NaN to signal insufficient data.
            return std::numeric_limits<double>::quiet_NaN();
        }

        double mean   = sum_ / n;
        double var    = (sum2_ / n) - (mean * mean);
        double stddev = (var > 0.0) ? std::sqrt(var) : 0.0;
        z = compute_z(value, mean, stddev);
        last_z_ = z;

        double absz = std::abs(z);
        if (absz >= config_.hard_threshold) {
            event = {value, z, mean, stddev, Severity::Hard};
            fire_hard     = true;
            hard_cb_copy  = config_.hard_cb;
        } else if (absz >= config_.soft_threshold) {
            event = {value, z, mean, stddev, Severity::Soft};
            fire_soft     = true;
            soft_cb_copy  = config_.soft_cb;
        }
    }

    if (fire_hard) {
        hard_anomalies_.fetch_add(1, std::memory_order_relaxed);
        spdlog::warn("[anomaly:{}] HARD z={:.3f} val={:.6f}", config_.name,
                     event.z_score, event.value);
        if (hard_cb_copy) hard_cb_copy(event);
    } else if (fire_soft) {
        soft_anomalies_.fetch_add(1, std::memory_order_relaxed);
        spdlog::info("[anomaly:{}] soft z={:.3f} val={:.6f}", config_.name,
                     event.z_score, event.value);
        if (soft_cb_copy) soft_cb_copy(event);
    }

    return z;
}

// ---------------------------------------------------------------------------
// reset
// ---------------------------------------------------------------------------

void AnomalyDetector::reset() {
    std::lock_guard<std::mutex> lk(mutex_);
    window_.clear();
    sum_    = 0.0;
    sum2_   = 0.0;
    last_z_ = 0.0;
}

// ---------------------------------------------------------------------------
// Query accessors
// ---------------------------------------------------------------------------

double AnomalyDetector::last_z_score() const noexcept {
    std::lock_guard<std::mutex> lk(mutex_);
    return last_z_;
}

double AnomalyDetector::rolling_mean() const noexcept {
    std::lock_guard<std::mutex> lk(mutex_);
    int n = static_cast<int>(window_.size());
    if (n == 0) return 0.0;
    return sum_ / n;
}

double AnomalyDetector::rolling_stddev() const noexcept {
    std::lock_guard<std::mutex> lk(mutex_);
    int n = static_cast<int>(window_.size());
    if (n < 2) return 0.0;
    double mean = sum_ / n;
    double var  = (sum2_ / n) - (mean * mean);
    return (var > 0.0) ? std::sqrt(var) : 0.0;
}

int AnomalyDetector::sample_count() const noexcept {
    std::lock_guard<std::mutex> lk(mutex_);
    return static_cast<int>(window_.size());
}

// ---------------------------------------------------------------------------
// update_config
// ---------------------------------------------------------------------------

void AnomalyDetector::update_config(const Config& config) {
    std::lock_guard<std::mutex> lk(mutex_);
    config_ = config;
    window_.clear();
    sum_    = 0.0;
    sum2_   = 0.0;
    last_z_ = 0.0;
}

// ---------------------------------------------------------------------------
// to_stats_json
// ---------------------------------------------------------------------------

std::string AnomalyDetector::to_stats_json() const {
    double mean = 0.0, stddev = 0.0, last_z = 0.0;
    int n = 0;
    {
        std::lock_guard<std::mutex> lk(mutex_);
        n       = static_cast<int>(window_.size());
        last_z  = last_z_;
        if (n > 0) {
            mean = sum_ / n;
            double var = (sum2_ / n) - (mean * mean);
            stddev = (var > 0.0) ? std::sqrt(var) : 0.0;
        }
    }
    std::ostringstream s;
    s << "{\"name\":\""        << config_.name << "\""
      << ",\"sample_count\":"  << n
      << ",\"rolling_mean\":"  << mean
      << ",\"rolling_stddev\":" << stddev
      << ",\"last_z_score\":"  << last_z
      << ",\"total_records\":" << total_records()
      << ",\"soft_anomalies\":" << soft_anomalies()
      << ",\"hard_anomalies\":" << hard_anomalies()
      << "}";
    return s.str();
}

} // namespace llmquant
