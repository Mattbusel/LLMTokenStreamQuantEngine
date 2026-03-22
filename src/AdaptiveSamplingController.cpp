#include "AdaptiveSamplingController.h"

#include <algorithm>
#include <cmath>
#include <sstream>

namespace llmquant {

AdaptiveSamplingController::AdaptiveSamplingController(Config config)
    : config_(std::move(config))
    , interval_ms_(config_.initial_interval_ms) {}

void AdaptiveSamplingController::record_activity(double activity) {
    total_records_.fetch_add(1, std::memory_order_relaxed);

    bool fire_cb    = false;
    int64_t new_iv  = 0;
    std::function<void(int64_t)> cb;

    {
        std::lock_guard<std::mutex> lk(mutex_);

        double old_iv = static_cast<double>(interval_ms_);
        double next_iv = old_iv;

        if (activity >= config_.high_activity_threshold) {
            next_iv = old_iv * config_.decay_factor;
            accelerations_.fetch_add(1, std::memory_order_relaxed);
        } else if (activity <= config_.low_activity_threshold) {
            next_iv = old_iv * config_.growth_factor;
            decelerations_.fetch_add(1, std::memory_order_relaxed);
        }

        // Clamp to [min, max].
        next_iv = std::max(static_cast<double>(config_.min_interval_ms),
                           std::min(static_cast<double>(config_.max_interval_ms), next_iv));

        int64_t new_interval = static_cast<int64_t>(std::round(next_iv));

        if (new_interval != interval_ms_) {
            fire_cb   = true;
            new_iv    = new_interval;
            cb        = config_.on_interval_change;
        }
        interval_ms_ = new_interval;
    }

    if (fire_cb && cb) cb(new_iv);
}

void AdaptiveSamplingController::reset() {
    std::lock_guard<std::mutex> lk(mutex_);
    interval_ms_ = config_.initial_interval_ms;
}

int64_t AdaptiveSamplingController::recommended_interval_ms() const noexcept {
    std::lock_guard<std::mutex> lk(mutex_);
    return interval_ms_;
}

bool AdaptiveSamplingController::is_at_min() const noexcept {
    std::lock_guard<std::mutex> lk(mutex_);
    return interval_ms_ <= config_.min_interval_ms;
}

bool AdaptiveSamplingController::is_at_max() const noexcept {
    std::lock_guard<std::mutex> lk(mutex_);
    return interval_ms_ >= config_.max_interval_ms;
}

void AdaptiveSamplingController::update_config(const Config& config) {
    std::lock_guard<std::mutex> lk(mutex_);
    config_      = config;
    interval_ms_ = config_.initial_interval_ms;
}

std::string AdaptiveSamplingController::to_stats_json() const {
    std::lock_guard<std::mutex> lk(mutex_);
    std::ostringstream os;
    os << "{"
       << "\"interval_ms\":"     << interval_ms_ << ","
       << "\"min_interval_ms\":" << config_.min_interval_ms << ","
       << "\"max_interval_ms\":" << config_.max_interval_ms << ","
       << "\"total_records\":"   << total_records_.load(std::memory_order_relaxed) << ","
       << "\"accelerations\":"   << accelerations_.load(std::memory_order_relaxed) << ","
       << "\"decelerations\":"   << decelerations_.load(std::memory_order_relaxed)
       << "}";
    return os.str();
}

} // namespace llmquant
