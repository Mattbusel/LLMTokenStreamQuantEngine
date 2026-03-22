#include "SignalPersistenceTracker.h"

#include <algorithm>
#include <cmath>
#include <spdlog/spdlog.h>
#include <sstream>

namespace llmquant {

// ---------------------------------------------------------------------------
// Construction
// ---------------------------------------------------------------------------

SignalPersistenceTracker::SignalPersistenceTracker(Config config)
    : config_(std::move(config)) {}

// ---------------------------------------------------------------------------
// record
// ---------------------------------------------------------------------------

int SignalPersistenceTracker::record(Direction dir) {
    if (dir != Direction::Neutral)
        total_signals_.fetch_add(1, std::memory_order_relaxed);

    std::function<void(int)>           conv_cb;
    std::function<void(int, Direction)> rev_cb;
    bool fire_conv = false, fire_rev = false;
    int  prev_streak = 0;
    Direction prev_dir = Direction::Neutral;
    int  new_streak = 0;

    {
        std::lock_guard<std::mutex> lk(mutex_);

        if (dir == Direction::Neutral) {
            // Neutral signals neither extend nor break the streak.
            return streak_;
        }

        if (direction_ == Direction::Neutral) {
            // First directional signal: start the streak.
            direction_ = dir;
            streak_    = 1;
            conviction_fired_ = false;
        } else if (dir == direction_) {
            // Same direction: extend streak.
            ++streak_;
        } else {
            // Reversal.
            prev_streak = streak_;
            prev_dir    = direction_;
            fire_rev    = true;
            rev_cb      = config_.on_reversal;

            direction_ = dir;
            streak_    = 1;
            conviction_fired_ = false;
        }

        // Check conviction threshold (edge-triggered).
        if (streak_ >= config_.conviction_threshold && !conviction_fired_) {
            conviction_fired_ = true;
            fire_conv = true;
            conv_cb   = config_.on_conviction;
        }

        new_streak = streak_;
    }

    if (fire_rev) {
        total_reversals_.fetch_add(1, std::memory_order_relaxed);
        spdlog::debug("[persistence] reversal streak={} dir={}",
                      prev_streak, prev_dir == Direction::Bullish ? "BULL" : "BEAR");
        if (rev_cb) rev_cb(prev_streak, prev_dir);
    }
    if (fire_conv) {
        conviction_events_.fetch_add(1, std::memory_order_relaxed);
        spdlog::info("[persistence] conviction streak={}", new_streak);
        if (conv_cb) conv_cb(new_streak);
    }

    return new_streak;
}

// ---------------------------------------------------------------------------
// record_bias
// ---------------------------------------------------------------------------

int SignalPersistenceTracker::record_bias(double bias_value, double deadband) {
    Direction dir = Direction::Neutral;
    if (bias_value > deadband)
        dir = Direction::Bullish;
    else if (bias_value < -deadband)
        dir = Direction::Bearish;
    return record(dir);
}

// ---------------------------------------------------------------------------
// reset
// ---------------------------------------------------------------------------

void SignalPersistenceTracker::reset() {
    std::lock_guard<std::mutex> lk(mutex_);
    streak_           = 0;
    direction_        = Direction::Neutral;
    conviction_fired_ = false;
}

// ---------------------------------------------------------------------------
// Query
// ---------------------------------------------------------------------------

int SignalPersistenceTracker::current_streak() const noexcept {
    std::lock_guard<std::mutex> lk(mutex_);
    return streak_;
}

SignalPersistenceTracker::Direction
SignalPersistenceTracker::current_direction() const noexcept {
    std::lock_guard<std::mutex> lk(mutex_);
    return direction_;
}

double SignalPersistenceTracker::conviction_scale() const noexcept {
    std::lock_guard<std::mutex> lk(mutex_);
    if (config_.conviction_threshold <= 0) return 1.0;
    double scale = 1.0 + static_cast<double>(streak_) /
                         static_cast<double>(config_.conviction_threshold);
    return std::min(scale, config_.max_scale);
}

// ---------------------------------------------------------------------------
// update_config
// ---------------------------------------------------------------------------

void SignalPersistenceTracker::update_config(const Config& config) {
    std::lock_guard<std::mutex> lk(mutex_);
    config_ = config;
}

// ---------------------------------------------------------------------------
// to_stats_json
// ---------------------------------------------------------------------------

std::string SignalPersistenceTracker::to_stats_json() const {
    int streak = 0;
    Direction dir = Direction::Neutral;
    double scale = 1.0;
    {
        std::lock_guard<std::mutex> lk(mutex_);
        streak = streak_;
        dir    = direction_;
        if (config_.conviction_threshold > 0) {
            scale = 1.0 + static_cast<double>(streak_) /
                          static_cast<double>(config_.conviction_threshold);
            scale = std::min(scale, config_.max_scale);
        }
    }
    const char* dir_str = (dir == Direction::Bullish) ? "bull"
                        : (dir == Direction::Bearish)  ? "bear" : "neutral";
    std::ostringstream s;
    s << "{\"current_streak\":"      << streak
      << ",\"direction\":\""         << dir_str << "\""
      << ",\"conviction_scale\":"    << scale
      << ",\"total_signals\":"       << total_signals()
      << ",\"total_reversals\":"     << total_reversals()
      << ",\"conviction_events\":"   << conviction_events()
      << "}";
    return s.str();
}

} // namespace llmquant
