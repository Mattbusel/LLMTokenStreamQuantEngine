#include "SignalFatigueMeter.h"

#include <algorithm>
#include <cmath>
#include <sstream>
#include <iomanip>

namespace llmquant {

SignalFatigueMeter::SignalFatigueMeter(Config cfg)
    : cfg_(std::move(cfg)) {}

void SignalFatigueMeter::record(double bias) {
    int    fire_streak = 0;
    double fire_score  = 0.0;
    bool   do_fire     = false;

    {
        std::lock_guard<std::mutex> lk(mutex_);

        obs_count_.fetch_add(1, std::memory_order_relaxed);

        int dir = 0;
        if (bias > 0.0)       dir = +1;
        else if (bias < 0.0)  dir = -1;

        if (dir == 0) {
            // neutral — break streak
            cur_streak_    = 0;
            cur_direction_ = 0;
        } else if (dir == cur_direction_) {
            ++cur_streak_;
        } else {
            cur_streak_    = 1;
            cur_direction_ = dir;
        }

        streak_.store(cur_streak_, std::memory_order_relaxed);
        direction_.store(cur_direction_, std::memory_order_relaxed);

        double score = static_cast<double>(cur_streak_) /
                       static_cast<double>(std::max(cfg_.saturation_streak, 1));
        score = std::clamp(score, 0.0, 1.0);
        fatigue_score_.store(score, std::memory_order_relaxed);

        bool now_fatigued = (cur_streak_ >= cfg_.fatigue_threshold);
        fatigued_.store(now_fatigued, std::memory_order_relaxed);

        if (now_fatigued && !prev_fatigued_ && cfg_.on_fatigue) {
            fire_streak = cur_streak_;
            fire_score  = score;
            do_fire     = true;
        }
        prev_fatigued_ = now_fatigued;
    }

    if (do_fire) cfg_.on_fatigue(fire_streak, fire_score);
}

std::string SignalFatigueMeter::to_stats_json() const {
    std::lock_guard<std::mutex> lk(mutex_);
    std::ostringstream oss;
    oss << std::fixed << std::setprecision(4);
    oss << "{\"fatigue_score\":" << fatigue_score_.load(std::memory_order_relaxed)
        << ",\"streak\":" << streak_.load(std::memory_order_relaxed)
        << ",\"direction\":" << direction_.load(std::memory_order_relaxed)
        << ",\"is_fatigued\":" << (fatigued_.load(std::memory_order_relaxed) ? "true" : "false")
        << ",\"obs_count\":" << obs_count_.load(std::memory_order_relaxed)
        << "}";
    return oss.str();
}

void SignalFatigueMeter::reset() {
    std::lock_guard<std::mutex> lk(mutex_);
    cur_streak_    = 0;
    cur_direction_ = 0;
    prev_fatigued_ = false;
    fatigue_score_.store(0.0, std::memory_order_relaxed);
    streak_.store(0, std::memory_order_relaxed);
    direction_.store(0, std::memory_order_relaxed);
    fatigued_.store(false, std::memory_order_relaxed);
    obs_count_.store(0, std::memory_order_relaxed);
}

void SignalFatigueMeter::update_config(const Config& cfg) {
    std::lock_guard<std::mutex> lk(mutex_);
    cfg_ = cfg;
}

} // namespace llmquant
