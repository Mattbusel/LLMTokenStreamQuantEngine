#include "SignalReversalDetector.h"

#include <cmath>
#include <sstream>
#include <iomanip>

namespace llmquant {

SignalReversalDetector::SignalReversalDetector(Config cfg)
    : cfg_(std::move(cfg))
{}

void SignalReversalDetector::record(double bias) {
    bool   fire_bull = false, fire_bear = false;
    std::function<void(double)> bull_cb, bear_cb;

    {
        std::lock_guard<std::mutex> lk(mutex_);
        bull_cb = cfg_.on_bullish_reversal;
        bear_cb = cfg_.on_bearish_reversal;
        total_.fetch_add(1, std::memory_order_relaxed);

        if (!seeded_) {
            extremum_   = bias;
            seeded_     = true;
            thrust_dir_ = 0;
            return;
        }

        double min_thrust = std::max(1e-9, cfg_.min_thrust);

        if (thrust_dir_ == 0) {
            // Looking for initial thrust
            double move = bias - extremum_;
            if (std::abs(move) >= min_thrust) {
                thrust_dir_  = (move > 0.0) ? 1 : -1;
                thrust_size_ = std::abs(move);
                extremum_    = (thrust_dir_ > 0) ? bias : bias;  // track end of thrust
            }
        } else if (thrust_dir_ > 0) {
            // In a bullish thrust — track new highs; watch for counter-move
            if (bias > extremum_) {
                extremum_    = bias;
                thrust_size_ = extremum_ - (extremum_ - thrust_size_);
                // Reset thrust_size relative to latest extremum
                // (simplified: keep running extremum)
            } else {
                double counter = extremum_ - bias;
                if (counter >= cfg_.reversal_fraction * thrust_size_ && thrust_size_ >= min_thrust) {
                    // Bearish reversal
                    fire_bear   = true;
                    thrust_dir_ = -1;
                    extremum_   = bias;
                    thrust_size_ = counter;
                    bear_events_.fetch_add(1, std::memory_order_relaxed);
                    last_dir_.store(-1, std::memory_order_relaxed);
                }
            }
        } else {
            // In a bearish thrust — track new lows; watch for counter-move
            if (bias < extremum_) {
                extremum_ = bias;
            } else {
                double counter = bias - extremum_;
                if (counter >= cfg_.reversal_fraction * thrust_size_ && thrust_size_ >= min_thrust) {
                    // Bullish reversal
                    fire_bull   = true;
                    thrust_dir_ = 1;
                    extremum_   = bias;
                    thrust_size_ = counter;
                    bull_events_.fetch_add(1, std::memory_order_relaxed);
                    last_dir_.store(1, std::memory_order_relaxed);
                }
            }
        }
    }

    if (fire_bull && bull_cb) bull_cb(bias);
    if (fire_bear && bear_cb) bear_cb(bias);
}

std::string SignalReversalDetector::to_stats_json() const {
    std::lock_guard<std::mutex> lk(mutex_);
    std::ostringstream ss;
    ss << std::fixed << std::setprecision(6);
    ss << "{"
       << "\"last_reversal_direction\":" << last_dir_.load(std::memory_order_relaxed) << ","
       << "\"bullish_reversals\":"       << bull_events_.load(std::memory_order_relaxed) << ","
       << "\"bearish_reversals\":"       << bear_events_.load(std::memory_order_relaxed) << ","
       << "\"total_records\":"           << total_.load(std::memory_order_relaxed)
       << "}";
    return ss.str();
}

void SignalReversalDetector::reset() {
    std::lock_guard<std::mutex> lk(mutex_);
    seeded_     = false;
    extremum_   = 0.0;
    thrust_dir_ = 0;
    thrust_size_ = 0.0;
    last_dir_.store(0, std::memory_order_relaxed);
    bull_events_.store(0, std::memory_order_relaxed);
    bear_events_.store(0, std::memory_order_relaxed);
    total_.store(0, std::memory_order_relaxed);
}

void SignalReversalDetector::update_config(const Config& cfg) {
    std::lock_guard<std::mutex> lk(mutex_);
    cfg_        = cfg;
    seeded_     = false;
    extremum_   = 0.0;
    thrust_dir_ = 0;
    thrust_size_ = 0.0;
}

} // namespace llmquant
