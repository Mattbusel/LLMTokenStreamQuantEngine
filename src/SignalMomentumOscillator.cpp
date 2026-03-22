#include "SignalMomentumOscillator.h"

#include <cmath>
#include <sstream>

namespace llmquant {

void SignalMomentumOscillator::record(double bias) noexcept {
    total_.fetch_add(1, std::memory_order_relaxed);

    bool fire_div = false;
    bool fire_cross = false;
    CrossDirection cross_dir = CrossDirection::None;
    double macd_val = 0.0, sig_val = 0.0, hist_val = 0.0;

    Config cfg_copy;
    {
        std::lock_guard<std::mutex> lk(mutex_);
        cfg_copy = config_;

        if (!seeded_) {
            fast_ema_    = bias;
            slow_ema_    = bias;
            signal_ema_  = 0.0;
            seeded_ = true;
        } else {
            fast_ema_   += config_.fast_alpha   * (bias      - fast_ema_);
            slow_ema_   += config_.slow_alpha   * (bias      - slow_ema_);
        }

        macd_val  = fast_ema_ - slow_ema_;
        signal_ema_ += config_.signal_alpha * (macd_val - signal_ema_);
        sig_val   = signal_ema_;
        hist_val  = macd_val - sig_val;

        // Detect histogram zero-cross.
        if (prev_histogram_ * hist_val < 0.0) {
            cross_dir  = (hist_val > 0.0) ? CrossDirection::Bullish : CrossDirection::Bearish;
            fire_cross = true;
            crosses_.fetch_add(1, std::memory_order_relaxed);
            last_cross_.store(static_cast<int>(cross_dir), std::memory_order_relaxed);
        }
        prev_histogram_ = hist_val;

        fire_div = std::abs(hist_val) >= config_.divergence_threshold;
    }

    macd_.store(macd_val,  std::memory_order_relaxed);
    signal_line_.store(sig_val,  std::memory_order_relaxed);
    histogram_.store(hist_val, std::memory_order_relaxed);

    if (fire_div  && cfg_copy.on_divergence) cfg_copy.on_divergence(macd_val, sig_val, hist_val);
    if (fire_cross && cfg_copy.on_cross)     cfg_copy.on_cross(cross_dir, hist_val);
}

void SignalMomentumOscillator::update_config(Config cfg) {
    std::lock_guard<std::mutex> lk(mutex_);
    config_ = std::move(cfg);
    fast_ema_       = 0.0;
    slow_ema_       = 0.0;
    signal_ema_     = 0.0;
    prev_histogram_ = 0.0;
    seeded_         = false;
    macd_.store(0.0, std::memory_order_relaxed);
    signal_line_.store(0.0, std::memory_order_relaxed);
    histogram_.store(0.0, std::memory_order_relaxed);
    last_cross_.store(static_cast<int>(CrossDirection::None), std::memory_order_relaxed);
    total_.store(0, std::memory_order_relaxed);
    crosses_.store(0, std::memory_order_relaxed);
}

SignalMomentumOscillator::Config SignalMomentumOscillator::get_config() const {
    std::lock_guard<std::mutex> lk(mutex_);
    return config_;
}

void SignalMomentumOscillator::reset() {
    std::lock_guard<std::mutex> lk(mutex_);
    fast_ema_       = 0.0;
    slow_ema_       = 0.0;
    signal_ema_     = 0.0;
    prev_histogram_ = 0.0;
    seeded_         = false;
    macd_.store(0.0, std::memory_order_relaxed);
    signal_line_.store(0.0, std::memory_order_relaxed);
    histogram_.store(0.0, std::memory_order_relaxed);
    last_cross_.store(static_cast<int>(CrossDirection::None), std::memory_order_relaxed);
    total_.store(0, std::memory_order_relaxed);
    crosses_.store(0, std::memory_order_relaxed);
}

std::string SignalMomentumOscillator::to_stats_json() const {
    double m   = macd_.load(std::memory_order_relaxed);
    double s   = signal_line_.load(std::memory_order_relaxed);
    double h   = histogram_.load(std::memory_order_relaxed);
    int    lc  = last_cross_.load(std::memory_order_relaxed);
    uint64_t t = total_.load(std::memory_order_relaxed);
    uint64_t c = crosses_.load(std::memory_order_relaxed);

    const char* cross_str = (lc == static_cast<int>(CrossDirection::Bullish)) ? "bullish"
                          : (lc == static_cast<int>(CrossDirection::Bearish)) ? "bearish"
                          : "none";

    std::ostringstream o;
    o << std::fixed;
    o.precision(6);
    o << "{\"macd\":"       << m
      << ",\"signal_line\":" << s
      << ",\"histogram\":"  << h
      << ",\"last_cross\":\"" << cross_str << "\""
      << ",\"total_crosses\":" << c
      << ",\"total_records\":" << t << "}";
    return o.str();
}

} // namespace llmquant
