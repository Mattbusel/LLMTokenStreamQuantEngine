#include "SignalGainLossRatio.h"

#include <algorithm>
#include <cmath>
#include <iomanip>
#include <sstream>

namespace llmquant {

SignalGainLossRatio::SignalGainLossRatio(Config cfg)
    : cfg_(std::move(cfg))
{
    int w = std::max(4, cfg_.window);
    delta_ring_.resize(static_cast<std::size_t>(w), 0.0);
}

void SignalGainLossRatio::record(double bias) {
    std::function<void(double)> gain_cb, loss_cb;
    bool fire_gain = false, fire_loss = false;
    double ratio_val = 1.0;

    {
        std::lock_guard<std::mutex> lk(mutex_);
        gain_cb = cfg_.on_high_gain;
        loss_cb = cfg_.on_high_loss;

        total_.fetch_add(1, std::memory_order_relaxed);

        double delta = 0.0;
        if (!seeded_) {
            prev_bias_ = bias;
            seeded_    = true;
        } else {
            delta      = bias - prev_bias_;
            prev_bias_ = bias;
        }

        // Evict oldest delta from running sums.
        double old_delta = delta_ring_[static_cast<std::size_t>(head_)];
        if (old_delta > 0.0) gain_sum_v_ -= old_delta;
        else if (old_delta < 0.0) loss_sum_v_ -= std::abs(old_delta);

        // Insert new delta.
        delta_ring_[static_cast<std::size_t>(head_)] = delta;
        head_ = (head_ + 1) % static_cast<int>(delta_ring_.size());
        if (fill_ < static_cast<int>(delta_ring_.size())) ++fill_;

        if (delta > 0.0) gain_sum_v_ += delta;
        else if (delta < 0.0) loss_sum_v_ += std::abs(delta);

        // Guard negative drift from floating-point errors.
        if (gain_sum_v_ < 0.0) gain_sum_v_ = 0.0;
        if (loss_sum_v_ < 0.0) loss_sum_v_ = 0.0;

        gain_sum_.store(gain_sum_v_, std::memory_order_relaxed);
        loss_sum_.store(loss_sum_v_, std::memory_order_relaxed);

        int min_obs = std::max(2, cfg_.min_samples);
        if (fill_ >= min_obs && loss_sum_v_ > 1e-12) {
            ratio_val = gain_sum_v_ / loss_sum_v_;
            ratio_val = std::min(ratio_val, cfg_.max_ratio);
        } else if (fill_ >= min_obs && loss_sum_v_ <= 1e-12 && gain_sum_v_ > 0.0) {
            ratio_val = cfg_.max_ratio;  // all gains, no losses
        } else {
            ratio_val = 1.0;
        }
        ratio_.store(ratio_val, std::memory_order_relaxed);

        bool now_high_gain = ratio_val >= cfg_.high_gain_threshold;
        bool now_high_loss = ratio_val <= cfg_.high_loss_threshold;

        if (now_high_gain && !prev_high_gain_) {
            fire_gain = true;
            high_gain_events_.fetch_add(1, std::memory_order_relaxed);
        }
        if (now_high_loss && !prev_high_loss_) {
            fire_loss = true;
            high_loss_events_.fetch_add(1, std::memory_order_relaxed);
        }
        prev_high_gain_ = now_high_gain;
        prev_high_loss_ = now_high_loss;
    }

    if (fire_gain && gain_cb) gain_cb(ratio_val);
    if (fire_loss && loss_cb) loss_cb(ratio_val);
}

std::string SignalGainLossRatio::to_stats_json() const {
    double r, gs, ls;
    uint64_t ge, le, total;
    {
        std::lock_guard<std::mutex> lk(mutex_);
        r     = ratio_.load(std::memory_order_relaxed);
        gs    = gain_sum_.load(std::memory_order_relaxed);
        ls    = loss_sum_.load(std::memory_order_relaxed);
        ge    = high_gain_events_.load(std::memory_order_relaxed);
        le    = high_loss_events_.load(std::memory_order_relaxed);
        total = total_.load(std::memory_order_relaxed);
    }
    std::ostringstream ss;
    ss << std::fixed << std::setprecision(6);
    ss << "{"
       << "\"ratio\":"             << r     << ","
       << "\"gain_sum\":"          << gs    << ","
       << "\"loss_sum\":"          << ls    << ","
       << "\"high_gain_events\":"  << ge    << ","
       << "\"high_loss_events\":"  << le    << ","
       << "\"total_records\":"     << total
       << "}";
    return ss.str();
}

void SignalGainLossRatio::reset() {
    std::lock_guard<std::mutex> lk(mutex_);
    std::fill(delta_ring_.begin(), delta_ring_.end(), 0.0);
    head_ = fill_ = 0;
    gain_sum_v_    = loss_sum_v_ = 0.0;
    prev_bias_     = 0.0;
    seeded_        = false;
    prev_high_gain_ = prev_high_loss_ = false;
    ratio_.store(1.0,   std::memory_order_relaxed);
    gain_sum_.store(0.0, std::memory_order_relaxed);
    loss_sum_.store(0.0, std::memory_order_relaxed);
    high_gain_events_.store(0, std::memory_order_relaxed);
    high_loss_events_.store(0, std::memory_order_relaxed);
    total_.store(0,     std::memory_order_relaxed);
}

void SignalGainLossRatio::update_config(const Config& cfg) {
    std::lock_guard<std::mutex> lk(mutex_);
    cfg_ = cfg;
    int w = std::max(4, cfg_.window);
    delta_ring_.assign(static_cast<std::size_t>(w), 0.0);
    head_ = fill_ = 0;
    gain_sum_v_    = loss_sum_v_ = 0.0;
    prev_bias_     = 0.0;
    seeded_        = false;
    prev_high_gain_ = prev_high_loss_ = false;
    ratio_.store(1.0,   std::memory_order_relaxed);
    gain_sum_.store(0.0, std::memory_order_relaxed);
    loss_sum_.store(0.0, std::memory_order_relaxed);
    high_gain_events_.store(0, std::memory_order_relaxed);
    high_loss_events_.store(0, std::memory_order_relaxed);
    total_.store(0,     std::memory_order_relaxed);
}

} // namespace llmquant
