#include "OptionsFlowSentimentBridge.h"

#include <cmath>
#include <sstream>
#include <iomanip>

namespace llmquant {

OptionsFlowSentimentBridge::OptionsFlowSentimentBridge(Config cfg)
    : cfg_(std::move(cfg))
{}

void OptionsFlowSentimentBridge::record_bias(double bias, double dt_s) {
    std::function<void(DivergenceKind, double, double, double)> cb;
    DivergenceKind kind   = DivergenceKind::None;
    double score_snap     = 0.0;
    double vel_snap       = 0.0;
    double skew_snap      = 0.0;
    bool   should_fire    = false;

    {
        std::lock_guard<std::mutex> lk(mutex_);
        cb = cfg_.on_divergence;

        if (!bias_seeded_) {
            prev_bias_   = bias;
            bias_seeded_ = true;
            vel_ema_val_ = 0.0;
        } else if (dt_s > 0.0) {
            double instant_vel = (bias - prev_bias_) / dt_s;
            prev_bias_ = bias;
            vel_ema_val_ += cfg_.velocity_alpha * (instant_vel - vel_ema_val_);
            ++warmup_count_;
        }

        vel_ema_.store(vel_ema_val_, std::memory_order_relaxed);

        if (warmup_count_ < cfg_.min_warmup) return;

        // Divergence: sentiment velocity moves opposite to what skew implies.
        // Positive skew_ema means put premium > call premium (bearish hedging).
        // score = vel * skew: bullish narrative (vel>0) + bearish hedge (skew>0)
        //   → score > 0 → SmartMoneyBear (market hedging against your bullish signal)
        //   → score < 0 → SmartMoneyBull (market bullish hedge despite bearish signal)
        double score = vel_ema_val_ * skew_ema_val_;
        div_score_.store(score, std::memory_order_relaxed);

        double threshold     = cfg_.div_threshold;
        double rearm_thresh  = threshold * (1.0 - cfg_.hysteresis);

        if (armed_) {
            if (score > threshold) {
                kind = DivergenceKind::SmartMoneyBear;
                armed_ = true;
                should_fire = true;
                div_count_.fetch_add(1, std::memory_order_relaxed);
            } else if (score < -threshold) {
                kind = DivergenceKind::SmartMoneyBull;
                armed_ = true;
                should_fire = true;
                div_count_.fetch_add(1, std::memory_order_relaxed);
            }
        } else {
            // Re-arm when score retreats toward zero.
            if (std::abs(score) < rearm_thresh) {
                armed_ = true;
                kind = DivergenceKind::None;
                should_fire = true;
            }
        }

        if (should_fire) {
            last_kind_.store(static_cast<int>(kind), std::memory_order_relaxed);
            score_snap = score;
            vel_snap   = vel_ema_val_;
            skew_snap  = skew_ema_val_;
        }
    }

    if (should_fire && cb) cb(kind, score_snap, vel_snap, skew_snap);
}

void OptionsFlowSentimentBridge::record_skew(double put_iv, double call_iv) {
    std::lock_guard<std::mutex> lk(mutex_);
    double skew = put_iv - call_iv;
    skew_ema_val_ += cfg_.skew_alpha * (skew - skew_ema_val_);
    skew_ema_.store(skew_ema_val_, std::memory_order_relaxed);
}

std::string OptionsFlowSentimentBridge::to_stats_json() const {
    double vel, sk, score;
    int kind;
    uint64_t cnt;
    {
        std::lock_guard<std::mutex> lk(mutex_);
        vel   = vel_ema_.load(std::memory_order_relaxed);
        sk    = skew_ema_.load(std::memory_order_relaxed);
        score = div_score_.load(std::memory_order_relaxed);
        kind  = last_kind_.load(std::memory_order_relaxed);
        cnt   = div_count_.load(std::memory_order_relaxed);
    }
    std::ostringstream ss;
    ss << std::fixed << std::setprecision(6);
    ss << "{"
       << "\"sentiment_velocity_ema\":" << vel  << ","
       << "\"skew_ema\":"               << sk   << ","
       << "\"divergence_score\":"       << score << ","
       << "\"last_divergence_kind\":"   << kind << ","
       << "\"divergence_count\":"       << cnt
       << "}";
    return ss.str();
}

void OptionsFlowSentimentBridge::reset() {
    std::lock_guard<std::mutex> lk(mutex_);
    prev_bias_   = 0.0;
    bias_seeded_ = false;
    warmup_count_ = 0;
    vel_ema_val_ = 0.0;
    skew_ema_val_ = 0.0;
    armed_ = true;
    vel_ema_.store(0.0, std::memory_order_relaxed);
    skew_ema_.store(0.0, std::memory_order_relaxed);
    div_score_.store(0.0, std::memory_order_relaxed);
    last_kind_.store(0, std::memory_order_relaxed);
    div_count_.store(0, std::memory_order_relaxed);
}

void OptionsFlowSentimentBridge::update_config(const Config& cfg) {
    std::lock_guard<std::mutex> lk(mutex_);
    cfg_ = cfg;
    prev_bias_   = 0.0;
    bias_seeded_ = false;
    warmup_count_ = 0;
    vel_ema_val_ = 0.0;
    skew_ema_val_ = 0.0;
    armed_ = true;
}

} // namespace llmquant
