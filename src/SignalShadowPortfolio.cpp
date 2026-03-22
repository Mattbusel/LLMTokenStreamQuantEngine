#include "SignalShadowPortfolio.h"

#include <algorithm>
#include <cmath>
#include <sstream>
#include <iomanip>

namespace llmquant {

SignalShadowPortfolio::SignalShadowPortfolio(Config cfg)
    : cfg_(std::move(cfg))
{}

void SignalShadowPortfolio::record_signal(double bias, double confidence) {
    std::lock_guard<std::mutex> lk(mutex_);
    double target = bias * confidence * cfg_.unit_size;
    target = std::clamp(target, -cfg_.max_position, cfg_.max_position);
    shadow_pos_val_ = target;
    shadow_pos_.store(shadow_pos_val_, std::memory_order_relaxed);
}

void SignalShadowPortfolio::record_live_signal(double actual_bias) {
    std::lock_guard<std::mutex> lk(mutex_);
    double target = actual_bias * cfg_.unit_size;
    target = std::clamp(target, -cfg_.max_position, cfg_.max_position);
    live_pos_val_ = target;
    live_pos_.store(live_pos_val_, std::memory_order_relaxed);
}

void SignalShadowPortfolio::record_price(double price) {
    std::function<void(double, double, double)> cb;
    bool   fire        = false;
    double drag_snap   = 0.0;
    double spnl_snap   = 0.0;
    double lpnl_snap   = 0.0;

    {
        std::lock_guard<std::mutex> lk(mutex_);
        cb = cfg_.on_drag_alert;
        price_updates_.fetch_add(1, std::memory_order_relaxed);

        if (!price_seeded_) {
            prev_price_   = price;
            price_seeded_ = true;
            return;
        }

        double dp = price - prev_price_;
        prev_price_ = price;

        shadow_pnl_val_ += shadow_pos_val_ * dp;
        live_pnl_val_   += live_pos_val_   * dp;

        shadow_pnl_.store(shadow_pnl_val_, std::memory_order_relaxed);
        live_pnl_.store(live_pnl_val_,     std::memory_order_relaxed);

        double drag = shadow_pnl_val_ - live_pnl_val_;

        if (drag_armed_ && std::abs(drag) > cfg_.drag_alert_threshold) {
            drag_armed_ = false;
            drag_alerts_.fetch_add(1, std::memory_order_relaxed);
            fire       = true;
            drag_snap  = drag;
            spnl_snap  = shadow_pnl_val_;
            lpnl_snap  = live_pnl_val_;
        } else if (!drag_armed_ && std::abs(drag) < cfg_.drag_alert_threshold * 0.5) {
            drag_armed_ = true;
        }
    }

    if (fire && cb) cb(drag_snap, spnl_snap, lpnl_snap);
}

std::string SignalShadowPortfolio::to_stats_json() const {
    double sp, lp, spos, lpos;
    uint64_t pu, da;
    {
        std::lock_guard<std::mutex> lk(mutex_);
        sp   = shadow_pnl_.load(std::memory_order_relaxed);
        lp   = live_pnl_.load(std::memory_order_relaxed);
        spos = shadow_pos_.load(std::memory_order_relaxed);
        lpos = live_pos_.load(std::memory_order_relaxed);
        pu   = price_updates_.load(std::memory_order_relaxed);
        da   = drag_alerts_.load(std::memory_order_relaxed);
    }
    std::ostringstream ss;
    ss << std::fixed << std::setprecision(6);
    ss << "{"
       << "\"shadow_pnl\":"       << sp           << ","
       << "\"live_pnl\":"         << lp           << ","
       << "\"constraint_drag\":"  << (sp - lp)    << ","
       << "\"shadow_position\":"  << spos         << ","
       << "\"live_position\":"    << lpos         << ","
       << "\"price_updates\":"    << pu           << ","
       << "\"drag_alert_count\":" << da
       << "}";
    return ss.str();
}

void SignalShadowPortfolio::reset() {
    std::lock_guard<std::mutex> lk(mutex_);
    shadow_pos_val_ = live_pos_val_ = 0.0;
    shadow_pnl_val_ = live_pnl_val_ = 0.0;
    prev_price_   = 0.0;
    price_seeded_ = false;
    drag_armed_   = true;
    shadow_pnl_.store(0.0, std::memory_order_relaxed);
    live_pnl_.store(0.0, std::memory_order_relaxed);
    shadow_pos_.store(0.0, std::memory_order_relaxed);
    live_pos_.store(0.0, std::memory_order_relaxed);
    price_updates_.store(0, std::memory_order_relaxed);
    drag_alerts_.store(0, std::memory_order_relaxed);
}

void SignalShadowPortfolio::update_config(const Config& cfg) {
    std::lock_guard<std::mutex> lk(mutex_);
    cfg_ = cfg;
    shadow_pos_val_ = live_pos_val_ = 0.0;
    shadow_pnl_val_ = live_pnl_val_ = 0.0;
    prev_price_   = 0.0;
    price_seeded_ = false;
    drag_armed_   = true;
}

} // namespace llmquant
