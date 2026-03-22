#include "DrawdownProtector.h"

#include <algorithm>
#include <cmath>
#include <spdlog/spdlog.h>
#include <sstream>

namespace llmquant {

// ---------------------------------------------------------------------------
// Construction
// ---------------------------------------------------------------------------

DrawdownProtector::DrawdownProtector(Config config)
    : config_(std::move(config)) {}

// ---------------------------------------------------------------------------
// scale_for_tier
// ---------------------------------------------------------------------------

double DrawdownProtector::scale_for_tier(int tier,
                                          const Config& cfg) noexcept {
    switch (tier) {
        case 0: return 1.0;
        case 1: return cfg.tier1.scale;
        case 2: return cfg.tier2.scale;
        case 3: return cfg.tier3.scale;
        default: return 0.0;
    }
}

// ---------------------------------------------------------------------------
// compute_tier
// ---------------------------------------------------------------------------

int DrawdownProtector::compute_tier(double drawdown_pct,
                                     const Config& cfg) noexcept {
    if (drawdown_pct >= cfg.tier3.drawdown_pct) return 3;
    if (drawdown_pct >= cfg.tier2.drawdown_pct) return 2;
    if (drawdown_pct >= cfg.tier1.drawdown_pct) return 1;
    return 0;
}

// ---------------------------------------------------------------------------
// record_pnl
// ---------------------------------------------------------------------------

void DrawdownProtector::record_pnl(double pnl) {
    std::function<void(int, double, double)> cb;
    int old_tier = current_tier_.load(std::memory_order_relaxed);
    int new_tier = old_tier;
    double drawdown = 0.0;
    {
        std::lock_guard<std::mutex> lk(mutex_);
        cumulative_pnl_ += pnl;
        if (cumulative_pnl_ > high_water_mark_)
            high_water_mark_ = cumulative_pnl_;

        // Drawdown = (HWM - current) / HWM if HWM > 0.
        if (high_water_mark_ > 1e-12)
            drawdown = (high_water_mark_ - cumulative_pnl_) / high_water_mark_;
        else
            drawdown = 0.0;
        drawdown = std::max(0.0, drawdown);

        new_tier = compute_tier(drawdown, config_);

        if (new_tier != old_tier) {
            current_tier_.store(new_tier, std::memory_order_relaxed);
            tier_transitions_.fetch_add(1, std::memory_order_relaxed);
            cb = config_.on_tier_change;
            spdlog::warn("[drawdown] tier {} → {}  drawdown={:.1f}%  scale={:.2f}",
                         old_tier, new_tier, drawdown * 100.0,
                         scale_for_tier(new_tier, config_));
        }
    }
    if (cb) cb(new_tier, scale_for_tier(new_tier, config_), drawdown);
}

// ---------------------------------------------------------------------------
// reset
// ---------------------------------------------------------------------------

void DrawdownProtector::reset() {
    std::lock_guard<std::mutex> lk(mutex_);
    cumulative_pnl_  = 0.0;
    high_water_mark_ = 0.0;
    int old_tier = current_tier_.load(std::memory_order_relaxed);
    current_tier_.store(0, std::memory_order_relaxed);
    if (old_tier != 0)
        tier_transitions_.fetch_add(1, std::memory_order_relaxed);
}

// ---------------------------------------------------------------------------
// current_scale
// ---------------------------------------------------------------------------

DrawdownProtector::ScalePair DrawdownProtector::current_scale() const noexcept {
    int tier = current_tier_.load(std::memory_order_relaxed);
    double s = 0.0;
    {
        std::lock_guard<std::mutex> lk(mutex_);
        s = scale_for_tier(tier, config_);
    }
    return {s, s};
}

// ---------------------------------------------------------------------------
// current_drawdown_pct
// ---------------------------------------------------------------------------

double DrawdownProtector::current_drawdown_pct() const noexcept {
    std::lock_guard<std::mutex> lk(mutex_);
    if (high_water_mark_ <= 1e-12) return 0.0;
    double dd = (high_water_mark_ - cumulative_pnl_) / high_water_mark_;
    return std::max(0.0, dd);
}

// ---------------------------------------------------------------------------
// cumulative_pnl / high_water_mark
// ---------------------------------------------------------------------------

double DrawdownProtector::cumulative_pnl() const noexcept {
    std::lock_guard<std::mutex> lk(mutex_);
    return cumulative_pnl_;
}

double DrawdownProtector::high_water_mark() const noexcept {
    std::lock_guard<std::mutex> lk(mutex_);
    return high_water_mark_;
}

// ---------------------------------------------------------------------------
// update_config
// ---------------------------------------------------------------------------

void DrawdownProtector::update_config(const Config& config) {
    std::lock_guard<std::mutex> lk(mutex_);
    config_ = config;
}

// ---------------------------------------------------------------------------
// to_stats_json
// ---------------------------------------------------------------------------

std::string DrawdownProtector::to_stats_json() const {
    int tier = current_tier_.load(std::memory_order_relaxed);
    double pnl = 0.0, hwm = 0.0, dd = 0.0;
    {
        std::lock_guard<std::mutex> lk(mutex_);
        pnl = cumulative_pnl_;
        hwm = high_water_mark_;
        dd  = (hwm > 1e-12) ? std::max(0.0, (hwm - pnl) / hwm) : 0.0;
    }
    std::ostringstream s;
    s << "{\"current_tier\":"      << tier
      << ",\"current_drawdown\":"  << dd
      << ",\"cumulative_pnl\":"    << pnl
      << ",\"high_water_mark\":"   << hwm
      << ",\"tier_transitions\":"  << tier_transitions()
      << "}";
    return s.str();
}

} // namespace llmquant
