#include "NarrativeMomentumClock.h"

#include <sstream>

namespace llmquant {

// ---------------------------------------------------------------------------
// Construction
// ---------------------------------------------------------------------------

NarrativeMomentumClock::NarrativeMomentumClock() = default;

NarrativeMomentumClock::NarrativeMomentumClock(const Config& cfg) : cfg_(cfg) {}

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

static const char* quadrant_name(NarrativeMomentumClock::Quadrant q) noexcept {
    switch (q) {
        case NarrativeMomentumClock::Quadrant::Rising:     return "Rising";
        case NarrativeMomentumClock::Quadrant::Fading:     return "Fading";
        case NarrativeMomentumClock::Quadrant::Falling:    return "Falling";
        case NarrativeMomentumClock::Quadrant::Recovering: return "Recovering";
    }
    return "Unknown";
}

NarrativeMomentumClock::Quadrant
NarrativeMomentumClock::compute_quadrant(double b, double v) noexcept {
    if (b >= 0.0 && v >= 0.0) return Quadrant::Rising;
    if (b >= 0.0 && v <  0.0) return Quadrant::Fading;
    if (b <  0.0 && v <  0.0) return Quadrant::Falling;
    return Quadrant::Recovering;
}

// ---------------------------------------------------------------------------
// Core logic
// ---------------------------------------------------------------------------

void NarrativeMomentumClock::record(double bias) {
    std::function<void(Quadrant, Quadrant, double, double)> cb;
    Quadrant old_q{}, new_q{};
    double   new_bias_ema{}, new_vel_ema{};
    bool     fired = false;

    {
        std::lock_guard<std::mutex> lk(mutex_);

        if (!seeded_) {
            bias_ema_val_     = bias;
            velocity_ema_val_ = 0.0;
            seeded_           = true;
        } else {
            double old_bias  = bias_ema_val_;
            bias_ema_val_    += cfg_.bias_alpha     * (bias - bias_ema_val_);
            double raw_vel    = bias_ema_val_ - old_bias;
            velocity_ema_val_ += cfg_.velocity_alpha * (raw_vel - velocity_ema_val_);
        }

        new_bias_ema = bias_ema_val_;
        new_vel_ema  = velocity_ema_val_;

        old_q = static_cast<Quadrant>(quadrant_.load(std::memory_order_relaxed));
        new_q = compute_quadrant(new_bias_ema, new_vel_ema);

        if (new_q != old_q && total_.load(std::memory_order_relaxed) > 0) {
            transitions_.fetch_add(1, std::memory_order_relaxed);
            quadrant_.store(static_cast<int>(new_q), std::memory_order_relaxed);
            cb    = cfg_.on_quadrant_change;
            fired = true;
        } else if (new_q != old_q) {
            // First placement — update quadrant without counting as transition.
            quadrant_.store(static_cast<int>(new_q), std::memory_order_relaxed);
        }
    }

    bias_out_.store(new_bias_ema,  std::memory_order_relaxed);
    velocity_out_.store(new_vel_ema, std::memory_order_relaxed);
    total_.fetch_add(1, std::memory_order_relaxed);

    if (fired && cb) cb(old_q, new_q, new_bias_ema, new_vel_ema);
}

// ---------------------------------------------------------------------------
// Accessors
// ---------------------------------------------------------------------------

NarrativeMomentumClock::Quadrant
NarrativeMomentumClock::quadrant() const noexcept {
    return static_cast<Quadrant>(quadrant_.load(std::memory_order_relaxed));
}

double NarrativeMomentumClock::bias_ema() const noexcept {
    return bias_out_.load(std::memory_order_relaxed);
}

double NarrativeMomentumClock::velocity_ema() const noexcept {
    return velocity_out_.load(std::memory_order_relaxed);
}

uint64_t NarrativeMomentumClock::total_records() const noexcept {
    return total_.load(std::memory_order_relaxed);
}

uint64_t NarrativeMomentumClock::quadrant_transitions() const noexcept {
    return transitions_.load(std::memory_order_relaxed);
}

bool NarrativeMomentumClock::is_rising() const noexcept {
    return quadrant() == Quadrant::Rising;
}

bool NarrativeMomentumClock::is_falling() const noexcept {
    return quadrant() == Quadrant::Falling;
}

// ---------------------------------------------------------------------------
// Mutation
// ---------------------------------------------------------------------------

void NarrativeMomentumClock::reset() {
    std::lock_guard<std::mutex> lk(mutex_);
    bias_ema_val_     = 0.0;
    velocity_ema_val_ = 0.0;
    seeded_           = false;
    quadrant_.store(0,   std::memory_order_relaxed);
    bias_out_.store(0.0, std::memory_order_relaxed);
    velocity_out_.store(0.0, std::memory_order_relaxed);
    total_.store(0,      std::memory_order_relaxed);
    transitions_.store(0, std::memory_order_relaxed);
}

void NarrativeMomentumClock::update_config(const Config& cfg) {
    std::lock_guard<std::mutex> lk(mutex_);
    cfg_              = cfg;
    bias_ema_val_     = 0.0;
    velocity_ema_val_ = 0.0;
    seeded_           = false;
    quadrant_.store(0,   std::memory_order_relaxed);
    bias_out_.store(0.0, std::memory_order_relaxed);
    velocity_out_.store(0.0, std::memory_order_relaxed);
    total_.store(0,      std::memory_order_relaxed);
    transitions_.store(0, std::memory_order_relaxed);
}

// ---------------------------------------------------------------------------
// Diagnostics
// ---------------------------------------------------------------------------

std::string NarrativeMomentumClock::to_stats_json() const {
    auto q    = quadrant();
    double b  = bias_ema();
    double v  = velocity_ema();
    auto   tot = total_records();
    auto   tr  = quadrant_transitions();

    std::ostringstream os;
    os << "{"
       << "\"quadrant\":\"" << quadrant_name(q) << "\","
       << "\"bias_ema\":"      << b   << ","
       << "\"velocity_ema\":"  << v   << ","
       << "\"transitions\":"   << tr  << ","
       << "\"total_records\":" << tot
       << "}";
    return os.str();
}

} // namespace llmquant
