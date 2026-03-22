#include "SignalConvexityMeter.h"

#include <cmath>
#include <sstream>
#include <iomanip>

namespace llmquant {

SignalConvexityMeter::SignalConvexityMeter(Config cfg)
    : cfg_(std::move(cfg)) {}

void SignalConvexityMeter::record(double x) {
    std::function<void(double)> accel_cb, decel_cb;
    std::function<void()>       stable_cb;
    bool fire_accel = false, fire_decel = false, fire_stable = false;
    double d2_val = 0.0;

    {
        std::lock_guard<std::mutex> lk(mutex_);
        accel_cb  = cfg_.on_accelerating;
        decel_cb  = cfg_.on_decelerating;
        stable_cb = cfg_.on_stabilized;

        uint64_t n = total_.fetch_add(1, std::memory_order_relaxed) + 1;

        if (!seeded_) {
            prev_x_    = x;
            d1_v_      = 0.0;
            d1_prev_   = 0.0;
            d2_v_      = 0.0;
            seeded_    = true;
            d1_.store(0.0, std::memory_order_relaxed);
            d2_.store(0.0, std::memory_order_relaxed);
            return;
        }

        double raw_d1 = x - prev_x_;
        prev_x_ = x;

        if (!d1_seeded_) {
            d1_v_     = raw_d1;
            d1_prev_  = raw_d1;
            d1_seeded_ = true;
            d1_.store(d1_v_, std::memory_order_relaxed);
            return;
        }

        double alpha1 = cfg_.alpha_d1;
        double new_d1 = alpha1 * raw_d1 + (1.0 - alpha1) * d1_v_;

        double raw_d2 = new_d1 - d1_prev_;
        double alpha2 = cfg_.alpha_d2;
        d2_v_ = alpha2 * raw_d2 + (1.0 - alpha2) * d2_v_;

        d1_prev_ = new_d1;
        d1_v_    = new_d1;
        d2_val   = d2_v_;

        d1_.store(d1_v_, std::memory_order_relaxed);
        d2_.store(d2_v_, std::memory_order_relaxed);

        Regime new_regime = Regime::STABLE;
        if (n >= static_cast<uint64_t>(cfg_.min_samples)) {
            if (d2_v_ > cfg_.convex_threshold)
                new_regime = Regime::ACCELERATING;
            else if (d2_v_ < -cfg_.convex_threshold)
                new_regime = Regime::DECELERATING;
        }

        regime_.store(static_cast<int>(new_regime), std::memory_order_relaxed);

        if (new_regime != prev_regime_) {
            regime_changes_.fetch_add(1, std::memory_order_relaxed);
            if (new_regime == Regime::ACCELERATING) fire_accel = true;
            else if (new_regime == Regime::DECELERATING) fire_decel = true;
            else fire_stable = true;
        }
        prev_regime_ = new_regime;
    }

    if (fire_accel  && accel_cb)  try { accel_cb(d2_val);  } catch (...) {}
    if (fire_decel  && decel_cb)  try { decel_cb(d2_val);  } catch (...) {}
    if (fire_stable && stable_cb) try { stable_cb();        } catch (...) {}
}

std::string SignalConvexityMeter::to_stats_json() const {
    double d1 = d1_.load(std::memory_order_relaxed);
    double d2 = d2_.load(std::memory_order_relaxed);
    int    r  = regime_.load(std::memory_order_relaxed);
    uint64_t t  = total_.load(std::memory_order_relaxed);
    uint64_t rc = regime_changes_.load(std::memory_order_relaxed);

    const char* rname = (r == static_cast<int>(Regime::ACCELERATING)) ? "accelerating"
                      : (r == static_cast<int>(Regime::DECELERATING))  ? "decelerating"
                      : "stable";

    char buf[256];
    std::snprintf(buf, sizeof(buf),
        "{\"first_derivative\":%.6f,\"second_derivative\":%.6f,"
        "\"regime\":\"%s\",\"regime_changes\":%llu,\"total_records\":%llu}",
        d1, d2, rname,
        static_cast<unsigned long long>(rc),
        static_cast<unsigned long long>(t));
    return buf;
}

void SignalConvexityMeter::reset() {
    std::lock_guard<std::mutex> lk(mutex_);
    prev_x_    = d1_v_ = d1_prev_ = d2_v_ = 0.0;
    seeded_    = false;
    d1_seeded_ = false;
    prev_regime_ = Regime::STABLE;
    d1_.store(0.0,   std::memory_order_relaxed);
    d2_.store(0.0,   std::memory_order_relaxed);
    regime_.store(static_cast<int>(Regime::STABLE), std::memory_order_relaxed);
    total_.store(0,  std::memory_order_relaxed);
    regime_changes_.store(0, std::memory_order_relaxed);
}

void SignalConvexityMeter::update_config(const Config& cfg) {
    std::lock_guard<std::mutex> lk(mutex_);
    cfg_       = cfg;
    prev_x_    = d1_v_ = d1_prev_ = d2_v_ = 0.0;
    seeded_    = false;
    d1_seeded_ = false;
    prev_regime_ = Regime::STABLE;
    d1_.store(0.0,   std::memory_order_relaxed);
    d2_.store(0.0,   std::memory_order_relaxed);
    regime_.store(static_cast<int>(Regime::STABLE), std::memory_order_relaxed);
    total_.store(0,  std::memory_order_relaxed);
    regime_changes_.store(0, std::memory_order_relaxed);
}

} // namespace llmquant
