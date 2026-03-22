#include "AdaptiveCooldownController.h"

#include <algorithm>
#include <cmath>
#include <spdlog/spdlog.h>
#include <sstream>

namespace llmquant {

// ---------------------------------------------------------------------------
// Construction
// ---------------------------------------------------------------------------

AdaptiveCooldownController::AdaptiveCooldownController(Config config)
    : config_(config)
    , ema_p99_us_(config.base_cooldown_us > 0 ? config.target_p99_us : 0.0)
    , current_cooldown_us_(config.base_cooldown_us)
    , config_snapshot_base_(config.base_cooldown_us)
{}

// ---------------------------------------------------------------------------
// P99 update (hot-ish path: called once per stats sample interval)
// ---------------------------------------------------------------------------

void AdaptiveCooldownController::update_p99(double p99_us) {
    // Update EMA.
    const double alpha   = config_.ema_alpha;
    double old_ema       = ema_p99_us_.load(std::memory_order_relaxed);
    double new_ema       = alpha * p99_us + (1.0 - alpha) * old_ema;
    ema_p99_us_.store(new_ema, std::memory_order_relaxed);

    double base   = config_.base_cooldown_us;
    double target = config_.target_p99_us;
    double gain   = config_.pressure_gain;
    double lo     = config_.min_cooldown_us;
    double hi     = config_.max_cooldown_us;

    double excess = new_ema - target;
    double new_cd;

    if (excess > 0.0) {
        // Latency is above target → expand cooldown.
        new_cd = std::clamp(base + gain * excess, lo, hi);

        double old_cd = current_cooldown_us_.load(std::memory_order_relaxed);
        if (new_cd > old_cd + 0.5) {
            pressure_expansions_.fetch_add(1, std::memory_order_relaxed);
            spdlog::debug("[adaptive_cd] pressure: P99={:.1f}µs → cooldown={:.0f}µs",
                          new_ema, new_cd);
        }
        current_cooldown_us_.store(new_cd, std::memory_order_relaxed);

        // Reset hysteresis timer.
        std::lock_guard<std::mutex> lk(mutex_);
        below_target_timer_running_ = false;

    } else {
        // Latency is at or below target.
        std::lock_guard<std::mutex> lk(mutex_);
        auto now = std::chrono::steady_clock::now();

        if (!below_target_timer_running_) {
            below_target_since_          = now;
            below_target_timer_running_  = true;
        } else {
            auto elapsed_s = std::chrono::duration_cast<std::chrono::seconds>(
                now - below_target_since_).count();
            if (elapsed_s >= config_.recovery_window_s) {
                // Recovery: return to base cooldown.
                new_cd = std::clamp(base, lo, hi);
                double old_cd = current_cooldown_us_.load(std::memory_order_relaxed);
                if (new_cd < old_cd - 0.5) {
                    recoveries_.fetch_add(1, std::memory_order_relaxed);
                    spdlog::debug("[adaptive_cd] recovery: P99={:.1f}µs → cooldown={:.0f}µs",
                                  new_ema, new_cd);
                }
                current_cooldown_us_.store(new_cd, std::memory_order_relaxed);
                below_target_timer_running_ = false;
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Config
// ---------------------------------------------------------------------------

void AdaptiveCooldownController::update_config(const Config& config) {
    std::lock_guard<std::mutex> lk(mutex_);
    config_ = config;
    config_snapshot_base_.store(config.base_cooldown_us, std::memory_order_relaxed);
}

AdaptiveCooldownController::Config AdaptiveCooldownController::get_config() const {
    std::lock_guard<std::mutex> lk(mutex_);
    return config_;
}

// ---------------------------------------------------------------------------
// Stats
// ---------------------------------------------------------------------------

std::string AdaptiveCooldownController::to_stats_json() const {
    double cd   = current_cooldown_us_.load(std::memory_order_relaxed);
    double ema  = ema_p99_us_.load(std::memory_order_relaxed);
    double base = config_snapshot_base_.load(std::memory_order_relaxed);
    bool   under_pressure = cd > base;
    std::ostringstream o;
    o << "{\"cooldown_us\":"         << cd
      << ",\"ema_p99_us\":"          << ema
      << ",\"under_pressure\":"      << (under_pressure ? "true" : "false")
      << ",\"pressure_expansions\":" << pressure_expansions_.load(std::memory_order_relaxed)
      << ",\"recoveries\":"          << recoveries_.load(std::memory_order_relaxed)
      << ",\"base_cooldown_us\":"    << base
      << ",\"target_p99_us\":"       << config_.target_p99_us
      << "}";
    return o.str();
}

void AdaptiveCooldownController::reset() {
    std::lock_guard<std::mutex> lk(mutex_);
    ema_p99_us_.store(config_.target_p99_us, std::memory_order_relaxed);
    current_cooldown_us_.store(config_.base_cooldown_us, std::memory_order_relaxed);
    below_target_timer_running_ = false;
    pressure_expansions_.store(0, std::memory_order_relaxed);
    recoveries_.store(0, std::memory_order_relaxed);
}

} // namespace llmquant
