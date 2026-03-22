#include "ConfidenceDecayTracker.h"

#include <algorithm>
#include <cmath>
#include <numeric>
#include <sstream>

namespace llmquant {

ConfidenceDecayTracker::ConfidenceDecayTracker(Config config)
    : config_(std::move(config)) {
    ring_.resize(static_cast<size_t>(config_.window_size));
    fast_threshold_.store(config_.fast_decay_threshold_ms, std::memory_order_relaxed);
    epoch_ = Clock::now();
}

// ---------------------------------------------------------------------------
// Internal refit
// ---------------------------------------------------------------------------

void ConfidenceDecayTracker::refit_locked() {
    int n = ring_fill_;
    if (n < config_.min_samples) {
        half_life_ms_.store(0.0, std::memory_order_relaxed);
        lambda_.store(0.0, std::memory_order_relaxed);
        return;
    }

    // Collect (t, log_c) pairs from ring buffer in chronological order.
    // We use relative age (t - t_min) so the fit origin is the oldest point.
    double t_min = std::numeric_limits<double>::max();
    for (int i = 0; i < n; ++i) {
        int idx = (ring_head_ - n + i + config_.window_size) % config_.window_size;
        t_min = std::min(t_min, ring_[static_cast<size_t>(idx)].age_ms);
    }

    // Ordinary least squares: fit log_c = a - lambda * t.
    // λ = (n Σ(t·y) - Σt Σy) / (n Σt² - (Σt)²)
    double sum_t = 0.0, sum_y = 0.0, sum_tt = 0.0, sum_ty = 0.0;
    for (int i = 0; i < n; ++i) {
        int    idx = (ring_head_ - n + i + config_.window_size) % config_.window_size;
        double t   = ring_[static_cast<size_t>(idx)].age_ms - t_min;
        double y   = ring_[static_cast<size_t>(idx)].log_conf;
        sum_t  += t;
        sum_y  += y;
        sum_tt += t * t;
        sum_ty += t * y;
    }

    double denom = static_cast<double>(n) * sum_tt - sum_t * sum_t;
    double lam   = 0.0;
    if (std::abs(denom) > 1e-12) {
        lam = -(static_cast<double>(n) * sum_ty - sum_t * sum_y) / denom;
    }

    // Only accept positive decay (confidence falling over time).
    lam = std::max(0.0, lam);

    double hl = (lam > 1e-12) ? (std::log(2.0) / lam) : 0.0;

    lambda_.store(lam, std::memory_order_relaxed);
    half_life_ms_.store(hl, std::memory_order_relaxed);
}

// ---------------------------------------------------------------------------
// Public interface
// ---------------------------------------------------------------------------

void ConfidenceDecayTracker::record(double confidence) noexcept {
    if (confidence <= 0.0) confidence = 1e-9;
    if (confidence >  1.0) confidence = 1.0;

    total_.fetch_add(1, std::memory_order_relaxed);

    double old_hl, new_hl;
    Config cfg_copy;

    {
        std::lock_guard<std::mutex> lk(mutex_);
        cfg_copy = config_;

        double age_ms = std::chrono::duration<double, std::milli>(
            Clock::now() - epoch_).count();

        ring_[static_cast<size_t>(ring_head_)] = {age_ms, std::log(confidence)};
        ring_head_ = (ring_head_ + 1) % config_.window_size;
        if (ring_fill_ < config_.window_size) ++ring_fill_;

        old_hl = half_life_ms_.load(std::memory_order_relaxed);
        refit_locked();
        new_hl = half_life_ms_.load(std::memory_order_relaxed);
        prev_half_life_ = new_hl;
    }

    // Fire callback if half-life changed significantly.
    double ref = std::max(old_hl, new_hl);
    if (ref > 1e-12 &&
        std::abs(new_hl - old_hl) / ref > cfg_copy.change_fraction &&
        cfg_copy.on_decay_change) {
        cfg_copy.on_decay_change(new_hl, old_hl);
    }
}

bool ConfidenceDecayTracker::is_fast_decay() const noexcept {
    double hl = half_life_ms_.load(std::memory_order_relaxed);
    double th = fast_threshold_.load(std::memory_order_relaxed);
    return hl > 0.0 && hl < th;
}

void ConfidenceDecayTracker::update_config(Config config) {
    std::lock_guard<std::mutex> lk(mutex_);
    config_ = std::move(config);
    ring_.assign(static_cast<size_t>(config_.window_size), {});
    ring_head_  = 0;
    ring_fill_  = 0;
    epoch_      = Clock::now();
    fast_threshold_.store(config_.fast_decay_threshold_ms, std::memory_order_relaxed);
    half_life_ms_.store(0.0, std::memory_order_relaxed);
    lambda_.store(0.0, std::memory_order_relaxed);
    total_.store(0, std::memory_order_relaxed);
    prev_half_life_ = 0.0;
}

void ConfidenceDecayTracker::reset() {
    std::lock_guard<std::mutex> lk(mutex_);
    std::fill(ring_.begin(), ring_.end(), Sample{});
    ring_head_  = 0;
    ring_fill_  = 0;
    epoch_      = Clock::now();
    half_life_ms_.store(0.0, std::memory_order_relaxed);
    lambda_.store(0.0, std::memory_order_relaxed);
    total_.store(0, std::memory_order_relaxed);
    prev_half_life_ = 0.0;
}

std::string ConfidenceDecayTracker::to_stats_json() const {
    double hl  = half_life_ms_.load(std::memory_order_relaxed);
    double lam = lambda_.load(std::memory_order_relaxed);
    uint64_t t = total_.load(std::memory_order_relaxed);

    std::ostringstream o;
    o << std::fixed;
    o.precision(4);
    o << "{\"half_life_ms\":"   << hl
      << ",\"lambda\":"         << lam
      << ",\"is_fast_decay\":"  << (is_fast_decay() ? "true" : "false")
      << ",\"total_records\":"  << t << "}";
    return o.str();
}

} // namespace llmquant
