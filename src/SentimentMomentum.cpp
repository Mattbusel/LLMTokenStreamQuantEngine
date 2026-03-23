/**
 * @file src/SentimentMomentum.cpp
 * @brief Implementation of SentimentMomentum.
 *
 * Maintains two ring buffers:
 *   1. sentiment samples  → linear-regression slope = momentum
 *   2. momentum samples   → linear-regression slope = acceleration
 *
 * The acceleration buffer stores the momentum value computed on every
 * add_sample() call, giving a second-order estimate.
 */

#include "SentimentMomentum.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>
#include <sstream>

namespace llmquant {

// ---------------------------------------------------------------------------
// Construction
// ---------------------------------------------------------------------------

SentimentMomentum::SentimentMomentum(Config config)
    : config_(config)
    , ring_(config.window, 0.0)
    , momentum_ring_(config.accel_window, 0.0)
{
    // Clamp accel_window to the primary window.
    if (config_.accel_window > config_.window) {
        config_.accel_window = config_.window;
        momentum_ring_.assign(config_.accel_window, 0.0);
    }
}

// ---------------------------------------------------------------------------
// add_sample
// ---------------------------------------------------------------------------

void SentimentMomentum::add_sample(double score) {
    samples_seen_.fetch_add(1, std::memory_order_relaxed);

    std::lock_guard<std::mutex> lk(mutex_);

    // Write into sentiment ring buffer.
    ring_[head_] = score;
    head_        = (head_ + 1) % config_.window;
    if (count_ < config_.window) ++count_;

    recompute_locked();
}

// ---------------------------------------------------------------------------
// recompute_locked  (called with mutex_ held)
// ---------------------------------------------------------------------------

void SentimentMomentum::recompute_locked() {
    if (count_ < 2) {
        regime_ = Regime::Insufficient;
        return;
    }

    // Reconstruct ordered sentiment window for regression.
    std::vector<double> ordered(count_);
    for (std::size_t i = 0; i < count_; ++i) {
        std::size_t idx = (head_ + config_.window - count_ + i) % config_.window;
        ordered[i]      = ring_[idx];
    }

    const double new_momentum = linear_slope(ordered);
    momentum_ = new_momentum;

    // Store momentum sample into acceleration ring buffer.
    momentum_ring_[mom_head_] = new_momentum;
    mom_head_                 = (mom_head_ + 1) % config_.accel_window;
    if (mom_count_ < config_.accel_window) ++mom_count_;

    if (mom_count_ >= 2) {
        std::vector<double> mom_ordered(mom_count_);
        for (std::size_t i = 0; i < mom_count_; ++i) {
            std::size_t idx = (mom_head_ + config_.accel_window - mom_count_ + i)
                              % config_.accel_window;
            mom_ordered[i]  = momentum_ring_[idx];
        }
        acceleration_ = linear_slope(mom_ordered);
    } else {
        acceleration_ = 0.0;
    }

    // Classify regime.
    const double nt = config_.neutral_threshold;
    const double at = config_.accel_threshold;

    if (std::abs(momentum_) < nt) {
        regime_ = Regime::Neutral;
    } else if (momentum_ > 0.0) {
        // Bullish momentum.
        if (acceleration_ > at) {
            regime_ = Regime::AcceleratingBullish;
        } else if (acceleration_ < -at) {
            regime_ = Regime::DeceleratingBullish;
        } else {
            regime_ = Regime::AcceleratingBullish; // flat accel → treat as sustaining
        }
    } else {
        // Bearish momentum.
        if (acceleration_ < -at) {
            regime_ = Regime::AcceleratingBearish;
        } else if (acceleration_ > at) {
            regime_ = Regime::DeceleratingBearish;
        } else {
            regime_ = Regime::AcceleratingBearish;
        }
    }
}

// ---------------------------------------------------------------------------
// reset
// ---------------------------------------------------------------------------

void SentimentMomentum::reset() {
    std::lock_guard<std::mutex> lk(mutex_);
    std::fill(ring_.begin(), ring_.end(), 0.0);
    std::fill(momentum_ring_.begin(), momentum_ring_.end(), 0.0);
    head_         = 0;
    count_        = 0;
    mom_head_     = 0;
    mom_count_    = 0;
    momentum_     = 0.0;
    acceleration_ = 0.0;
    regime_       = Regime::Insufficient;
    samples_seen_.store(0, std::memory_order_relaxed);
}

// ---------------------------------------------------------------------------
// Query
// ---------------------------------------------------------------------------

SentimentMomentum::Snapshot SentimentMomentum::snapshot() const {
    std::lock_guard<std::mutex> lk(mutex_);
    Snapshot s;
    s.momentum         = momentum_;
    s.acceleration     = acceleration_;
    s.regime           = regime_;
    s.reversal_warning = (regime_ == Regime::DeceleratingBullish
                       || regime_ == Regime::DeceleratingBearish);
    s.samples_seen     = samples_seen_.load(std::memory_order_relaxed);
    return s;
}

double SentimentMomentum::momentum() const {
    std::lock_guard<std::mutex> lk(mutex_);
    return momentum_;
}

double SentimentMomentum::acceleration() const {
    std::lock_guard<std::mutex> lk(mutex_);
    return acceleration_;
}

SentimentMomentum::Regime SentimentMomentum::regime() const {
    std::lock_guard<std::mutex> lk(mutex_);
    return regime_;
}

bool SentimentMomentum::reversal_warning() const {
    std::lock_guard<std::mutex> lk(mutex_);
    return (regime_ == Regime::DeceleratingBullish
         || regime_ == Regime::DeceleratingBearish);
}

// ---------------------------------------------------------------------------
// static helpers
// ---------------------------------------------------------------------------

/* static */
const char* SentimentMomentum::regime_name(Regime r) noexcept {
    switch (r) {
        case Regime::Insufficient:        return "Insufficient";
        case Regime::AcceleratingBullish: return "AcceleratingBullish";
        case Regime::DeceleratingBullish: return "DeceleratingBullish";
        case Regime::AcceleratingBearish: return "AcceleratingBearish";
        case Regime::DeceleratingBearish: return "DeceleratingBearish";
        case Regime::Neutral:             return "Neutral";
    }
    return "Unknown";
}

/* static */
double SentimentMomentum::linear_slope(const std::vector<double>& data) {
    const std::size_t n = data.size();
    if (n < 2) return 0.0;

    // Simple OLS slope: sum((x_i - x_mean)(y_i - y_mean)) / sum((x_i - x_mean)^2)
    // x_i = index i, so x_mean = (n-1)/2.
    const double x_mean = static_cast<double>(n - 1) / 2.0;
    double y_mean       = 0.0;
    for (const double v : data) y_mean += v;
    y_mean /= static_cast<double>(n);

    double num = 0.0;
    double den = 0.0;
    for (std::size_t i = 0; i < n; ++i) {
        const double dx = static_cast<double>(i) - x_mean;
        num += dx * (data[i] - y_mean);
        den += dx * dx;
    }
    return (den > 1e-15) ? (num / den) : 0.0;
}

// ---------------------------------------------------------------------------
// JSON serialisation
// ---------------------------------------------------------------------------

std::string SentimentMomentum::to_json() const {
    auto s = snapshot();
    std::ostringstream os;
    os << "{"
       << "\"momentum\":"         << s.momentum         << ","
       << "\"acceleration\":"     << s.acceleration     << ","
       << "\"regime\":\""         << regime_name(s.regime) << "\","
       << "\"reversal_warning\":" << (s.reversal_warning ? "true" : "false") << ","
       << "\"samples_seen\":"     << s.samples_seen
       << "}";
    return os.str();
}

} // namespace llmquant
