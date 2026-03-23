/// @file src/sentiment_trend.cpp
/// @brief Implementation of SentimentTrendTracker and RegimeSwitchDetector.
///
/// See include/sentiment_trend.hpp for full API documentation.

#include "sentiment_trend.hpp"

#include <algorithm>
#include <cassert>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <numeric>
#include <sstream>
#include <stdexcept>

namespace llmquant {

// ─── TrendDirection ───────────────────────────────────────────────────────────

const char* to_string(TrendDirection d) noexcept {
    switch (d) {
        case TrendDirection::Bullish:   return "Bullish";
        case TrendDirection::Bearish:   return "Bearish";
        case TrendDirection::Flat:      return "Flat";
        case TrendDirection::Reversing: return "Reversing";
    }
    return "Unknown";
}

// ─── TrendResult ──────────────────────────────────────────────────────────────

std::string TrendResult::to_string() const {
    std::ostringstream oss;
    oss << "TrendResult["
        << "dir=" << llmquant::to_string(trend_direction)
        << " slope=" << slope
        << " intercept=" << intercept
        << " R2=" << r_squared
        << " n=" << n_samples
        << " mean=" << mean_bias
        << "]";
    return oss.str();
}

// ─── SentimentTrendTracker ────────────────────────────────────────────────────

SentimentTrendTracker::SentimentTrendTracker(Config config) noexcept
    : config_(config) {}

void SentimentTrendTracker::push(const TradeSignal& signal) {
    push_raw(signal.delta_bias_shift,
             static_cast<uint64_t>(signal.timestamp_ns / 1'000'000ULL));
}

void SentimentTrendTracker::push_raw(double bias_value, uint64_t /*timestamp_ms*/) {
    std::lock_guard<std::mutex> lock(mutex_);
    window_.push_back(bias_value);
    if (window_.size() > config_.window_size) {
        window_.pop_front();
    }
    latest_ = fit_locked();
}

std::optional<TrendResult> SentimentTrendTracker::fit() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return fit_locked();
}

std::optional<TrendResult> SentimentTrendTracker::latest_result() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return latest_;
}

std::size_t SentimentTrendTracker::window_fill() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return window_.size();
}

void SentimentTrendTracker::reset() {
    std::lock_guard<std::mutex> lock(mutex_);
    window_.clear();
    latest_.reset();
}

std::optional<TrendResult> SentimentTrendTracker::fit_locked() const {
    const std::size_t n = window_.size();
    if (n < config_.min_samples) return std::nullopt;

    // Online OLS: y_i = slope × i + intercept
    // x_i = index in [0, n-1].
    const double n_d = static_cast<double>(n);

    double sum_x  = 0.0;
    double sum_y  = 0.0;
    double sum_xy = 0.0;
    double sum_xx = 0.0;

    for (std::size_t i = 0; i < n; ++i) {
        const double x = static_cast<double>(i);
        const double y = window_[i];
        sum_x  += x;
        sum_y  += y;
        sum_xy += x * y;
        sum_xx += x * x;
    }

    const double denom = n_d * sum_xx - sum_x * sum_x;
    if (std::fabs(denom) < 1e-12) {
        // Degenerate: all x values identical or no variance.
        TrendResult tr;
        tr.slope           = 0.0;
        tr.intercept       = sum_y / n_d;
        tr.r_squared       = 0.0;
        tr.trend_direction = TrendDirection::Flat;
        tr.n_samples       = n;
        tr.mean_bias       = sum_y / n_d;
        tr.timestamp       = std::chrono::high_resolution_clock::now();
        return tr;
    }

    const double slope     = (n_d * sum_xy - sum_x * sum_y) / denom;
    const double intercept = (sum_y - slope * sum_x) / n_d;
    const double mean_y    = sum_y / n_d;

    // R² = 1 − SS_res / SS_tot
    double ss_tot = 0.0;
    double ss_res = 0.0;
    for (std::size_t i = 0; i < n; ++i) {
        const double y_hat = slope * static_cast<double>(i) + intercept;
        const double res   = window_[i] - y_hat;
        const double dev   = window_[i] - mean_y;
        ss_res += res * res;
        ss_tot += dev * dev;
    }
    const double r_squared = (ss_tot < 1e-15) ? 0.0 : (1.0 - ss_res / ss_tot);

    TrendResult tr;
    tr.slope           = slope;
    tr.intercept       = intercept;
    tr.r_squared       = std::max(0.0, std::min(1.0, r_squared));
    tr.trend_direction = classify(slope, tr.r_squared, n);
    tr.n_samples       = n;
    tr.mean_bias       = mean_y;
    tr.timestamp       = std::chrono::high_resolution_clock::now();
    return tr;
}

TrendDirection SentimentTrendTracker::classify(
    double slope, double r_squared, std::size_t n
) const noexcept {
    // Too few samples to classify meaningfully.
    if (n < config_.min_samples) return TrendDirection::Flat;

    const double abs_slope = std::fabs(slope);

    // Flat: |slope| is too small to be directional.
    if (abs_slope <= config_.flat_slope_threshold) return TrendDirection::Flat;

    // Reversing: poor fit (low R²) with a non-trivial slope suggests
    // the underlying signal is oscillating rather than trending.
    if (r_squared < config_.reversing_r2_threshold) return TrendDirection::Reversing;

    return (slope > 0.0) ? TrendDirection::Bullish : TrendDirection::Bearish;
}

// ─── RegimeSwitch ─────────────────────────────────────────────────────────────

std::string RegimeSwitch::to_string() const {
    std::ostringstream oss;
    oss << "RegimeSwitch["
        << "from=" << llmquant::to_string(from)
        << " to="   << llmquant::to_string(to)
        << " confidence=" << confidence
        << " ts_ms=" << timestamp_ms
        << "]";
    return oss.str();
}

// ─── RegimeSwitchDetector ─────────────────────────────────────────────────────

RegimeSwitchDetector::RegimeSwitchDetector(Config config) noexcept
    : config_(config)
    , tracker_(config.tracker_cfg)
{}

void RegimeSwitchDetector::set_switch_callback(RegimeSwitchCallback cb) {
    std::lock_guard<std::mutex> lock(mutex_);
    switch_cb_ = std::move(cb);
}

void RegimeSwitchDetector::reset() {
    std::lock_guard<std::mutex> lock(mutex_);
    tracker_.reset();
    current_regime_        = TrendDirection::Flat;
    same_sign_count_       = 0;
    switching_             = false;
    uncertainty_remaining_ = 0;
    last_switch_.reset();
}

bool RegimeSwitchDetector::is_switching() const noexcept {
    std::lock_guard<std::mutex> lock(mutex_);
    return switching_;
}

std::optional<RegimeSwitch> RegimeSwitchDetector::last_switch() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return last_switch_;
}

std::optional<TrendResult> RegimeSwitchDetector::current_trend() const {
    return tracker_.latest_result();
}

void RegimeSwitchDetector::push(const TradeSignal& signal) {
    const uint64_t ts_ms = signal.timestamp_ns / 1'000'000ULL;
    push_locked(signal.delta_bias_shift, ts_ms);
}

void RegimeSwitchDetector::push_raw(double bias_value, uint64_t timestamp_ms) {
    push_locked(bias_value, timestamp_ms);
}

double RegimeSwitchDetector::compute_confidence(const TrendResult& tr) noexcept {
    // Confidence is a blend of R² and normalised |slope|.
    // Both components are in [0, 1]; we average them.
    const double slope_norm = std::fabs(tr.slope) / (std::fabs(tr.slope) + 1.0);
    return 0.5 * (tr.r_squared + slope_norm);
}

void RegimeSwitchDetector::push_locked(double bias_value, uint64_t timestamp_ms) {
    // Update tracker (acquires its own lock).
    tracker_.push_raw(bias_value, timestamp_ms);
    auto result = tracker_.latest_result();

    std::lock_guard<std::mutex> lock(mutex_);

    // Decrement uncertainty window.
    if (switching_) {
        if (uncertainty_remaining_ > 0) {
            --uncertainty_remaining_;
        } else {
            switching_ = false;
        }
    }

    if (!result.has_value()) return;

    const TrendDirection new_dir = result->trend_direction;

    // Check for a regime change: Bullish ↔ Bearish (ignore Flat / Reversing).
    const bool is_new_directional =
        (new_dir == TrendDirection::Bullish || new_dir == TrendDirection::Bearish);
    const bool is_old_directional =
        (current_regime_ == TrendDirection::Bullish ||
         current_regime_ == TrendDirection::Bearish);

    if (is_new_directional && is_old_directional &&
        new_dir != current_regime_) {
        ++same_sign_count_;
    } else {
        same_sign_count_ = 0;
    }

    // Confirm the switch after enough consecutive same-direction samples.
    if (same_sign_count_ >= static_cast<int>(config_.confirm_bars)) {
        const double confidence = compute_confidence(*result);
        if (confidence >= config_.min_confidence) {
            RegimeSwitch sw{
                .from         = current_regime_,
                .to           = new_dir,
                .confidence   = confidence,
                .timestamp_ms = timestamp_ms,
            };

            // Enter uncertainty window.
            switching_             = true;
            uncertainty_remaining_ = config_.uncertainty_bars;
            current_regime_        = new_dir;
            same_sign_count_       = 0;
            last_switch_           = sw;

            // Fire callback outside the lock scope would be safer but we
            // document that the callback MUST NOT call push() to avoid deadlock.
            if (switch_cb_) {
                switch_cb_(sw);
            }
            return;
        }
    }

    // If no switch detected but we have a new directional regime, update.
    if (is_new_directional && !is_old_directional) {
        current_regime_  = new_dir;
        same_sign_count_ = 1;
    }
}

}  // namespace llmquant
