#include "SentimentTrajectoryAnalyzer.h"

#include <algorithm>
#include <cinttypes>
#include <cmath>
#include <cstdio>
#include <numeric>

namespace llmquant {

SentimentTrajectoryAnalyzer::SentimentTrajectoryAnalyzer(Config config)
    : config_(std::move(config)) {
    // Clamp window_size to a sensible minimum.
    if (config_.window_size < 2) config_.window_size = 2;
    if (config_.min_samples_for_trend < 2) config_.min_samples_for_trend = 2;
    if (config_.min_samples_for_trend > config_.window_size)
        config_.min_samples_for_trend = config_.window_size;
    window_.reserve(static_cast<size_t>(config_.window_size));
}

void SentimentTrajectoryAnalyzer::add_sample(double sentiment_score) {
    total_samples_.fetch_add(1, std::memory_order_relaxed);
    std::lock_guard<std::mutex> lk(window_mutex_);
    if (static_cast<int>(window_.size()) >= config_.window_size) {
        // Evict the oldest sample (front) in O(n); acceptable for typical
        // window sizes (10–100).  Could be replaced with a circular buffer
        // for very large windows.
        window_.erase(window_.begin());
    }
    window_.push_back(sentiment_score);
}

SentimentTrajectoryAnalyzer::RegressionResult
SentimentTrajectoryAnalyzer::compute_regression_locked() const {
    RegressionResult r;
    int n = static_cast<int>(window_.size());
    if (n < 2) return r;

    // Welford-style mean computation for numerical stability.
    double sum_x = 0.0, sum_y = 0.0;
    for (int i = 0; i < n; ++i) {
        sum_x += static_cast<double>(i);
        sum_y += window_[static_cast<size_t>(i)];
    }
    double mean_x = sum_x / static_cast<double>(n);
    double mean_y = sum_y / static_cast<double>(n);
    r.mean = mean_y;

    // Least-squares slope and intercept.
    double ss_xx = 0.0, ss_xy = 0.0, ss_yy = 0.0;
    for (int i = 0; i < n; ++i) {
        double dx = static_cast<double>(i) - mean_x;
        double dy = window_[static_cast<size_t>(i)] - mean_y;
        ss_xx += dx * dx;
        ss_xy += dx * dy;
        ss_yy += dy * dy;
    }

    if (ss_xx < 1e-12) return r;  // all x-values identical — degenerate

    r.slope = ss_xy / ss_xx;
    double intercept = mean_y - r.slope * mean_x;

    // R² = 1 - SS_res / SS_tot
    if (ss_yy < 1e-15) {
        // Perfectly flat signal: R²=1, residual_std=0, slope~0.
        r.r_squared = 1.0;
        r.residual_std = 0.0;
        return r;
    }
    double ss_res = 0.0;
    for (int i = 0; i < n; ++i) {
        double predicted = r.slope * static_cast<double>(i) + intercept;
        double residual  = window_[static_cast<size_t>(i)] - predicted;
        ss_res += residual * residual;
    }
    r.r_squared = 1.0 - ss_res / ss_yy;
    if (r.r_squared < 0.0) r.r_squared = 0.0;

    // Residual std dev (unbiased, n-2 denominator).
    r.residual_std = (n >= 3) ? std::sqrt(ss_res / static_cast<double>(n - 2)) : 0.0;
    return r;
}

SentimentTrend SentimentTrajectoryAnalyzer::get_trend() const {
    std::lock_guard<std::mutex> lk(window_mutex_);
    int n = static_cast<int>(window_.size());
    if (n < config_.min_samples_for_trend) return SentimentTrend::Undefined;

    auto reg = compute_regression_locked();

    // Volatility takes priority: if the fit is noisy, classify as Volatile.
    if (reg.residual_std > config_.volatility_threshold) return SentimentTrend::Volatile;

    if (reg.slope >  config_.trend_threshold) return SentimentTrend::Improving;
    if (reg.slope < -config_.trend_threshold) return SentimentTrend::Declining;
    return SentimentTrend::Stable;
}

double SentimentTrajectoryAnalyzer::get_slope() const {
    std::lock_guard<std::mutex> lk(window_mutex_);
    return compute_regression_locked().slope;
}

double SentimentTrajectoryAnalyzer::get_r_squared() const {
    std::lock_guard<std::mutex> lk(window_mutex_);
    return compute_regression_locked().r_squared;
}

double SentimentTrajectoryAnalyzer::get_residual_std() const {
    std::lock_guard<std::mutex> lk(window_mutex_);
    return compute_regression_locked().residual_std;
}

double SentimentTrajectoryAnalyzer::get_mean() const {
    std::lock_guard<std::mutex> lk(window_mutex_);
    return compute_regression_locked().mean;
}

int SentimentTrajectoryAnalyzer::get_sample_count() const {
    std::lock_guard<std::mutex> lk(window_mutex_);
    return static_cast<int>(window_.size());
}

const char* SentimentTrajectoryAnalyzer::trend_name(SentimentTrend trend) noexcept {
    switch (trend) {
        case SentimentTrend::Improving:  return "Improving";
        case SentimentTrend::Declining:  return "Declining";
        case SentimentTrend::Stable:     return "Stable";
        case SentimentTrend::Volatile:   return "Volatile";
        case SentimentTrend::Undefined:  return "Undefined";
        default:                         return "Unknown";
    }
}

std::string SentimentTrajectoryAnalyzer::to_stats_json() const {
    std::lock_guard<std::mutex> lk(window_mutex_);
    auto reg = compute_regression_locked();
    int n = static_cast<int>(window_.size());

    // Classify while holding the lock (avoids a second lock acquisition).
    SentimentTrend trend = SentimentTrend::Undefined;
    if (n >= config_.min_samples_for_trend) {
        if (reg.residual_std > config_.volatility_threshold)
            trend = SentimentTrend::Volatile;
        else if (reg.slope >  config_.trend_threshold)
            trend = SentimentTrend::Improving;
        else if (reg.slope < -config_.trend_threshold)
            trend = SentimentTrend::Declining;
        else
            trend = SentimentTrend::Stable;
    }

    char buf[512];
    std::snprintf(buf, sizeof(buf),
        "{\"trend\":\"%s\",\"slope\":%.6f,\"r_squared\":%.6f"
        ",\"residual_std\":%.6f,\"mean\":%.6f"
        ",\"sample_count\":%d,\"total_samples\":%" PRIu64 "}",
        trend_name(trend),
        reg.slope, reg.r_squared, reg.residual_std, reg.mean,
        n,
        total_samples_.load(std::memory_order_relaxed));
    return buf;
}

void SentimentTrajectoryAnalyzer::reset() {
    std::lock_guard<std::mutex> lk(window_mutex_);
    window_.clear();
    total_samples_.store(0, std::memory_order_relaxed);
}

} // namespace llmquant
