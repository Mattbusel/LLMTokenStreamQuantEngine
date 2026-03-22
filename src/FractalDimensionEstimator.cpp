#include "FractalDimensionEstimator.h"

#include <algorithm>
#include <cmath>
#include <numeric>
#include <sstream>

namespace llmquant {

FractalDimensionEstimator::FractalDimensionEstimator(Config config)
    : config_(std::move(config)) {}

// ---------------------------------------------------------------------------
// Internal: compute R/S over window[begin..end)
// ---------------------------------------------------------------------------
/* static */
double FractalDimensionEstimator::compute_rs(
        const std::deque<double>& w, int begin, int end) noexcept {
    int n = end - begin;
    if (n < 2) return 0.0;

    // Mean.
    double sum = 0.0;
    for (int i = begin; i < end; ++i) sum += w[static_cast<std::size_t>(i)];
    double mean = sum / n;

    // Cumulative deviations and standard deviation.
    double cum = 0.0, r_max = 0.0, r_min = 0.0, sq = 0.0;
    for (int i = begin; i < end; ++i) {
        double dev = w[static_cast<std::size_t>(i)] - mean;
        cum += dev;
        sq  += dev * dev;
        if (cum > r_max) r_max = cum;
        if (cum < r_min) r_min = cum;
    }

    double r = r_max - r_min;
    double s = std::sqrt(sq / n);

    if (s < 1e-14) return 0.0;
    return r / s;
}

// ---------------------------------------------------------------------------
// record
// ---------------------------------------------------------------------------
void FractalDimensionEstimator::record(double value) {
    total_records_.fetch_add(1, std::memory_order_relaxed);

    double new_hurst       = 0.5;
    double new_rs          = 0.0;
    bool   fire_cb         = false;
    double prev_hurst_cb   = 0.5;
    RegimeCallback cb;

    {
        std::lock_guard<std::mutex> lk(mutex_);

        window_.push_back(value);
        if (static_cast<int>(window_.size()) > config_.window_size)
            window_.pop_front();

        int n = static_cast<int>(window_.size());

        if (n >= config_.min_samples) {
            // Full-window R/S.
            double rs_n    = compute_rs(window_, 0, n);
            // Half-window R/S (use the most recent n/2 observations).
            int    half    = n / 2;
            double rs_half = compute_rs(window_, n - half, n);
            new_rs = rs_n;

            if (rs_half > 1e-14 && rs_n > 1e-14 && half >= 2) {
                // Two-point log–log regression.
                new_hurst = std::log(rs_n / rs_half) / std::log(2.0);
                // Clamp to (0, 1) — extreme values can occur in small windows.
                new_hurst = std::max(0.01, std::min(0.99, new_hurst));
            } else {
                new_hurst = 0.5;
            }

            // Detect regime change across the 0.5 boundary.
            bool prev_trend = (last_hurst_ > 0.5);
            bool new_trend  = (new_hurst  > 0.5);
            if (prev_trend != new_trend && config_.on_regime_change) {
                prev_hurst_cb = last_hurst_;
                regime_changes_.fetch_add(1, std::memory_order_relaxed);
                fire_cb = true;
                cb      = config_.on_regime_change;
            }

            last_hurst_ = new_hurst;
            rs_full_    = new_rs;
        }
    }

    // Fire callback outside the lock.
    if (fire_cb) cb(prev_hurst_cb, new_hurst);
}

void FractalDimensionEstimator::reset() {
    std::lock_guard<std::mutex> lk(mutex_);
    window_.clear();
    last_hurst_ = 0.5;
    rs_full_    = 0.0;
}

double FractalDimensionEstimator::hurst() const noexcept {
    std::lock_guard<std::mutex> lk(mutex_);
    return last_hurst_;
}

double FractalDimensionEstimator::rs_full() const noexcept {
    std::lock_guard<std::mutex> lk(mutex_);
    return rs_full_;
}

int FractalDimensionEstimator::sample_count() const noexcept {
    std::lock_guard<std::mutex> lk(mutex_);
    return static_cast<int>(window_.size());
}

void FractalDimensionEstimator::update_config(const Config& config) {
    std::lock_guard<std::mutex> lk(mutex_);
    config_     = config;
    window_.clear();
    last_hurst_ = 0.5;
    rs_full_    = 0.0;
}

std::string FractalDimensionEstimator::to_stats_json() const {
    std::lock_guard<std::mutex> lk(mutex_);
    std::ostringstream os;
    os << std::fixed;
    os << "{"
       << "\"hurst\":"          << last_hurst_    << ","
       << "\"rs_full\":"        << rs_full_        << ","
       << "\"sample_count\":"   << window_.size() << ","
       << "\"window_size\":"    << config_.window_size << ","
       << "\"total_records\":"  << total_records_.load(std::memory_order_relaxed) << ","
       << "\"regime_changes\":" << regime_changes_.load(std::memory_order_relaxed) << ","
       << "\"is_trending\":"    << (last_hurst_ > 0.5 ? "true" : "false")
       << "}";
    return os.str();
}

} // namespace llmquant
