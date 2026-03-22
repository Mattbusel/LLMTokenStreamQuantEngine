#include "LeadLagDetector.h"

#include <algorithm>
#include <cmath>
#include <numeric>
#include <sstream>

namespace llmquant {

LeadLagDetector::LeadLagDetector(Config config) : config_(config) {
    int sz = std::max(config_.window_size, 2 * config_.max_lag + 2);
    config_.window_size = sz;
    ring_x_.assign(sz, 0.0);
    ring_y_.assign(sz, 0.0);
}

void LeadLagDetector::push_x(double value) noexcept {
    total_x_.fetch_add(1, std::memory_order_relaxed);
    std::lock_guard<std::mutex> lk(mutex_);
    ring_x_[head_x_] = value;
    head_x_ = (head_x_ + 1) % static_cast<int>(ring_x_.size());
    if (fill_x_ < static_cast<int>(ring_x_.size())) ++fill_x_;
}

void LeadLagDetector::push_y(double value) noexcept {
    total_y_.fetch_add(1, std::memory_order_relaxed);
    std::lock_guard<std::mutex> lk(mutex_);
    ring_y_[head_y_] = value;
    head_y_ = (head_y_ + 1) % static_cast<int>(ring_y_.size());
    if (fill_y_ < static_cast<int>(ring_y_.size())) ++fill_y_;
}

void LeadLagDetector::push(double x, double y) noexcept {
    total_x_.fetch_add(1, std::memory_order_relaxed);
    total_y_.fetch_add(1, std::memory_order_relaxed);
    std::lock_guard<std::mutex> lk(mutex_);
    ring_x_[head_x_] = x;
    head_x_ = (head_x_ + 1) % static_cast<int>(ring_x_.size());
    if (fill_x_ < static_cast<int>(ring_x_.size())) ++fill_x_;
    ring_y_[head_y_] = y;
    head_y_ = (head_y_ + 1) % static_cast<int>(ring_y_.size());
    if (fill_y_ < static_cast<int>(ring_y_.size())) ++fill_y_;
}

std::vector<double> LeadLagDetector::snapshot(const std::vector<double>& ring,
                                               int head, int fill) {
    if (fill == 0) return {};
    std::vector<double> out(fill);
    int sz    = static_cast<int>(ring.size());
    int start = (fill == sz) ? head : 0;
    for (int i = 0; i < fill; ++i)
        out[i] = ring[(start + i) % sz];
    return out;
}

double LeadLagDetector::pearson_at_lag(const std::vector<double>& x,
                                       const std::vector<double>& y,
                                       int lag) {
    // Positive lag: x leads y.  We align x[0..n-1-lag] with y[lag..n-1].
    int n = static_cast<int>(std::min(x.size(), y.size()));
    int abs_lag = std::abs(lag);
    int pairs   = n - abs_lag;
    if (pairs < 2) return 0.0;

    const double* px = x.data();
    const double* py = y.data();
    if (lag > 0) {
        // x leads y: compare x[0..pairs-1] with y[lag..lag+pairs-1]
        py = y.data() + lag;
    } else if (lag < 0) {
        // y leads x: compare x[abs_lag..abs_lag+pairs-1] with y[0..pairs-1]
        px = x.data() + abs_lag;
    }

    // Compute Pearson r over 'pairs' elements.
    double sum_x = 0, sum_y = 0, sum_xy = 0, sum_x2 = 0, sum_y2 = 0;
    for (int i = 0; i < pairs; ++i) {
        sum_x  += px[i];
        sum_y  += py[i];
        sum_xy += px[i] * py[i];
        sum_x2 += px[i] * px[i];
        sum_y2 += py[i] * py[i];
    }
    double n_d = static_cast<double>(pairs);
    double denom = std::sqrt((sum_x2 - sum_x * sum_x / n_d) *
                             (sum_y2 - sum_y * sum_y / n_d));
    if (denom < 1e-12) return 0.0;
    return (sum_xy - sum_x * sum_y / n_d) / denom;
}

LeadLagDetector::Correlogram LeadLagDetector::compute() const {
    Correlogram result;

    std::vector<double> sx, sy;
    int max_lag;
    {
        std::lock_guard<std::mutex> lk(mutex_);
        sx      = snapshot(ring_x_, head_x_, fill_x_);
        sy      = snapshot(ring_y_, head_y_, fill_y_);
        max_lag = config_.max_lag;
    }

    int min_n = static_cast<int>(std::min(sx.size(), sy.size()));
    if (min_n < 2 * max_lag + 2) return result;

    result.lags.reserve(2 * max_lag + 1);
    double best_abs = -1.0;

    for (int lag = -max_lag; lag <= max_lag; ++lag) {
        double r = pearson_at_lag(sx, sy, lag);
        result.lags.push_back({lag, r});
        if (std::abs(r) > best_abs) {
            best_abs           = std::abs(r);
            result.optimal_lag = lag;
            result.max_corr    = r;
        }
    }

    result.x_leads = result.optimal_lag > 0;
    result.y_leads = result.optimal_lag < 0;
    return result;
}

std::string LeadLagDetector::Correlogram::describe() const {
    if (lags.empty()) return "insufficient data";
    std::ostringstream o;
    if (optimal_lag == 0) {
        o << "contemporaneous (r=" << std::showpos;
    } else if (x_leads) {
        o << "X leads Y by " << optimal_lag << " (r=";
        o << std::showpos;
    } else {
        o << "Y leads X by " << -optimal_lag << " (r=";
        o << std::showpos;
    }
    o.precision(2);
    o << std::fixed << max_corr << ")";
    return o.str();
}

std::string LeadLagDetector::to_stats_json() const {
    auto c = compute();
    std::ostringstream o;
    o << std::fixed;
    o.precision(4);
    o << "{\"optimal_lag\":"  << c.optimal_lag
      << ",\"max_corr\":"     << c.max_corr
      << ",\"x_leads\":"      << (c.x_leads ? "true" : "false")
      << ",\"y_leads\":"      << (c.y_leads ? "true" : "false")
      << ",\"total_x\":"      << total_x_.load(std::memory_order_relaxed)
      << ",\"total_y\":"      << total_y_.load(std::memory_order_relaxed)
      << "}";
    return o.str();
}

void LeadLagDetector::update_config(const Config& config) {
    std::lock_guard<std::mutex> lk(mutex_);
    config_ = config;
    int sz  = std::max(config_.window_size, 2 * config_.max_lag + 2);
    config_.window_size = sz;
    ring_x_.assign(sz, 0.0);
    ring_y_.assign(sz, 0.0);
    head_x_ = head_y_ = fill_x_ = fill_y_ = 0;
}

LeadLagDetector::Config LeadLagDetector::get_config() const {
    std::lock_guard<std::mutex> lk(mutex_);
    return config_;
}

void LeadLagDetector::reset() {
    std::lock_guard<std::mutex> lk(mutex_);
    std::fill(ring_x_.begin(), ring_x_.end(), 0.0);
    std::fill(ring_y_.begin(), ring_y_.end(), 0.0);
    head_x_ = head_y_ = fill_x_ = fill_y_ = 0;
    total_x_.store(0, std::memory_order_relaxed);
    total_y_.store(0, std::memory_order_relaxed);
}

} // namespace llmquant
