#include "BiasAutocorrelationFunction.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <sstream>

namespace llmquant {

static constexpr double kNaN = std::numeric_limits<double>::quiet_NaN();

BiasAutocorrelationFunction::BiasAutocorrelationFunction(Config cfg)
    : cfg_(std::move(cfg))
{
    int w = std::max(4, cfg_.window);
    int m = std::max(1, std::min(cfg_.max_lag, w / 2 - 1));
    cfg_.window  = w;
    cfg_.max_lag = m;
    if (cfg_.min_samples <= 0) cfg_.min_samples = w;
    buf_.resize(static_cast<std::size_t>(w), 0.0);
    acf_.assign(static_cast<std::size_t>(m), kNaN);
}

void BiasAutocorrelationFunction::recompute_locked() {
    int n = fill_;
    if (n < cfg_.min_samples) {
        std::fill(acf_.begin(), acf_.end(), kNaN);
        r1_.store(kNaN, std::memory_order_relaxed);
        dom_acf_.store(kNaN, std::memory_order_relaxed);
        dom_lag_.store(0,    std::memory_order_relaxed);
        return;
    }

    int w = cfg_.window;
    int m = cfg_.max_lag;

    // Build linear array from circular buffer (oldest first)
    std::vector<double> x(static_cast<std::size_t>(n));
    for (int i = 0; i < n; ++i) {
        int idx = (head_ - n + i + w * 2) % w;
        x[static_cast<std::size_t>(i)] = buf_[static_cast<std::size_t>(idx)];
    }

    // Sample mean
    double sum = 0.0;
    for (double v : x) sum += v;
    double mu = sum / static_cast<double>(n);

    // Variance (denominator n — biased)
    double var = 0.0;
    for (double v : x) { double d = v - mu; var += d * d; }
    // var = sum of squared deviations (not divided by n yet)

    if (var < 1e-12) {
        // Constant signal — all ACF = 0 by convention
        std::fill(acf_.begin(), acf_.end(), 0.0);
        r1_.store(0.0, std::memory_order_relaxed);
        dom_acf_.store(0.0, std::memory_order_relaxed);
        dom_lag_.store(1,   std::memory_order_relaxed);
        return;
    }

    // Compute r_k for k = 1..max_lag
    double best_abs = -1.0;
    int    best_lag = 1;
    for (int k = 1; k <= m; ++k) {
        double cov = 0.0;
        for (int t = k; t < n; ++t) {
            cov += (x[static_cast<std::size_t>(t)] - mu) *
                   (x[static_cast<std::size_t>(t - k)] - mu);
        }
        double rk = cov / var;   // biased: var is Σ(x-μ)², cov uses n-k terms
        acf_[static_cast<std::size_t>(k - 1)] = rk;
        if (std::abs(rk) > best_abs) {
            best_abs = std::abs(rk);
            best_lag = k;
        }
    }

    double r1_val = acf_[0];
    r1_.store(r1_val, std::memory_order_relaxed);
    dom_acf_.store(acf_[static_cast<std::size_t>(best_lag - 1)], std::memory_order_relaxed);
    dom_lag_.store(static_cast<uint32_t>(best_lag), std::memory_order_relaxed);
}

void BiasAutocorrelationFunction::record(double x) {
    bool   fire_periodic = false, fire_lost = false, fire_r1_flip = false;
    int    event_lag = 0;
    double event_acf = kNaN, event_r1 = kNaN;
    std::function<void(int, double)> periodic_cb;
    std::function<void()>            lost_cb;
    std::function<void(double)>      r1_flip_cb;

    {
        std::lock_guard<std::mutex> lk(mutex_);
        periodic_cb = cfg_.on_periodic_pattern;
        lost_cb     = cfg_.on_pattern_lost;
        r1_flip_cb  = cfg_.on_lag1_sign_change;

        total_.fetch_add(1, std::memory_order_relaxed);

        int w = cfg_.window;
        buf_[static_cast<std::size_t>(head_)] = x;
        head_ = (head_ + 1) % w;
        if (fill_ < w) ++fill_;

        recompute_locked();

        double r1_val  = r1_.load(std::memory_order_relaxed);
        double d_acf   = dom_acf_.load(std::memory_order_relaxed);
        int    d_lag   = static_cast<int>(dom_lag_.load(std::memory_order_relaxed));

        if (!std::isfinite(r1_val)) return;

        // r1 sign flip
        bool now_r1_positive = (r1_val >= 0.0);
        if (fill_ >= cfg_.min_samples && now_r1_positive != prev_r1_positive_) {
            fire_r1_flip = true;
            event_r1 = r1_val;
        }
        prev_r1_positive_ = now_r1_positive;

        // Periodic pattern detection (hysteresis: clear at 0.9× threshold)
        if (std::isfinite(d_acf)) {
            bool now_periodic = std::abs(d_acf) >= cfg_.periodic_threshold;
            bool was_periodic = prev_periodic_;
            if (now_periodic && !was_periodic) {
                fire_periodic = true;
                event_lag = d_lag;
                event_acf = d_acf;
                periodic_events_.fetch_add(1, std::memory_order_relaxed);
            } else if (!now_periodic && was_periodic &&
                       std::abs(d_acf) < cfg_.periodic_threshold * 0.9) {
                fire_lost = true;
            }
            prev_periodic_ = now_periodic;
            periodic_.store(now_periodic, std::memory_order_relaxed);
        }
    }

    if (fire_periodic && periodic_cb) try { periodic_cb(event_lag, event_acf); } catch (...) {}
    if (fire_lost     && lost_cb)     try { lost_cb(); }                          catch (...) {}
    if (fire_r1_flip  && r1_flip_cb)  try { r1_flip_cb(event_r1); }              catch (...) {}
}

std::vector<double> BiasAutocorrelationFunction::acf_snapshot() const {
    std::lock_guard<std::mutex> lk(mutex_);
    return acf_;
}

std::string BiasAutocorrelationFunction::to_stats_json() const {
    double r1v  = r1_.load(std::memory_order_relaxed);
    double dacf = dom_acf_.load(std::memory_order_relaxed);
    int    dlag = static_cast<int>(dom_lag_.load(std::memory_order_relaxed));
    bool   per  = periodic_.load(std::memory_order_relaxed);
    uint64_t tot = total_.load(std::memory_order_relaxed);
    uint64_t pe  = periodic_events_.load(std::memory_order_relaxed);

    auto fmt = [](double v) -> std::string {
        if (!std::isfinite(v)) return "null";
        char tmp[32]; std::snprintf(tmp, sizeof(tmp), "%.6f", v); return tmp;
    };
    char buf[256];
    std::snprintf(buf, sizeof(buf),
        "{\"r1\":%s,\"dominant_lag\":%d,\"dominant_acf\":%s,"
        "\"is_periodic\":%s,\"periodic_events\":%llu,\"total_records\":%llu}",
        fmt(r1v).c_str(), dlag, fmt(dacf).c_str(),
        per ? "true" : "false",
        static_cast<unsigned long long>(pe),
        static_cast<unsigned long long>(tot));
    return buf;
}

void BiasAutocorrelationFunction::reset() {
    std::lock_guard<std::mutex> lk(mutex_);
    std::fill(buf_.begin(), buf_.end(), 0.0);
    std::fill(acf_.begin(), acf_.end(), kNaN);
    head_ = fill_ = 0;
    prev_r1_positive_ = false;
    prev_periodic_    = false;
    prev_dom_lag_     = 0;
    r1_.store(kNaN, std::memory_order_relaxed);
    dom_acf_.store(kNaN, std::memory_order_relaxed);
    dom_lag_.store(0,    std::memory_order_relaxed);
    periodic_.store(false, std::memory_order_relaxed);
    total_.store(0, std::memory_order_relaxed);
    periodic_events_.store(0, std::memory_order_relaxed);
}

void BiasAutocorrelationFunction::update_config(const Config& cfg) {
    std::lock_guard<std::mutex> lk(mutex_);
    cfg_ = cfg;
    int w = std::max(4, cfg_.window);
    int m = std::max(1, std::min(cfg_.max_lag, w / 2 - 1));
    cfg_.window  = w;
    cfg_.max_lag = m;
    if (cfg_.min_samples <= 0) cfg_.min_samples = w;
    buf_.assign(static_cast<std::size_t>(w), 0.0);
    acf_.assign(static_cast<std::size_t>(m), kNaN);
    head_ = fill_ = 0;
    prev_r1_positive_ = false;
    prev_periodic_    = false;
    prev_dom_lag_     = 0;
    r1_.store(kNaN, std::memory_order_relaxed);
    dom_acf_.store(kNaN, std::memory_order_relaxed);
    dom_lag_.store(0,    std::memory_order_relaxed);
    periodic_.store(false, std::memory_order_relaxed);
    total_.store(0, std::memory_order_relaxed);
    periodic_events_.store(0, std::memory_order_relaxed);
}

} // namespace llmquant
