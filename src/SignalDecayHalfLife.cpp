#include "SignalDecayHalfLife.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <sstream>
#include <iomanip>

namespace llmquant {

static constexpr double kNaN = std::numeric_limits<double>::quiet_NaN();

SignalDecayHalfLife::SignalDecayHalfLife(Config cfg)
    : cfg_(std::move(cfg))
{
    int w = std::max(4, cfg_.window);
    cfg_.window = w;
    buf_.resize(static_cast<std::size_t>(w));
}

void SignalDecayHalfLife::recompute_locked() {
    int n = fill_;
    if (n < cfg_.min_samples) {
        half_life_.store(kNaN, std::memory_order_relaxed);
        decay_rate_.store(kNaN, std::memory_order_relaxed);
        r_squared_.store(kNaN, std::memory_order_relaxed);
        return;
    }

    // Collect valid samples from circular buffer
    int w = cfg_.window;
    double sum_t = 0.0, sum_y = 0.0, sum_tt = 0.0, sum_ty = 0.0, sum_yy = 0.0;
    int valid = 0;
    for (int i = 0; i < n; ++i) {
        int idx = (head_ - n + i + w * 2) % w;
        const auto& s = buf_[static_cast<std::size_t>(idx)];
        if (!std::isfinite(s.log_abs)) continue;
        sum_t  += s.t;
        sum_y  += s.log_abs;
        sum_tt += s.t * s.t;
        sum_ty += s.t * s.log_abs;
        sum_yy += s.log_abs * s.log_abs;
        ++valid;
    }

    if (valid < cfg_.min_samples) {
        half_life_.store(kNaN, std::memory_order_relaxed);
        decay_rate_.store(kNaN, std::memory_order_relaxed);
        r_squared_.store(kNaN, std::memory_order_relaxed);
        return;
    }

    double nv  = static_cast<double>(valid);
    double denom = nv * sum_tt - sum_t * sum_t;
    if (std::abs(denom) < 1e-12) {
        half_life_.store(kNaN, std::memory_order_relaxed);
        decay_rate_.store(kNaN, std::memory_order_relaxed);
        r_squared_.store(kNaN, std::memory_order_relaxed);
        return;
    }

    // OLS: log|x| = a + b*t, b = decay slope = -λ
    double b = (nv * sum_ty - sum_t * sum_y) / denom;
    // b is the slope; λ = -b
    double lambda = -b;

    double half_life_val = (lambda > 1e-12) ? std::log(2.0) / lambda : kNaN;
    double decay_rate_val = lambda;

    // R² = 1 - SS_res / SS_tot
    double a = (sum_y - b * sum_t) / nv;
    double ss_res = 0.0, ss_tot = 0.0;
    double y_mean = sum_y / nv;
    for (int i = 0; i < n; ++i) {
        int idx = (head_ - n + i + w * 2) % w;
        const auto& s = buf_[static_cast<std::size_t>(idx)];
        if (!std::isfinite(s.log_abs)) continue;
        double y_hat = a + b * s.t;
        double res   = s.log_abs - y_hat;
        double dev   = s.log_abs - y_mean;
        ss_res += res * res;
        ss_tot += dev * dev;
    }
    double r2 = (ss_tot > 1e-12) ? 1.0 - ss_res / ss_tot : kNaN;

    half_life_.store(half_life_val,  std::memory_order_relaxed);
    decay_rate_.store(decay_rate_val, std::memory_order_relaxed);
    r_squared_.store(r2,              std::memory_order_relaxed);
}

void SignalDecayHalfLife::record(double bias) {
    bool   fire_fast = false, fire_slow = false;
    double hl_val = kNaN;
    std::function<void(double)> fast_cb, slow_cb;

    {
        std::lock_guard<std::mutex> lk(mutex_);
        fast_cb = cfg_.on_fast_decay;
        slow_cb = cfg_.on_slow_decay;
        total_.fetch_add(1, std::memory_order_relaxed);

        double abs_bias = std::abs(bias);
        double log_abs  = (abs_bias >= cfg_.min_abs)
                          ? std::log(abs_bias) : kNaN;

        int w = cfg_.window;
        buf_[static_cast<std::size_t>(head_)] = {static_cast<double>(tick_), log_abs};
        head_ = (head_ + 1) % w;
        if (fill_ < w) ++fill_;
        ++tick_;

        recompute_locked();

        hl_val = half_life_.load(std::memory_order_relaxed);

        if (std::isfinite(hl_val)) {
            bool now_fast = (hl_val < cfg_.fast_threshold);
            bool now_slow = (hl_val > cfg_.slow_threshold);

            if (now_fast && !prev_fast_) {
                fire_fast = true;
                fast_events_.fetch_add(1, std::memory_order_relaxed);
            }
            if (now_slow && !prev_slow_) {
                fire_slow = true;
                slow_events_.fetch_add(1, std::memory_order_relaxed);
            }
            prev_fast_ = now_fast;
            prev_slow_ = now_slow;
        }
    }

    if (fire_fast && fast_cb) try { fast_cb(hl_val); } catch (...) {}
    if (fire_slow && slow_cb) try { slow_cb(hl_val); } catch (...) {}
}

std::string SignalDecayHalfLife::to_stats_json() const {
    double hl = half_life_.load(std::memory_order_relaxed);
    double dr = decay_rate_.load(std::memory_order_relaxed);
    double r2 = r_squared_.load(std::memory_order_relaxed);
    uint64_t t  = total_.load(std::memory_order_relaxed);
    uint64_t fe = fast_events_.load(std::memory_order_relaxed);
    uint64_t se = slow_events_.load(std::memory_order_relaxed);

    char buf[256];
    // Use null for NaN fields
    auto fmt = [](double v) -> std::string {
        if (!std::isfinite(v)) return "null";
        char tmp[32]; std::snprintf(tmp, sizeof(tmp), "%.6f", v); return tmp;
    };
    std::snprintf(buf, sizeof(buf),
        "{\"half_life\":%s,\"decay_rate\":%s,\"r_squared\":%s,"
        "\"fast_decay_events\":%llu,\"slow_decay_events\":%llu,\"total_records\":%llu}",
        fmt(hl).c_str(), fmt(dr).c_str(), fmt(r2).c_str(),
        static_cast<unsigned long long>(fe),
        static_cast<unsigned long long>(se),
        static_cast<unsigned long long>(t));
    return buf;
}

void SignalDecayHalfLife::reset() {
    std::lock_guard<std::mutex> lk(mutex_);
    std::fill(buf_.begin(), buf_.end(), Sample{0.0, kNaN});
    head_ = fill_ = 0;
    tick_ = 0;
    prev_fast_ = prev_slow_ = false;
    half_life_.store(kNaN, std::memory_order_relaxed);
    decay_rate_.store(kNaN, std::memory_order_relaxed);
    r_squared_.store(kNaN, std::memory_order_relaxed);
    total_.store(0, std::memory_order_relaxed);
    fast_events_.store(0, std::memory_order_relaxed);
    slow_events_.store(0, std::memory_order_relaxed);
}

void SignalDecayHalfLife::update_config(const Config& cfg) {
    std::lock_guard<std::mutex> lk(mutex_);
    cfg_ = cfg;
    int w = std::max(4, cfg_.window);
    cfg_.window = w;
    buf_.assign(static_cast<std::size_t>(w), Sample{0.0, kNaN});
    head_ = fill_ = 0;
    tick_ = 0;
    prev_fast_ = prev_slow_ = false;
    half_life_.store(kNaN, std::memory_order_relaxed);
    decay_rate_.store(kNaN, std::memory_order_relaxed);
    r_squared_.store(kNaN, std::memory_order_relaxed);
    total_.store(0, std::memory_order_relaxed);
    fast_events_.store(0, std::memory_order_relaxed);
    slow_events_.store(0, std::memory_order_relaxed);
}

} // namespace llmquant
