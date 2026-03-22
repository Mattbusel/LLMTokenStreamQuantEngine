#include "SignalConfidenceInterval.h"

#include <algorithm>
#include <cmath>
#include <numeric>
#include <sstream>
#include <iomanip>

namespace llmquant {

SignalConfidenceInterval::SignalConfidenceInterval(Config cfg)
    : cfg_(std::move(cfg))
{
    window_.resize(cfg_.window_size, 0.0);
}

void SignalConfidenceInterval::recompute_locked() {
    if (count_ < 2) {
        mean_.store(count_ == 1 ? window_[0] : 0.0, std::memory_order_relaxed);
        half_width_.store(0.0, std::memory_order_relaxed);
        return;
    }

    int n = static_cast<int>(count_);

    // Collect active window.
    std::vector<double> vals;
    vals.reserve(count_);
    std::size_t start = (head_ >= count_) ? (head_ - count_) : (cfg_.window_size - (count_ - head_));
    for (std::size_t i = 0; i < count_; ++i)
        vals.push_back(window_[(start + i) % cfg_.window_size]);

    double sum = std::accumulate(vals.begin(), vals.end(), 0.0);
    double mu  = sum / n;
    mean_.store(mu, std::memory_order_relaxed);

    // Jackknife: compute leave-one-out means (pseudo-values).
    // pseudo_i = n * mu - (n-1) * mu_{-i}
    //          = n * mu - (sum - vals[i])
    //          = vals[i]   (when simplified: pseudo_i = vals[i])
    // Jackknife SE = sqrt((n-1)/n * Σ(vals[i] - mu)^2 / (n-1))
    //              = sqrt(variance_of_vals / n) = sample_std / sqrt(n)
    // which is just the standard error of the mean.
    double ss = 0.0;
    for (double v : vals) { double d = v - mu; ss += d * d; }
    double sample_var = ss / (n - 1);
    double se = std::sqrt(sample_var / n);
    double hw = cfg_.z * se;
    half_width_.store(hw, std::memory_order_relaxed);
}

void SignalConfidenceInterval::record(double value) {
    bool fire_narrow = false, fire_wide = false;
    double cur_mean = 0.0, cur_hw = 0.0;
    std::function<void(double, double)> cb_narrow, cb_wide;

    {
        std::lock_guard<std::mutex> lk(mutex_);
        cb_narrow = cfg_.on_narrow_interval;
        cb_wide   = cfg_.on_wide_interval;

        window_[head_] = value;
        head_ = (head_ + 1) % cfg_.window_size;
        if (count_ < cfg_.window_size) ++count_;
        total_.fetch_add(1, std::memory_order_relaxed);

        recompute_locked();
        cur_mean = mean_.load(std::memory_order_relaxed);
        cur_hw   = half_width_.load(std::memory_order_relaxed);

        bool now_narrow = (cur_hw < cfg_.narrow_threshold);
        bool now_wide   = (cur_hw > cfg_.wide_threshold);

        if (now_narrow && !prev_narrow_) {
            prev_narrow_ = true;
            prev_wide_   = false;
            fire_narrow  = true;
        } else if (!now_narrow) {
            prev_narrow_ = false;
        }

        if (now_wide && !prev_wide_) {
            prev_wide_  = true;
            prev_narrow_ = false;
            fire_wide   = true;
        } else if (!now_wide) {
            prev_wide_ = false;
        }
    }

    if (fire_narrow && cb_narrow) cb_narrow(cur_mean, cur_hw);
    if (fire_wide   && cb_wide)   cb_wide(cur_mean, cur_hw);
}

bool SignalConfidenceInterval::is_narrow() const noexcept {
    return half_width_.load(std::memory_order_relaxed) < cfg_.narrow_threshold;
}

bool SignalConfidenceInterval::is_wide() const noexcept {
    return half_width_.load(std::memory_order_relaxed) > cfg_.wide_threshold;
}

std::size_t SignalConfidenceInterval::window_fill() const noexcept {
    std::lock_guard<std::mutex> lk(mutex_);
    return count_;
}

void SignalConfidenceInterval::reset() {
    std::lock_guard<std::mutex> lk(mutex_);
    std::fill(window_.begin(), window_.end(), 0.0);
    head_  = 0;
    count_ = 0;
    mean_.store(0.0, std::memory_order_relaxed);
    half_width_.store(0.0, std::memory_order_relaxed);
    prev_narrow_ = prev_wide_ = false;
}

void SignalConfidenceInterval::update_config(const Config& cfg) {
    std::lock_guard<std::mutex> lk(mutex_);
    cfg_ = cfg;
    window_.assign(cfg_.window_size, 0.0);
    head_  = 0;
    count_ = 0;
    mean_.store(0.0, std::memory_order_relaxed);
    half_width_.store(0.0, std::memory_order_relaxed);
}

std::string SignalConfidenceInterval::to_stats_json() const {
    double mu, hw;
    uint64_t tot;
    std::size_t wf;
    {
        std::lock_guard<std::mutex> lk(mutex_);
        mu  = mean_.load(std::memory_order_relaxed);
        hw  = half_width_.load(std::memory_order_relaxed);
        tot = total_.load(std::memory_order_relaxed);
        wf  = count_;
    }
    std::ostringstream ss;
    ss << std::fixed << std::setprecision(6);
    ss << "{"
       << "\"mean\":"          << mu       << ","
       << "\"half_width\":"    << hw       << ","
       << "\"lower\":"         << (mu - hw) << ","
       << "\"upper\":"         << (mu + hw) << ","
       << "\"is_narrow\":"     << (hw < cfg_.narrow_threshold ? "true" : "false") << ","
       << "\"is_wide\":"       << (hw > cfg_.wide_threshold   ? "true" : "false") << ","
       << "\"window_fill\":"   << wf       << ","
       << "\"total_records\":" << tot
       << "}";
    return ss.str();
}

} // namespace llmquant
