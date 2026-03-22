#include "CVaRCalculator.h"

#include <algorithm>
#include <numeric>
#include <sstream>
#include <vector>

namespace llmquant {

CVaRCalculator::CVaRCalculator(Config config)
    : config_(std::move(config)) {}

// ---------------------------------------------------------------------------
// recompute — must be called under lock
// ---------------------------------------------------------------------------
void CVaRCalculator::recompute() {
    int n = static_cast<int>(window_.size());
    if (n < config_.min_samples) {
        cvar_ = 0.0;
        var_  = 0.0;
        return;
    }

    // Sort a copy ascending (worst losses first at the front).
    std::vector<double> sorted(window_.begin(), window_.end());
    std::sort(sorted.begin(), sorted.end());

    // VaR: (1 - alpha) quantile — the boundary loss.
    int var_idx = static_cast<int>((1.0 - config_.alpha) * n);
    if (var_idx >= n) var_idx = n - 1;
    var_ = sorted[static_cast<std::size_t>(var_idx)];

    // CVaR: mean of observations ≤ VaR.
    double sum  = 0.0;
    int    cnt  = 0;
    for (int i = 0; i <= var_idx; ++i) {
        sum += sorted[static_cast<std::size_t>(i)];
        ++cnt;
    }
    cvar_ = (cnt > 0) ? sum / cnt : 0.0;
}

// ---------------------------------------------------------------------------
// record
// ---------------------------------------------------------------------------
void CVaRCalculator::record(double pnl) {
    total_records_.fetch_add(1, std::memory_order_relaxed);

    bool fire_cb    = false;
    AlertCallback cb;

    {
        std::lock_guard<std::mutex> lk(mutex_);

        window_.push_back(pnl);
        if (static_cast<int>(window_.size()) > config_.window_size)
            window_.pop_front();

        recompute();

        if (config_.breach_threshold != 0.0 &&
            cvar_ < config_.breach_threshold &&
            static_cast<int>(window_.size()) >= config_.min_samples) {
            breach_events_.fetch_add(1, std::memory_order_relaxed);
            fire_cb = true;
            cb      = config_.on_breach;
        }
    }

    if (fire_cb && cb) cb(cvar_, var_, config_.alpha);
}

void CVaRCalculator::reset() {
    std::lock_guard<std::mutex> lk(mutex_);
    window_.clear();
    cvar_ = 0.0;
    var_  = 0.0;
}

double CVaRCalculator::cvar() const noexcept {
    std::lock_guard<std::mutex> lk(mutex_);
    return cvar_;
}

double CVaRCalculator::var() const noexcept {
    std::lock_guard<std::mutex> lk(mutex_);
    return var_;
}

int CVaRCalculator::sample_count() const noexcept {
    std::lock_guard<std::mutex> lk(mutex_);
    return static_cast<int>(window_.size());
}

bool CVaRCalculator::is_in_breach() const noexcept {
    std::lock_guard<std::mutex> lk(mutex_);
    return (static_cast<int>(window_.size()) >= config_.min_samples)
        && (config_.breach_threshold != 0.0)
        && (cvar_ < config_.breach_threshold);
}

void CVaRCalculator::update_config(const Config& config) {
    std::lock_guard<std::mutex> lk(mutex_);
    config_ = config;
    window_.clear();
    cvar_ = 0.0;
    var_  = 0.0;
}

std::string CVaRCalculator::to_stats_json() const {
    std::lock_guard<std::mutex> lk(mutex_);
    std::ostringstream os;
    os << std::fixed;
    os << "{"
       << "\"cvar\":"           << cvar_ << ","
       << "\"var\":"            << var_  << ","
       << "\"alpha\":"          << config_.alpha << ","
       << "\"sample_count\":"   << window_.size() << ","
       << "\"window_size\":"    << config_.window_size << ","
       << "\"total_records\":"  << total_records_.load(std::memory_order_relaxed) << ","
       << "\"breach_events\":"  << breach_events_.load(std::memory_order_relaxed) << ","
       << "\"is_in_breach\":"   << ((static_cast<int>(window_.size()) >= config_.min_samples
                                       && config_.breach_threshold != 0.0
                                       && cvar_ < config_.breach_threshold) ? "true" : "false")
       << "}";
    return os.str();
}

} // namespace llmquant
