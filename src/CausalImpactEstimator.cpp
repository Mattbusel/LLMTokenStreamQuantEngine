#include "CausalImpactEstimator.h"

#include <cmath>
#include <sstream>
#include <iomanip>
#include <numeric>

namespace llmquant {

CausalImpactEstimator::CausalImpactEstimator(Config cfg)
    : cfg_(std::move(cfg))
{
    mu_hat_ = cfg_.baseline_mean;
    return_window_.resize(static_cast<std::size_t>(std::max(1, cfg_.event_lookback)));
}

double CausalImpactEstimator::record_return(double ret) {
    std::function<void(double, const std::string&, double)> cb;
    bool   fire   = false;
    double stat   = 0.0;
    double impact = 0.0;
    std::string event_label;

    {
        std::lock_guard<std::mutex> lk(mutex_);
        cb = cfg_.on_break;

        uint64_t idx = obs_count_.load(std::memory_order_relaxed);
        obs_count_.store(idx + 1, std::memory_order_relaxed);

        // Update rolling baseline (Welford single-pass).
        ++n_obs_;
        if (!baseline_locked_) {
            mu_hat_ += (ret - mu_hat_) / n_obs_;
            if (n_obs_ >= cfg_.warmup_window)
                baseline_locked_ = true;
        }

        double mu = baseline_locked_ ? mu_hat_ : cfg_.baseline_mean;

        // Update rolling return window.
        return_window_[static_cast<std::size_t>(return_head_)] = ret;
        return_head_ = (return_head_ + 1) % static_cast<int>(return_window_.size());
        if (return_fill_ < static_cast<int>(return_window_.size())) ++return_fill_;

        // CUSUM update: upper-side (positive deviation).
        cusum_s_ = std::max(0.0, cusum_s_ + (ret - mu - cfg_.sensitivity));
        cusum_.store(cusum_s_, std::memory_order_relaxed);
        stat = cusum_s_;

        // Break detection.
        bool was_break = break_detected_.load(std::memory_order_relaxed);
        if (!was_break && cusum_s_ > cfg_.threshold) {
            break_detected_.store(true, std::memory_order_relaxed);
            break_count_.fetch_add(1, std::memory_order_relaxed);
            // Find attributed event.
            event_label = find_recent_event_locked(idx);
            impact      = post_break_impact_locked();
            fire        = true;
            // Reset CUSUM after break.
            cusum_s_ = 0.0;
            cusum_.store(0.0, std::memory_order_relaxed);
        } else if (was_break && cusum_s_ <= 0.0) {
            break_detected_.store(false, std::memory_order_relaxed);
        }
    }

    if (fire && cb) cb(stat, event_label, impact);
    return stat;
}

void CausalImpactEstimator::record_event(const std::string& label) {
    std::lock_guard<std::mutex> lk(mutex_);
    uint64_t idx = obs_count_.load(std::memory_order_relaxed);
    events_.push_back({label, idx});
    // Trim old events beyond 2× lookback.
    while (!events_.empty() &&
           idx - events_.front().obs_index >
               static_cast<uint64_t>(2 * cfg_.event_lookback)) {
        events_.erase(events_.begin());
    }
}

std::string CausalImpactEstimator::find_recent_event_locked(uint64_t obs_index) const {
    // Find the most recent event within event_lookback steps.
    for (auto it = events_.rbegin(); it != events_.rend(); ++it) {
        if (obs_index >= it->obs_index &&
            obs_index - it->obs_index <= static_cast<uint64_t>(cfg_.event_lookback)) {
            return it->label;
        }
    }
    return {};
}

double CausalImpactEstimator::post_break_impact_locked() const {
    if (return_fill_ == 0) return 0.0;
    double sum = 0.0;
    for (int i = 0; i < return_fill_; ++i) sum += return_window_[i];
    double post_mean = sum / return_fill_;
    return post_mean - mu_hat_;
}

double CausalImpactEstimator::baseline_mean_estimate() const noexcept {
    std::lock_guard<std::mutex> lk(mutex_);
    return mu_hat_;
}

void CausalImpactEstimator::reset_cusum() {
    std::lock_guard<std::mutex> lk(mutex_);
    cusum_s_ = 0.0;
    cusum_.store(0.0, std::memory_order_relaxed);
    break_detected_.store(false, std::memory_order_relaxed);
}

void CausalImpactEstimator::reset() {
    std::lock_guard<std::mutex> lk(mutex_);
    cusum_s_  = 0.0;
    mu_hat_   = cfg_.baseline_mean;
    n_obs_    = 0;
    baseline_locked_ = false;
    return_head_ = return_fill_ = 0;
    std::fill(return_window_.begin(), return_window_.end(), 0.0);
    events_.clear();
    cusum_.store(0.0, std::memory_order_relaxed);
    break_detected_.store(false, std::memory_order_relaxed);
    break_count_.store(0, std::memory_order_relaxed);
    obs_count_.store(0, std::memory_order_relaxed);
}

void CausalImpactEstimator::update_config(const Config& cfg) {
    std::lock_guard<std::mutex> lk(mutex_);
    cfg_ = cfg;
}

std::string CausalImpactEstimator::to_stats_json() const {
    double cs;
    bool bd;
    uint64_t bc, oc;
    double mu;
    {
        std::lock_guard<std::mutex> lk(mutex_);
        cs = cusum_.load(std::memory_order_relaxed);
        bd = break_detected_.load(std::memory_order_relaxed);
        bc = break_count_.load(std::memory_order_relaxed);
        oc = obs_count_.load(std::memory_order_relaxed);
        mu = mu_hat_;
    }
    std::ostringstream ss;
    ss << std::fixed << std::setprecision(6);
    ss << "{"
       << "\"cusum_stat\":"       << cs << ","
       << "\"break_detected\":"   << (bd ? "true" : "false") << ","
       << "\"break_count\":"      << bc << ","
       << "\"observations\":"     << oc << ","
       << "\"baseline_mean\":"    << mu
       << "}";
    return ss.str();
}

} // namespace llmquant
