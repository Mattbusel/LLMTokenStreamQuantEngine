#include "SignalDriftMonitor.h"

#include <algorithm>
#include <cmath>
#include <numeric>
#include <sstream>

namespace llmquant {

// ---------------------------------------------------------------------------
// Construction
// ---------------------------------------------------------------------------

SignalDriftMonitor::SignalDriftMonitor() {
    baseline_.resize(cfg_.baseline_size, 0.0);
    recent_.resize(cfg_.recent_size, 0.0);
}

SignalDriftMonitor::SignalDriftMonitor(const Config& cfg) : cfg_(cfg) {
    baseline_.resize(cfg_.baseline_size, 0.0);
    recent_.resize(cfg_.recent_size, 0.0);
}

// ---------------------------------------------------------------------------
// Internal helper
// ---------------------------------------------------------------------------

double SignalDriftMonitor::compute_w1(std::vector<double> a,
                                       std::vector<double> b) noexcept {
    if (a.empty() || b.empty()) return 0.0;
    std::sort(a.begin(), a.end());
    std::sort(b.begin(), b.end());

    // Merge-and-step integration of |CDF_a(x) - CDF_b(x)|.
    // For equal-sized sorted arrays the W1 = (1/N) * Σ |a[i] - b[i]|.
    if (a.size() == b.size()) {
        double s = 0.0;
        for (size_t i = 0; i < a.size(); ++i)
            s += std::abs(a[i] - b[i]);
        return s / static_cast<double>(a.size());
    }

    // Different sizes: resample b to the size of a via linear interpolation.
    int na = static_cast<int>(a.size());
    int nb = static_cast<int>(b.size());
    double s = 0.0;
    for (int i = 0; i < na; ++i) {
        double t = static_cast<double>(i) / (na - 1) * (nb - 1);
        int    j = static_cast<int>(t);
        double f = t - j;
        double bv = (j + 1 < nb) ? (b[j] * (1.0 - f) + b[j+1] * f) : b[j];
        s += std::abs(a[i] - bv);
    }
    return s / static_cast<double>(na);
}

// ---------------------------------------------------------------------------
// Core
// ---------------------------------------------------------------------------

void SignalDriftMonitor::record(double value) {
    std::function<void(double)> cb;
    double w1_val = 0.0;
    bool fire_drift   = false;
    bool fire_cleared = false;

    {
        std::lock_guard<std::mutex> lk(mutex_);

        int bs = static_cast<int>(baseline_.size());
        int rs = static_cast<int>(recent_.size());

        // Push into baseline ring.
        baseline_[baseline_head_] = value;
        baseline_head_ = (baseline_head_ + 1) % bs;
        if (baseline_count_ < bs) ++baseline_count_;

        // Push into recent ring.
        recent_[recent_head_] = value;
        recent_head_ = (recent_head_ + 1) % rs;
        if (recent_count_ < rs) ++recent_count_;

        // Only compute W1 when both windows are at least half full.
        if (baseline_count_ >= bs / 2 && recent_count_ >= rs) {
            // Build sorted views
            std::vector<double> bvec(baseline_count_);
            for (int i = 0; i < baseline_count_; ++i)
                bvec[i] = baseline_[(baseline_head_ - baseline_count_ + i + bs) % bs];

            std::vector<double> rvec(recent_count_);
            for (int i = 0; i < recent_count_; ++i)
                rvec[i] = recent_[(recent_head_ - recent_count_ + i + rs) % rs];

            w1_val = compute_w1(std::move(bvec), std::move(rvec));

            if (!drifting_ && w1_val > cfg_.drift_threshold) {
                drifting_ = true;
                drift_events_.fetch_add(1, std::memory_order_relaxed);
                fire_drift = true;
                cb = cfg_.on_drift_detected;
            } else if (drifting_ && w1_val < cfg_.drift_threshold * cfg_.clear_hysteresis) {
                drifting_ = false;
                fire_cleared = true;
                cb = cfg_.on_drift_cleared;
            }
        }
    }

    last_w1_.store(w1_val, std::memory_order_relaxed);
    drift_flag_.store(drifting_, std::memory_order_relaxed);
    total_.fetch_add(1, std::memory_order_relaxed);

    if (fire_drift && cb)   cb(w1_val);
    if (fire_cleared && cb) cb(w1_val);
}

// ---------------------------------------------------------------------------
// Accessors
// ---------------------------------------------------------------------------

double SignalDriftMonitor::last_w1() const noexcept {
    return last_w1_.load(std::memory_order_relaxed);
}

bool SignalDriftMonitor::is_drifting() const noexcept {
    return drift_flag_.load(std::memory_order_relaxed);
}

uint64_t SignalDriftMonitor::total_records() const noexcept {
    return total_.load(std::memory_order_relaxed);
}

uint64_t SignalDriftMonitor::drift_events() const noexcept {
    return drift_events_.load(std::memory_order_relaxed);
}

double SignalDriftMonitor::wasserstein_distance() const noexcept {
    std::lock_guard<std::mutex> lk(mutex_);
    int bs = static_cast<int>(baseline_.size());
    int rs = static_cast<int>(recent_.size());
    if (baseline_count_ == 0 || recent_count_ == 0) return 0.0;

    std::vector<double> bvec(baseline_count_);
    for (int i = 0; i < baseline_count_; ++i)
        bvec[i] = baseline_[(baseline_head_ - baseline_count_ + i + bs) % bs];

    std::vector<double> rvec(recent_count_);
    for (int i = 0; i < recent_count_; ++i)
        rvec[i] = recent_[(recent_head_ - recent_count_ + i + rs) % rs];

    return compute_w1(std::move(bvec), std::move(rvec));
}

double SignalDriftMonitor::baseline_mean() const noexcept {
    std::lock_guard<std::mutex> lk(mutex_);
    if (baseline_count_ == 0) return 0.0;
    int bs = static_cast<int>(baseline_.size());
    double s = 0.0;
    for (int i = 0; i < baseline_count_; ++i)
        s += baseline_[(baseline_head_ - baseline_count_ + i + bs) % bs];
    return s / baseline_count_;
}

double SignalDriftMonitor::recent_mean() const noexcept {
    std::lock_guard<std::mutex> lk(mutex_);
    if (recent_count_ == 0) return 0.0;
    int rs = static_cast<int>(recent_.size());
    double s = 0.0;
    for (int i = 0; i < recent_count_; ++i)
        s += recent_[(recent_head_ - recent_count_ + i + rs) % rs];
    return s / recent_count_;
}

// ---------------------------------------------------------------------------
// Mutation
// ---------------------------------------------------------------------------

void SignalDriftMonitor::reset() {
    std::lock_guard<std::mutex> lk(mutex_);
    std::fill(baseline_.begin(), baseline_.end(), 0.0);
    std::fill(recent_.begin(), recent_.end(), 0.0);
    baseline_head_  = 0;
    baseline_count_ = 0;
    recent_head_    = 0;
    recent_count_   = 0;
    drifting_       = false;

    last_w1_.store(0.0, std::memory_order_relaxed);
    drift_flag_.store(false, std::memory_order_relaxed);
    total_.store(0, std::memory_order_relaxed);
    drift_events_.store(0, std::memory_order_relaxed);
}

void SignalDriftMonitor::update_config(const Config& cfg) {
    std::lock_guard<std::mutex> lk(mutex_);
    cfg_ = cfg;
    baseline_.assign(cfg_.baseline_size, 0.0);
    recent_.assign(cfg_.recent_size, 0.0);
    baseline_head_  = 0;
    baseline_count_ = 0;
    recent_head_    = 0;
    recent_count_   = 0;
    drifting_       = false;

    last_w1_.store(0.0, std::memory_order_relaxed);
    drift_flag_.store(false, std::memory_order_relaxed);
    total_.store(0, std::memory_order_relaxed);
    drift_events_.store(0, std::memory_order_relaxed);
}

// ---------------------------------------------------------------------------
// Diagnostics
// ---------------------------------------------------------------------------

std::string SignalDriftMonitor::to_stats_json() const {
    double w1    = last_w1();
    bool   drift = is_drifting();
    auto   tot   = total_records();
    auto   evts  = drift_events();

    std::ostringstream os;
    os << "{"
       << "\"wasserstein_1\":"  << w1    << ","
       << "\"is_drifting\":"    << (drift ? "true" : "false") << ","
       << "\"drift_events\":"   << evts  << ","
       << "\"total_records\":"  << tot
       << "}";
    return os.str();
}

}  // namespace llmquant
