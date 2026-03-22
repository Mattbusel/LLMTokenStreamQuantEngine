#include "SignalPolarizationMonitor.h"

#include <cmath>
#include <iomanip>
#include <numeric>
#include <sstream>

namespace llmquant {

SignalPolarizationMonitor::SignalPolarizationMonitor(Config cfg)
    : cfg_(std::move(cfg))
{
    int w = std::max(4, cfg_.window);
    buf_.resize(static_cast<std::size_t>(w), 0.0);
}

void SignalPolarizationMonitor::record(double bias) {
    std::function<void(double)> pol_cb, uni_cb;
    bool   fire_pol = false, fire_uni = false;
    double bc_val   = 0.0;

    {
        std::lock_guard<std::mutex> lk(mutex_);
        pol_cb = cfg_.on_polarized;
        uni_cb = cfg_.on_unified;
        total_.fetch_add(1, std::memory_order_relaxed);

        buf_[static_cast<std::size_t>(head_)] = bias;
        head_ = (head_ + 1) % static_cast<int>(buf_.size());
        if (fill_ < static_cast<int>(buf_.size())) ++fill_;

        if (fill_ >= std::max(4, cfg_.min_observations)) {
            recompute_locked();
            bc_val = bc_.load(std::memory_order_relaxed);

            bool pol = bc_val > cfg_.polarization_threshold;
            bool was = prev_polarized_;

            polarized_.store(pol, std::memory_order_relaxed);

            if (pol && !was) {
                fire_pol = true;
                events_.fetch_add(1, std::memory_order_relaxed);
            } else if (!pol && was &&
                       bc_val < cfg_.polarization_threshold * cfg_.clear_hysteresis) {
                fire_uni = true;
            }
            prev_polarized_ = pol;
        }
    }

    if (fire_pol && pol_cb) pol_cb(bc_val);
    if (fire_uni && uni_cb) uni_cb(bc_val);
}

void SignalPolarizationMonitor::recompute_locked() {
    int n    = fill_;
    int size = static_cast<int>(buf_.size());

    // Collect the filled window in temporal order.
    double sum = 0.0;
    for (int i = 0; i < n; ++i) {
        int idx = (head_ - n + i + size * 2) % size;
        sum += buf_[static_cast<std::size_t>(idx)];
    }
    double mu = sum / n;

    // Central moments.
    double m2 = 0.0, m3 = 0.0, m4 = 0.0;
    for (int i = 0; i < n; ++i) {
        int    idx = (head_ - n + i + size * 2) % size;
        double d   = buf_[static_cast<std::size_t>(idx)] - mu;
        double d2  = d * d;
        m2 += d2;
        m3 += d2 * d;
        m4 += d2 * d2;
    }
    m2 /= n;
    m3 /= n;
    m4 /= n;

    double skew = 0.0, kurt = 0.0, bc = 0.0;
    if (m2 > 0.0) {
        double sigma = std::sqrt(m2);
        skew = m3 / (sigma * sigma * sigma);
        kurt = m4 / (m2 * m2) - 3.0;  // excess kurtosis

        // Sarle's bimodality coefficient:
        // BC = (γ² + 1) / (κ + 3·(n−1)²/((n−2)·(n−3)))
        // Denominator correction term handles small-sample kurtosis bias.
        double correction = 0.0;
        if (n > 3) {
            double nm1 = static_cast<double>(n - 1);
            double nm2 = static_cast<double>(n - 2);
            double nm3 = static_cast<double>(n - 3);
            correction = 3.0 * nm1 * nm1 / (nm2 * nm3);
        }
        double denom = kurt + correction;
        // Guard against degenerate denominator (platykurtic distributions).
        if (denom > 1e-10) {
            bc = (skew * skew + 1.0) / denom;
        } else {
            // Flat / uniform-like distribution is maximally bimodal-leaning.
            bc = 1.0;
        }
        // Clamp to [0, 1].
        if (bc < 0.0) bc = 0.0;
        if (bc > 1.0) bc = 1.0;
    }

    bc_.store(bc,   std::memory_order_relaxed);
    skewness_.store(skew, std::memory_order_relaxed);
    kurtosis_.store(kurt, std::memory_order_relaxed);
}

std::string SignalPolarizationMonitor::to_stats_json() const {
    double bc, skew, kurt;
    bool   pol;
    uint64_t events, total;
    {
        std::lock_guard<std::mutex> lk(mutex_);
        bc     = bc_.load(std::memory_order_relaxed);
        skew   = skewness_.load(std::memory_order_relaxed);
        kurt   = kurtosis_.load(std::memory_order_relaxed);
        pol    = polarized_.load(std::memory_order_relaxed);
        events = events_.load(std::memory_order_relaxed);
        total  = total_.load(std::memory_order_relaxed);
    }
    std::ostringstream ss;
    ss << std::fixed << std::setprecision(6);
    ss << "{"
       << "\"bimodality_coefficient\":" << bc << ","
       << "\"is_polarized\":"           << (pol ? "true" : "false") << ","
       << "\"skewness\":"               << skew << ","
       << "\"excess_kurtosis\":"        << kurt << ","
       << "\"polarization_events\":"    << events << ","
       << "\"total_records\":"          << total
       << "}";
    return ss.str();
}

void SignalPolarizationMonitor::reset() {
    std::lock_guard<std::mutex> lk(mutex_);
    std::fill(buf_.begin(), buf_.end(), 0.0);
    head_ = fill_ = 0;
    prev_polarized_ = false;
    bc_.store(0.0,   std::memory_order_relaxed);
    skewness_.store(0.0, std::memory_order_relaxed);
    kurtosis_.store(0.0, std::memory_order_relaxed);
    polarized_.store(false, std::memory_order_relaxed);
    events_.store(0, std::memory_order_relaxed);
    total_.store(0, std::memory_order_relaxed);
}

void SignalPolarizationMonitor::update_config(const Config& cfg) {
    std::lock_guard<std::mutex> lk(mutex_);
    cfg_  = cfg;
    int w = std::max(4, cfg_.window);
    buf_.assign(static_cast<std::size_t>(w), 0.0);
    head_ = fill_ = 0;
    prev_polarized_ = false;
    bc_.store(0.0,   std::memory_order_relaxed);
    skewness_.store(0.0, std::memory_order_relaxed);
    kurtosis_.store(0.0, std::memory_order_relaxed);
    polarized_.store(false, std::memory_order_relaxed);
}

} // namespace llmquant
