#include "NarrativeDriftDetector.h"

#include <algorithm>
#include <cmath>
#include <sstream>
#include <iomanip>

namespace llmquant {

NarrativeDriftDetector::NarrativeDriftDetector(Config cfg)
    : cfg_(std::move(cfg))
{}

void NarrativeDriftDetector::record(double bias) {
    bool   fire_up = false, fire_dn = false;
    double mag_up  = 0.0,   mag_dn  = 0.0;
    std::function<void(double)> up_cb, dn_cb;

    {
        std::lock_guard<std::mutex> lk(mutex_);
        up_cb = cfg_.on_upward_drift;
        dn_cb = cfg_.on_downward_drift;
        total_.fetch_add(1, std::memory_order_relaxed);

        // Update running mean
        ++n_;
        sum_ += bias;
        double mu = sum_ / static_cast<double>(n_);
        mean_.store(mu, std::memory_order_relaxed);

        // Update Page-Hinkley statistics
        double m_up = bias - mu - cfg_.delta;
        double m_dn = mu - bias - cfg_.delta;

        U_ = std::max(0.0, U_ + m_up);
        D_ = std::max(0.0, D_ + m_dn);

        up_.store(U_, std::memory_order_relaxed);
        dn_.store(D_, std::memory_order_relaxed);

        if (n_ >= static_cast<uint64_t>(std::max(1, cfg_.min_samples))) {
            if (U_ > cfg_.lambda) {
                mag_up  = U_;
                fire_up = true;
                U_      = 0.0;
                up_alarms_.fetch_add(1, std::memory_order_relaxed);
                up_.store(U_, std::memory_order_relaxed);
            }
            if (D_ > cfg_.lambda) {
                mag_dn  = D_;
                fire_dn = true;
                D_      = 0.0;
                dn_alarms_.fetch_add(1, std::memory_order_relaxed);
                dn_.store(D_, std::memory_order_relaxed);
            }
        }
    }

    if (fire_up && up_cb) up_cb(mag_up);
    if (fire_dn && dn_cb) dn_cb(mag_dn);
}

std::string NarrativeDriftDetector::to_stats_json() const {
    std::lock_guard<std::mutex> lk(mutex_);
    std::ostringstream ss;
    ss << std::fixed << std::setprecision(6);
    ss << "{"
       << "\"upward_stat\":"      << up_.load(std::memory_order_relaxed) << ","
       << "\"downward_stat\":"    << dn_.load(std::memory_order_relaxed) << ","
       << "\"running_mean\":"     << mean_.load(std::memory_order_relaxed) << ","
       << "\"upward_alarms\":"    << up_alarms_.load(std::memory_order_relaxed) << ","
       << "\"downward_alarms\":"  << dn_alarms_.load(std::memory_order_relaxed) << ","
       << "\"total_records\":"    << total_.load(std::memory_order_relaxed)
       << "}";
    return ss.str();
}

void NarrativeDriftDetector::reset() {
    std::lock_guard<std::mutex> lk(mutex_);
    U_ = D_ = sum_ = 0.0;
    n_ = 0;
    up_.store(0.0, std::memory_order_relaxed);
    dn_.store(0.0, std::memory_order_relaxed);
    mean_.store(0.0, std::memory_order_relaxed);
    up_alarms_.store(0, std::memory_order_relaxed);
    dn_alarms_.store(0, std::memory_order_relaxed);
    total_.store(0, std::memory_order_relaxed);
}

void NarrativeDriftDetector::update_config(const Config& cfg) {
    std::lock_guard<std::mutex> lk(mutex_);
    cfg_ = cfg;
    U_ = D_ = sum_ = 0.0;
    n_ = 0;
    up_.store(0.0, std::memory_order_relaxed);
    dn_.store(0.0, std::memory_order_relaxed);
    mean_.store(0.0, std::memory_order_relaxed);
}

} // namespace llmquant
