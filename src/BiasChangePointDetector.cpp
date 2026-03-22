#include "BiasChangePointDetector.h"

#include <algorithm>
#include <iomanip>
#include <sstream>

namespace llmquant {

BiasChangePointDetector::BiasChangePointDetector(Config cfg)
    : cfg_(std::move(cfg))
{}

int BiasChangePointDetector::record(double bias) {
    std::function<void(double)> up_cb, dn_cb;
    bool fire_up = false, fire_dn = false;
    double cp_val = 0.0, cm_val = 0.0;
    int result = 0;

    {
        std::lock_guard<std::mutex> lk(mutex_);
        up_cb = cfg_.on_upshift;
        dn_cb = cfg_.on_downshift;

        total_.fetch_add(1, std::memory_order_relaxed);

        double diff = bias - cfg_.reference_mean;

        cusum_p_ = std::max(0.0, cusum_p_ + diff - cfg_.allowance);
        cusum_m_ = std::max(0.0, cusum_m_ - diff - cfg_.allowance);

        cp_val = cusum_p_;
        cm_val = cusum_m_;

        cusum_plus_.store(cp_val,  std::memory_order_relaxed);
        cusum_minus_.store(cm_val, std::memory_order_relaxed);

        if (cusum_p_ > cfg_.threshold) {
            fire_up   = true;
            result    = +1;
            cusum_p_  = 0.0;  // zero-restart
            cusum_plus_.store(0.0, std::memory_order_relaxed);
            upshift_count_.fetch_add(1, std::memory_order_relaxed);
        } else if (cusum_m_ > cfg_.threshold) {
            fire_dn   = true;
            result    = -1;
            cusum_m_  = 0.0;  // zero-restart
            cusum_minus_.store(0.0, std::memory_order_relaxed);
            downshift_count_.fetch_add(1, std::memory_order_relaxed);
        }
    }

    if (fire_up && up_cb) up_cb(cp_val);
    if (fire_dn && dn_cb) dn_cb(cm_val);
    return result;
}

std::string BiasChangePointDetector::to_stats_json() const {
    double cp, cm;
    uint64_t ups, dns, total;
    {
        std::lock_guard<std::mutex> lk(mutex_);
        cp    = cusum_plus_.load(std::memory_order_relaxed);
        cm    = cusum_minus_.load(std::memory_order_relaxed);
        ups   = upshift_count_.load(std::memory_order_relaxed);
        dns   = downshift_count_.load(std::memory_order_relaxed);
        total = total_.load(std::memory_order_relaxed);
    }
    std::ostringstream ss;
    ss << std::fixed << std::setprecision(6);
    ss << "{"
       << "\"cusum_plus\":"       << cp    << ","
       << "\"cusum_minus\":"      << cm    << ","
       << "\"upshift_count\":"    << ups   << ","
       << "\"downshift_count\":"  << dns   << ","
       << "\"total_records\":"    << total
       << "}";
    return ss.str();
}

void BiasChangePointDetector::reset() {
    std::lock_guard<std::mutex> lk(mutex_);
    cusum_p_ = cusum_m_ = 0.0;
    cusum_plus_.store(0.0,  std::memory_order_relaxed);
    cusum_minus_.store(0.0, std::memory_order_relaxed);
    upshift_count_.store(0,   std::memory_order_relaxed);
    downshift_count_.store(0, std::memory_order_relaxed);
    total_.store(0, std::memory_order_relaxed);
}

void BiasChangePointDetector::update_config(const Config& cfg) {
    std::lock_guard<std::mutex> lk(mutex_);
    cfg_     = cfg;
    cusum_p_ = cusum_m_ = 0.0;
    cusum_plus_.store(0.0,  std::memory_order_relaxed);
    cusum_minus_.store(0.0, std::memory_order_relaxed);
    upshift_count_.store(0,   std::memory_order_relaxed);
    downshift_count_.store(0, std::memory_order_relaxed);
    total_.store(0, std::memory_order_relaxed);
}

} // namespace llmquant
