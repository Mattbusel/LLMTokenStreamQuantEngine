#include "BiasZScoreNormaliser.h"
#include <cmath>
#include <numeric>
#include <sstream>
#include <iomanip>

namespace llmquant {

BiasZScoreNormaliser::BiasZScoreNormaliser(Config cfg)
    : cfg_(std::move(cfg))
    , buf_(cfg_.window, 0.0)
{}

void BiasZScoreNormaliser::record(double bias) {
    double new_z = 0.0, new_mean = 0.0, new_std = 0.0;
    bool pos_ev = false, neg_ev = false;

    {
        std::lock_guard<std::mutex> lk(mutex_);

        buf_[head_] = bias;
        head_ = (head_ + 1) % cfg_.window;
        if (fill_ < cfg_.window) ++fill_;

        if (fill_ >= cfg_.min_samples) {
            int n = fill_;
            double sum = 0.0;
            for (int i = 0; i < n; ++i) sum += buf_[i];
            new_mean = sum / n;

            double sq_sum = 0.0;
            for (int i = 0; i < n; ++i) {
                double d = buf_[i] - new_mean;
                sq_sum += d * d;
            }
            new_std = std::sqrt(sq_sum / n);
            new_z   = (new_std > 1e-12) ? (bias - new_mean) / new_std : 0.0;

            if (new_z > cfg_.extreme_threshold) {
                pos_ev = true;
                pos_ext_.fetch_add(1, std::memory_order_relaxed);
            } else if (new_z < -cfg_.extreme_threshold) {
                neg_ev = true;
                neg_ext_.fetch_add(1, std::memory_order_relaxed);
            }
        }
    }

    z_.store(new_z, std::memory_order_relaxed);
    mean_.store(new_mean, std::memory_order_relaxed);
    std_.store(new_std, std::memory_order_relaxed);
    total_.fetch_add(1, std::memory_order_relaxed);

    if (pos_ev && cfg_.on_positive_extreme) cfg_.on_positive_extreme(new_z);
    if (neg_ev && cfg_.on_negative_extreme) cfg_.on_negative_extreme(new_z);
}

double BiasZScoreNormaliser::z_score()      const noexcept { return z_.load(std::memory_order_relaxed); }
double BiasZScoreNormaliser::rolling_mean() const noexcept { return mean_.load(std::memory_order_relaxed); }
double BiasZScoreNormaliser::rolling_std()  const noexcept { return std_.load(std::memory_order_relaxed); }

std::size_t BiasZScoreNormaliser::positive_extremes() const noexcept { return pos_ext_.load(std::memory_order_relaxed); }
std::size_t BiasZScoreNormaliser::negative_extremes() const noexcept { return neg_ext_.load(std::memory_order_relaxed); }
std::size_t BiasZScoreNormaliser::total_records()     const noexcept { return total_.load(std::memory_order_relaxed); }

std::string BiasZScoreNormaliser::to_stats_json() const {
    std::lock_guard<std::mutex> lk(mutex_);
    std::ostringstream ss;
    ss << std::fixed << std::setprecision(4);
    ss << "{\"z_score\":"         << z_.load(std::memory_order_relaxed)
       << ",\"rolling_mean\":"    << mean_.load(std::memory_order_relaxed)
       << ",\"rolling_std\":"     << std_.load(std::memory_order_relaxed)
       << ",\"positive_extremes\":" << pos_ext_.load(std::memory_order_relaxed)
       << ",\"negative_extremes\":" << neg_ext_.load(std::memory_order_relaxed)
       << ",\"total_records\":"   << total_.load(std::memory_order_relaxed)
       << "}";
    return ss.str();
}

void BiasZScoreNormaliser::reset() {
    std::lock_guard<std::mutex> lk(mutex_);
    std::fill(buf_.begin(), buf_.end(), 0.0);
    head_ = 0; fill_ = 0;
    z_.store(0.0, std::memory_order_relaxed);
    mean_.store(0.0, std::memory_order_relaxed);
    std_.store(0.0, std::memory_order_relaxed);
    pos_ext_.store(0, std::memory_order_relaxed);
    neg_ext_.store(0, std::memory_order_relaxed);
    total_.store(0, std::memory_order_relaxed);
}

void BiasZScoreNormaliser::update_config(const Config& cfg) {
    std::lock_guard<std::mutex> lk(mutex_);
    cfg_ = cfg;
    buf_.assign(cfg_.window, 0.0);
    head_ = 0; fill_ = 0;
}

} // namespace llmquant
