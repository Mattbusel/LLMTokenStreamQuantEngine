#include "NarrativeMassIndex.h"
#include <cmath>
#include <numeric>
#include <sstream>
#include <iomanip>

namespace llmquant {

NarrativeMassIndex::NarrativeMassIndex(Config cfg)
    : cfg_(std::move(cfg))
    , alpha1_(2.0 / (cfg_.fast_period + 1))
    , alpha2_(2.0 / (cfg_.slow_period + 1))
    , ratio_buf_(cfg_.period, 1.0) // initialise to 1.0 (neutral ratio)
{
    if (cfg_.min_samples < 0) cfg_.min_samples = cfg_.period;
}

void NarrativeMassIndex::record(double bias) {
    double new_mi{};
    bool   bulge_start = false;
    bool   reversal_sig = false;

    {
        std::lock_guard<std::mutex> lk(mutex_);

        double abs_bias = std::fabs(bias);
        ema1_ = ema1_ + alpha1_ * (abs_bias - ema1_);
        ema2_ = ema2_ + alpha2_ * (ema1_    - ema2_);

        double ratio = (ema2_ > 1e-12) ? ema1_ / ema2_ : 1.0;

        ratio_buf_[ratio_head_] = ratio;
        ratio_head_ = (ratio_head_ + 1) % cfg_.period;
        if (ratio_fill_ < cfg_.period) ++ratio_fill_;

        if (ratio_fill_ >= cfg_.min_samples) {
            double sum = 0.0;
            for (int i = 0; i < ratio_fill_; ++i) sum += ratio_buf_[i];
            new_mi = sum;

            bool in_b = (new_mi > cfg_.reversal_bulge);
            if (in_b && !was_in_bulge_) {
                bulge_start = true;
                in_bulge_.store(true, std::memory_order_relaxed);
                bulge_events_.fetch_add(1, std::memory_order_relaxed);
            }
            if (!in_b && was_in_bulge_ && new_mi < cfg_.reversal_trigger) {
                reversal_sig = true;
                in_bulge_.store(false, std::memory_order_relaxed);
                reversal_events_.fetch_add(1, std::memory_order_relaxed);
            }
            if (!in_b) in_bulge_.store(false, std::memory_order_relaxed);
            was_in_bulge_ = in_b;
        }
    }

    mi_.store(new_mi, std::memory_order_relaxed);
    total_.fetch_add(1, std::memory_order_relaxed);

    if (bulge_start  && cfg_.on_bulge_start)    cfg_.on_bulge_start(new_mi);
    if (reversal_sig && cfg_.on_reversal_signal) cfg_.on_reversal_signal(new_mi);
}

std::string NarrativeMassIndex::to_stats_json() const {
    std::lock_guard<std::mutex> lk(mutex_);
    std::ostringstream ss;
    ss << std::fixed << std::setprecision(4);
    ss << "{\"mass_index\":" << mi_.load(std::memory_order_relaxed)
       << ",\"in_bulge\":" << (in_bulge_.load(std::memory_order_relaxed) ? "true" : "false")
       << ",\"bulge_events\":" << bulge_events_.load(std::memory_order_relaxed)
       << ",\"reversal_events\":" << reversal_events_.load(std::memory_order_relaxed)
       << ",\"total_records\":" << total_.load(std::memory_order_relaxed)
       << "}";
    return ss.str();
}

void NarrativeMassIndex::reset() {
    std::lock_guard<std::mutex> lk(mutex_);
    ema1_ = 0.0; ema2_ = 0.0;
    std::fill(ratio_buf_.begin(), ratio_buf_.end(), 1.0);
    ratio_head_ = 0; ratio_fill_ = 0;
    was_in_bulge_ = false;
    mi_.store(0.0, std::memory_order_relaxed);
    in_bulge_.store(false, std::memory_order_relaxed);
    bulge_events_.store(0, std::memory_order_relaxed);
    reversal_events_.store(0, std::memory_order_relaxed);
    total_.store(0, std::memory_order_relaxed);
}

void NarrativeMassIndex::update_config(const Config& cfg) {
    std::lock_guard<std::mutex> lk(mutex_);
    cfg_    = cfg;
    alpha1_ = 2.0 / (cfg_.fast_period + 1);
    alpha2_ = 2.0 / (cfg_.slow_period + 1);
    ratio_buf_.assign(cfg_.period, 1.0);
    ratio_head_ = 0; ratio_fill_ = 0;
}

} // namespace llmquant
