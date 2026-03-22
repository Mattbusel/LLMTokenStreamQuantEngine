#include "BiasConditionalDistribution.h"

#include <algorithm>
#include <cmath>
#include <sstream>
#include <iomanip>

namespace llmquant {

BiasConditionalDistribution::BiasConditionalDistribution(Config cfg)
    : cfg_(std::move(cfg))
{
    int w = std::max(2, cfg_.window);
    int b = std::max(2, cfg_.n_bins);
    pos_win_.assign(static_cast<std::size_t>(w), 0);
    neg_win_.assign(static_cast<std::size_t>(w), 0);
    pos_hist_.assign(static_cast<std::size_t>(b), 0);
    neg_hist_.assign(static_cast<std::size_t>(b), 0);
}

int BiasConditionalDistribution::bin_for(double bias) const {
    int b = static_cast<int>(pos_hist_.size());
    double rng = cfg_.range_max - cfg_.range_min;
    if (rng <= 0.0) return 0;
    double frac = (bias - cfg_.range_min) / rng;
    frac = std::max(0.0, std::min(1.0 - 1e-12, frac));
    return static_cast<int>(frac * b);
}

double BiasConditionalDistribution::compute_tv_locked() const {
    int pos_n = pos_fill_;
    int neg_n = neg_fill_;
    if (pos_n == 0 || neg_n == 0) return 0.0;

    int b = static_cast<int>(pos_hist_.size());
    double tv = 0.0;
    for (int i = 0; i < b; ++i) {
        double p = pos_hist_[static_cast<std::size_t>(i)] / static_cast<double>(pos_n);
        double q = neg_hist_[static_cast<std::size_t>(i)] / static_cast<double>(neg_n);
        tv += std::abs(p - q);
    }
    return 0.5 * tv;
}

void BiasConditionalDistribution::record(double bias) {
    bool   fire_asym = false, fire_sym = false;
    double tv_val    = 0.0;
    std::function<void(double)> asym_cb, sym_cb;

    {
        std::lock_guard<std::mutex> lk(mutex_);
        asym_cb = cfg_.on_asymmetry_detected;
        sym_cb  = cfg_.on_symmetry_restored;
        total_.fetch_add(1, std::memory_order_relaxed);

        if (!seeded_) {
            prev_bias_ = bias;
            seeded_    = true;
            return;
        }

        int bkt = bin_for(bias);
        int w   = static_cast<int>(pos_win_.size());

        if (prev_bias_ > 0.0) {
            // Record into pos conditional histogram
            if (pos_fill_ == w) {
                int old = pos_win_[static_cast<std::size_t>(pos_head_)];
                if (pos_hist_[static_cast<std::size_t>(old)] > 0)
                    --pos_hist_[static_cast<std::size_t>(old)];
            } else {
                ++pos_fill_;
            }
            pos_win_[static_cast<std::size_t>(pos_head_)] = bkt;
            pos_head_ = (pos_head_ + 1) % w;
            ++pos_hist_[static_cast<std::size_t>(bkt)];
        } else if (prev_bias_ < 0.0) {
            // Record into neg conditional histogram
            if (neg_fill_ == w) {
                int old = neg_win_[static_cast<std::size_t>(neg_head_)];
                if (neg_hist_[static_cast<std::size_t>(old)] > 0)
                    --neg_hist_[static_cast<std::size_t>(old)];
            } else {
                ++neg_fill_;
            }
            neg_win_[static_cast<std::size_t>(neg_head_)] = bkt;
            neg_head_ = (neg_head_ + 1) % w;
            ++neg_hist_[static_cast<std::size_t>(bkt)];
        }

        prev_bias_ = bias;

        tv_val = compute_tv_locked();
        tv_.store(tv_val, std::memory_order_relaxed);

        bool now_asym = tv_val > cfg_.asymmetry_threshold;
        asymmetric_.store(now_asym, std::memory_order_relaxed);

        if (now_asym && !prev_asymmetric_) fire_asym = true;
        if (!now_asym && prev_asymmetric_ && tv_val < cfg_.asymmetry_threshold * 0.7)
            fire_sym = true;
        prev_asymmetric_ = now_asym;
    }

    if (fire_asym && asym_cb) asym_cb(tv_val);
    if (fire_sym  && sym_cb)  sym_cb(tv_val);
}

std::string BiasConditionalDistribution::to_stats_json() const {
    std::lock_guard<std::mutex> lk(mutex_);
    std::ostringstream ss;
    ss << std::fixed << std::setprecision(6);
    ss << "{"
       << "\"tv_distance\":"   << tv_.load(std::memory_order_relaxed) << ","
       << "\"is_asymmetric\":" << (asymmetric_.load(std::memory_order_relaxed) ? "true" : "false") << ","
       << "\"total_records\":" << total_.load(std::memory_order_relaxed)
       << "}";
    return ss.str();
}

void BiasConditionalDistribution::reset() {
    std::lock_guard<std::mutex> lk(mutex_);
    std::fill(pos_win_.begin(), pos_win_.end(), 0);
    std::fill(neg_win_.begin(), neg_win_.end(), 0);
    std::fill(pos_hist_.begin(), pos_hist_.end(), 0);
    std::fill(neg_hist_.begin(), neg_hist_.end(), 0);
    pos_head_ = pos_fill_ = 0;
    neg_head_ = neg_fill_ = 0;
    seeded_ = false;
    prev_asymmetric_ = false;
    tv_.store(0.0, std::memory_order_relaxed);
    asymmetric_.store(false, std::memory_order_relaxed);
    total_.store(0, std::memory_order_relaxed);
}

void BiasConditionalDistribution::update_config(const Config& cfg) {
    std::lock_guard<std::mutex> lk(mutex_);
    cfg_ = cfg;
    int w = std::max(2, cfg_.window);
    int b = std::max(2, cfg_.n_bins);
    pos_win_.assign(static_cast<std::size_t>(w), 0);
    neg_win_.assign(static_cast<std::size_t>(w), 0);
    pos_hist_.assign(static_cast<std::size_t>(b), 0);
    neg_hist_.assign(static_cast<std::size_t>(b), 0);
    pos_head_ = pos_fill_ = 0;
    neg_head_ = neg_fill_ = 0;
    seeded_ = false;
    prev_asymmetric_ = false;
    tv_.store(0.0, std::memory_order_relaxed);
    asymmetric_.store(false, std::memory_order_relaxed);
    total_.store(0, std::memory_order_relaxed);
}

} // namespace llmquant
