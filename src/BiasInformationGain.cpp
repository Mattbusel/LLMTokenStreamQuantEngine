#include "BiasInformationGain.h"

#include <algorithm>
#include <cmath>
#include <sstream>
#include <iomanip>

namespace llmquant {

BiasInformationGain::BiasInformationGain(Config cfg)
    : cfg_(std::move(cfg))
{
    int b = std::max(2, cfg_.n_bins);
    int w = std::max(4, cfg_.window);
    ring_.assign(static_cast<std::size_t>(w), 0);
    joint_.assign(static_cast<std::size_t>(b * b), 0);
}

int BiasInformationGain::bin_for(double v) const {
    int b   = std::max(2, cfg_.n_bins);
    double rng = cfg_.range_max - cfg_.range_min;
    if (rng <= 0.0) return 0;
    double frac = (v - cfg_.range_min) / rng;
    frac = std::max(0.0, std::min(1.0 - 1e-12, frac));
    return static_cast<int>(frac * b);
}

void BiasInformationGain::compute_mi_locked() {
    int n = ring_fill_;
    if (n < std::max(2, cfg_.min_samples)) return;

    int b = std::max(2, cfg_.n_bins);
    double N = static_cast<double>(n);
    constexpr double eps = 1e-12;

    // Marginals from joint
    std::vector<double> px(static_cast<std::size_t>(b), 0.0);
    std::vector<double> py(static_cast<std::size_t>(b), 0.0);
    for (int i = 0; i < b; ++i)
        for (int j = 0; j < b; ++j) {
            double c = joint_[static_cast<std::size_t>(i * b + j)] / N;
            px[static_cast<std::size_t>(i)] += c;
            py[static_cast<std::size_t>(j)] += c;
        }

    // MI = Σ p(i,j) log2(p(i,j) / (p(i)*p(j)))
    double MI = 0.0;
    for (int i = 0; i < b; ++i)
        for (int j = 0; j < b; ++j) {
            double pij = joint_[static_cast<std::size_t>(i * b + j)] / N;
            if (pij < eps) continue;
            double denom = px[static_cast<std::size_t>(i)] * py[static_cast<std::size_t>(j)];
            if (denom < eps) continue;
            MI += pij * std::log2(pij / denom);
        }
    MI = std::max(0.0, MI);

    // H(X) = -Σ px[i] log2(px[i])
    double HX = 0.0;
    for (int i = 0; i < b; ++i) {
        double p = px[static_cast<std::size_t>(i)];
        if (p > eps) HX -= p * std::log2(p);
    }

    double NMI = (HX > eps) ? std::min(1.0, MI / HX) : 0.0;

    mi_.store(MI, std::memory_order_relaxed);
    nmi_.store(NMI, std::memory_order_relaxed);
}

void BiasInformationGain::record(double bias) {
    bool   fire     = false;
    double old_nmi  = 0.0, new_nmi = 0.0;
    std::function<void(double, double)> cb;

    {
        std::lock_guard<std::mutex> lk(mutex_);
        cb = cfg_.on_mi_change;
        total_.fetch_add(1, std::memory_order_relaxed);

        int bin = bin_for(bias);
        int b   = std::max(2, cfg_.n_bins);

        if (!seeded_) {
            prev_bin_ = bin;
            seeded_   = true;
            return;
        }

        int pair = prev_bin_ * b + bin;
        int w    = static_cast<int>(ring_.size());

        // Evict oldest pair
        if (ring_fill_ == w) {
            int old_pair = ring_[static_cast<std::size_t>(ring_head_)];
            if (joint_[static_cast<std::size_t>(old_pair)] > 0)
                --joint_[static_cast<std::size_t>(old_pair)];
        } else {
            ++ring_fill_;
        }
        ring_[static_cast<std::size_t>(ring_head_)] = pair;
        ring_head_ = (ring_head_ + 1) % w;
        ++joint_[static_cast<std::size_t>(pair)];

        prev_bin_ = bin;

        compute_mi_locked();

        double cur_nmi = nmi_.load(std::memory_order_relaxed);
        if (prev_nmi_ >= 0.0 && std::abs(cur_nmi - prev_nmi_) > cfg_.change_threshold) {
            old_nmi = prev_nmi_;
            new_nmi = cur_nmi;
            fire    = true;
            change_events_.fetch_add(1, std::memory_order_relaxed);
        }
        prev_nmi_ = cur_nmi;
    }

    if (fire && cb) cb(old_nmi, new_nmi);
}

std::string BiasInformationGain::to_stats_json() const {
    std::lock_guard<std::mutex> lk(mutex_);
    std::ostringstream ss;
    ss << std::fixed << std::setprecision(6);
    ss << "{"
       << "\"mutual_information\":"  << mi_.load(std::memory_order_relaxed) << ","
       << "\"normalised_mi\":"       << nmi_.load(std::memory_order_relaxed) << ","
       << "\"mi_change_events\":"    << change_events_.load(std::memory_order_relaxed) << ","
       << "\"total_records\":"       << total_.load(std::memory_order_relaxed)
       << "}";
    return ss.str();
}

void BiasInformationGain::reset() {
    std::lock_guard<std::mutex> lk(mutex_);
    std::fill(ring_.begin(),  ring_.end(),  0);
    std::fill(joint_.begin(), joint_.end(), 0);
    ring_head_ = ring_fill_ = 0;
    seeded_    = false;
    prev_bin_  = -1;
    prev_nmi_  = -1.0;
    mi_.store(0.0, std::memory_order_relaxed);
    nmi_.store(0.0, std::memory_order_relaxed);
    change_events_.store(0, std::memory_order_relaxed);
    total_.store(0, std::memory_order_relaxed);
}

void BiasInformationGain::update_config(const Config& cfg) {
    std::lock_guard<std::mutex> lk(mutex_);
    cfg_ = cfg;
    int b = std::max(2, cfg_.n_bins);
    int w = std::max(4, cfg_.window);
    ring_.assign(static_cast<std::size_t>(w), 0);
    joint_.assign(static_cast<std::size_t>(b * b), 0);
    ring_head_ = ring_fill_ = 0;
    seeded_    = false;
    prev_bin_  = -1;
    prev_nmi_  = -1.0;
    mi_.store(0.0,            std::memory_order_relaxed);
    nmi_.store(0.0,           std::memory_order_relaxed);
    change_events_.store(0,   std::memory_order_relaxed);
    total_.store(0,           std::memory_order_relaxed);
}

} // namespace llmquant
