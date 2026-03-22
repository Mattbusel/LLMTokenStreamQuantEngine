#include "MutualInformationEstimator.h"

#include <algorithm>
#include <cmath>
#include <numeric>
#include <sstream>

namespace llmquant {

MutualInformationEstimator::MutualInformationEstimator(Config config)
    : config_(std::move(config)) {
    int b = config_.bins;
    hist_.assign(static_cast<std::size_t>(b * b), 0);
    window_.resize(static_cast<std::size_t>(config_.window_size), {0, 0});
}

int MutualInformationEstimator::sentiment_bin(double v) const noexcept {
    int b = config_.bins;
    double lo = config_.sentiment_lo, hi = config_.sentiment_hi;
    if (v <= lo) return 0;
    if (v >= hi) return b - 1;
    return static_cast<int>((v - lo) / (hi - lo) * b);
}

int MutualInformationEstimator::return_bin(double v) const noexcept {
    int b = config_.bins;
    double lo = config_.return_lo, hi = config_.return_hi;
    if (v <= lo) return 0;
    if (v >= hi) return b - 1;
    return static_cast<int>((v - lo) / (hi - lo) * b);
}

void MutualInformationEstimator::recompute() {
    int n = window_count_;
    if (n < config_.min_samples) {
        mi_  = 0.0;
        nmi_ = 0.0;
        return;
    }

    int b = config_.bins;
    double inv_n = 1.0 / n;

    // Marginal counts.
    std::vector<int> marg_i(static_cast<std::size_t>(b), 0);
    std::vector<int> marg_j(static_cast<std::size_t>(b), 0);
    for (int i = 0; i < b; ++i)
        for (int j = 0; j < b; ++j) {
            int c = hist_[static_cast<std::size_t>(i * b + j)];
            marg_i[static_cast<std::size_t>(i)] += c;
            marg_j[static_cast<std::size_t>(j)] += c;
        }

    double mi = 0.0, hx = 0.0, hy = 0.0;
    for (int i = 0; i < b; ++i) {
        double pi = marg_i[static_cast<std::size_t>(i)] * inv_n;
        if (pi > 0.0) hx -= pi * std::log(pi);
        for (int j = 0; j < b; ++j) {
            double pj  = marg_j[static_cast<std::size_t>(j)] * inv_n;
            double pij = hist_[static_cast<std::size_t>(i * b + j)] * inv_n;
            if (pij > 0.0 && pi > 0.0 && pj > 0.0)
                mi += pij * std::log(pij / (pi * pj));
        }
    }
    for (int j = 0; j < b; ++j) {
        double pj = marg_j[static_cast<std::size_t>(j)] * inv_n;
        if (pj > 0.0) hy -= pj * std::log(pj);
    }

    mi_  = std::max(0.0, mi);
    double denom = std::min(hx, hy);
    nmi_ = (denom > 1e-14) ? mi_ / denom : 0.0;
    nmi_ = std::min(1.0, nmi_);
}

void MutualInformationEstimator::record(double sentiment, double ret) {
    total_records_.fetch_add(1, std::memory_order_relaxed);

    {
        std::lock_guard<std::mutex> lk(mutex_);

        int bi = sentiment_bin(sentiment);
        int bj = return_bin(ret);
        int b  = config_.bins;

        if (window_count_ < config_.window_size) {
            // Still filling.
            window_[static_cast<std::size_t>(window_head_)] = {bi, bj};
            hist_[static_cast<std::size_t>(bi * b + bj)]++;
            ++window_count_;
        } else {
            // Evict oldest entry.
            auto& old = window_[static_cast<std::size_t>(window_head_)];
            hist_[static_cast<std::size_t>(old.i * b + old.j)]--;
            // Insert new.
            window_[static_cast<std::size_t>(window_head_)] = {bi, bj};
            hist_[static_cast<std::size_t>(bi * b + bj)]++;
        }

        window_head_ = (window_head_ + 1) % config_.window_size;
        recompute();
    }
}

void MutualInformationEstimator::reset() {
    std::lock_guard<std::mutex> lk(mutex_);
    int b = config_.bins;
    hist_.assign(static_cast<std::size_t>(b * b), 0);
    std::fill(window_.begin(), window_.end(), Obs{0, 0});
    window_head_  = 0;
    window_count_ = 0;
    mi_  = 0.0;
    nmi_ = 0.0;
}

double MutualInformationEstimator::mi() const noexcept {
    std::lock_guard<std::mutex> lk(mutex_);
    return mi_;
}

double MutualInformationEstimator::normalized_mi() const noexcept {
    std::lock_guard<std::mutex> lk(mutex_);
    return nmi_;
}

int MutualInformationEstimator::sample_count() const noexcept {
    std::lock_guard<std::mutex> lk(mutex_);
    return window_count_;
}

void MutualInformationEstimator::update_config(const Config& config) {
    std::lock_guard<std::mutex> lk(mutex_);
    config_ = config;
    int b = config_.bins;
    hist_.assign(static_cast<std::size_t>(b * b), 0);
    window_.assign(static_cast<std::size_t>(config_.window_size), {0, 0});
    window_head_  = 0;
    window_count_ = 0;
    mi_  = 0.0;
    nmi_ = 0.0;
}

std::string MutualInformationEstimator::to_stats_json() const {
    std::lock_guard<std::mutex> lk(mutex_);
    std::ostringstream os;
    os << std::fixed;
    os << "{"
       << "\"mi\":"             << mi_ << ","
       << "\"normalized_mi\":"  << nmi_ << ","
       << "\"sample_count\":"   << window_count_ << ","
       << "\"bins\":"           << config_.bins << ","
       << "\"window_size\":"    << config_.window_size << ","
       << "\"total_records\":"  << total_records_.load(std::memory_order_relaxed)
       << "}";
    return os.str();
}

} // namespace llmquant
