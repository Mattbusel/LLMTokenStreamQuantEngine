#include "SignalSpectralEntropy.h"

#include <algorithm>
#include <cmath>
#include <sstream>
#include <iomanip>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

namespace llmquant {

SignalSpectralEntropy::SignalSpectralEntropy(Config cfg)
    : cfg_(std::move(cfg))
{
    int w = std::max(4, cfg_.window);
    win_buf_.assign(static_cast<std::size_t>(w), 0.0);
}

void SignalSpectralEntropy::compute_spectral_entropy_locked() {
    int n = win_fill_;
    if (n < std::max(2, cfg_.min_samples)) return;

    int w = static_cast<int>(win_buf_.size());

    // Build chronological snapshot
    std::vector<double> x(static_cast<std::size_t>(n));
    for (int i = 0; i < n; ++i) {
        int idx = (win_head_ - n + i + w * 2) % w;
        x[static_cast<std::size_t>(i)] = win_buf_[static_cast<std::size_t>(idx)];
    }

    // DCT-II: X[k] = Σ_{j=0}^{N-1} x[j] cos(π/N (j+0.5) k)
    int N = n;
    std::vector<double> X(static_cast<std::size_t>(N), 0.0);
    for (int k = 0; k < N; ++k) {
        double sum = 0.0;
        for (int j = 0; j < N; ++j)
            sum += x[static_cast<std::size_t>(j)] *
                   std::cos(M_PI / N * (j + 0.5) * k);
        X[static_cast<std::size_t>(k)] = sum;
    }

    // Power spectrum |X[k]|²; skip DC (k=0)
    double total_power = 0.0;
    for (int k = 1; k < N; ++k)
        total_power += X[static_cast<std::size_t>(k)] * X[static_cast<std::size_t>(k)];

    double Hs = 0.0;
    constexpr double eps = 1e-14;
    if (total_power > eps) {
        for (int k = 1; k < N; ++k) {
            double p = X[static_cast<std::size_t>(k)] * X[static_cast<std::size_t>(k)] / total_power;
            if (p > eps) Hs -= p * std::log2(p);
        }
    }

    double max_Hs = (N > 1) ? std::log2(static_cast<double>(N - 1)) : 1.0;
    double Hs_norm = (max_Hs > eps) ? std::min(1.0, Hs / max_Hs) : 0.0;

    hs_.store(Hs,      std::memory_order_relaxed);
    hs_norm_.store(Hs_norm, std::memory_order_relaxed);
}

void SignalSpectralEntropy::record(double bias) {
    bool   fire   = false;
    double old_hs = 0.0, new_hs = 0.0;
    std::function<void(double, double)> cb;

    {
        std::lock_guard<std::mutex> lk(mutex_);
        cb = cfg_.on_entropy_change;
        total_.fetch_add(1, std::memory_order_relaxed);

        int w = static_cast<int>(win_buf_.size());
        if (win_fill_ < w) ++win_fill_;
        win_buf_[static_cast<std::size_t>(win_head_)] = bias;
        win_head_ = (win_head_ + 1) % w;

        compute_spectral_entropy_locked();

        double cur_hs = hs_.load(std::memory_order_relaxed);
        if (prev_hs_ >= 0.0 && std::abs(cur_hs - prev_hs_) > cfg_.change_threshold) {
            old_hs = prev_hs_;
            new_hs = cur_hs;
            fire   = true;
            change_events_.fetch_add(1, std::memory_order_relaxed);
        }
        prev_hs_ = cur_hs;
    }

    if (fire && cb) cb(old_hs, new_hs);
}

std::string SignalSpectralEntropy::to_stats_json() const {
    std::lock_guard<std::mutex> lk(mutex_);
    std::ostringstream ss;
    ss << std::fixed << std::setprecision(6);
    ss << "{"
       << "\"spectral_entropy\":"     << hs_.load(std::memory_order_relaxed) << ","
       << "\"normalised_entropy\":"   << hs_norm_.load(std::memory_order_relaxed) << ","
       << "\"change_events\":"        << change_events_.load(std::memory_order_relaxed) << ","
       << "\"total_records\":"        << total_.load(std::memory_order_relaxed)
       << "}";
    return ss.str();
}

void SignalSpectralEntropy::reset() {
    std::lock_guard<std::mutex> lk(mutex_);
    std::fill(win_buf_.begin(), win_buf_.end(), 0.0);
    win_head_ = win_fill_ = 0;
    prev_hs_  = -1.0;
    hs_.store(0.0,    std::memory_order_relaxed);
    hs_norm_.store(0.0, std::memory_order_relaxed);
    change_events_.store(0, std::memory_order_relaxed);
    total_.store(0,      std::memory_order_relaxed);
}

void SignalSpectralEntropy::update_config(const Config& cfg) {
    std::lock_guard<std::mutex> lk(mutex_);
    cfg_ = cfg;
    int w = std::max(4, cfg_.window);
    win_buf_.assign(static_cast<std::size_t>(w), 0.0);
    win_head_ = win_fill_ = 0;
    prev_hs_  = -1.0;
}

} // namespace llmquant
