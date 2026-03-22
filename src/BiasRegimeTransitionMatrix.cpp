#include "BiasRegimeTransitionMatrix.h"

#include <algorithm>
#include <cmath>
#include <sstream>
#include <iomanip>

namespace llmquant {

BiasRegimeTransitionMatrix::BiasRegimeTransitionMatrix(Config cfg)
    : cfg_(std::move(cfg))
{
    int n = std::min(std::max(2, cfg_.n_regimes), kMaxRegimes);
    int w = std::max(2, cfg_.window);
    win_buf_.assign(static_cast<std::size_t>(w), Transition{});
    counts_.assign(static_cast<std::size_t>(n * n), 0);
}

int BiasRegimeTransitionMatrix::bucket_for(double bias) const {
    int n   = std::min(std::max(2, cfg_.n_regimes), kMaxRegimes);
    double rng = cfg_.range_max - cfg_.range_min;
    if (rng <= 0.0) return 0;
    double frac = (bias - cfg_.range_min) / rng;
    frac = std::max(0.0, std::min(1.0 - 1e-12, frac));
    return static_cast<int>(frac * n);
}

void BiasRegimeTransitionMatrix::recompute_transition_locked() {
    int n = std::min(std::max(2, cfg_.n_regimes), kMaxRegimes);
    int cur = current_regime_.load(std::memory_order_relaxed);
    if (cur < 0 || cur >= n) return;

    // Find most likely next regime from current row
    int    best_j   = 0;
    double best_p   = 0.0;
    double row_sum  = 0.0;
    for (int j = 0; j < n; ++j)
        row_sum += counts_[static_cast<std::size_t>(cur * n + j)];

    if (row_sum > 0.0) {
        for (int j = 0; j < n; ++j) {
            double p = counts_[static_cast<std::size_t>(cur * n + j)] / row_sum;
            if (p > best_p) { best_p = p; best_j = j; }
        }
    }
    most_likely_next_.store(best_j, std::memory_order_relaxed);
    most_likely_prob_.store(best_p, std::memory_order_relaxed);
}

void BiasRegimeTransitionMatrix::record(double bias) {
    bool   fire      = false;
    int    from_r    = 0, to_r = 0;
    double prob      = 0.0;
    std::function<void(int,int,double)> cb;

    {
        std::lock_guard<std::mutex> lk(mutex_);
        cb = cfg_.on_likely_transition;
        total_.fetch_add(1, std::memory_order_relaxed);

        int n   = std::min(std::max(2, cfg_.n_regimes), kMaxRegimes);
        int bkt = bucket_for(bias);
        current_regime_.store(bkt, std::memory_order_relaxed);

        if (prev_regime_ >= 0) {
            int w = static_cast<int>(win_buf_.size());

            // Evict oldest transition
            if (win_fill_ == w) {
                Transition old = win_buf_[static_cast<std::size_t>(win_head_)];
                int& ct = counts_[static_cast<std::size_t>(old.from * n + old.to)];
                if (ct > 0) --ct;
            } else {
                ++win_fill_;
            }

            // Insert new transition
            Transition t{prev_regime_, bkt};
            win_buf_[static_cast<std::size_t>(win_head_)] = t;
            win_head_ = (win_head_ + 1) % w;
            ++counts_[static_cast<std::size_t>(prev_regime_ * n + bkt)];

            recompute_transition_locked();

            int    ml = most_likely_next_.load(std::memory_order_relaxed);
            double mp = most_likely_prob_.load(std::memory_order_relaxed);
            if (mp > cfg_.alert_prob && ml != prev_most_likely_) {
                fire    = true;
                from_r  = bkt;
                to_r    = ml;
                prob    = mp;
                prev_most_likely_ = ml;
            }
        }

        prev_regime_ = bkt;
    }

    if (fire && cb) cb(from_r, to_r, prob);
}

double BiasRegimeTransitionMatrix::transition_prob(int from_r, int to_r) const {
    std::lock_guard<std::mutex> lk(mutex_);
    int n = std::min(std::max(2, cfg_.n_regimes), kMaxRegimes);
    if (from_r < 0 || from_r >= n || to_r < 0 || to_r >= n) return 0.0;

    double row_sum = 0.0;
    for (int j = 0; j < n; ++j)
        row_sum += counts_[static_cast<std::size_t>(from_r * n + j)];
    if (row_sum <= 0.0) return 0.0;
    return counts_[static_cast<std::size_t>(from_r * n + to_r)] / row_sum;
}

std::string BiasRegimeTransitionMatrix::to_stats_json() const {
    std::lock_guard<std::mutex> lk(mutex_);
    std::ostringstream ss;
    ss << std::fixed << std::setprecision(6);
    ss << "{"
       << "\"current_regime\":"   << current_regime_.load(std::memory_order_relaxed) << ","
       << "\"most_likely_next\":" << most_likely_next_.load(std::memory_order_relaxed) << ","
       << "\"most_likely_prob\":" << most_likely_prob_.load(std::memory_order_relaxed) << ","
       << "\"total_records\":"    << total_.load(std::memory_order_relaxed)
       << "}";
    return ss.str();
}

void BiasRegimeTransitionMatrix::reset() {
    std::lock_guard<std::mutex> lk(mutex_);
    std::fill(win_buf_.begin(), win_buf_.end(), Transition{});
    std::fill(counts_.begin(), counts_.end(), 0);
    win_head_ = win_fill_ = 0;
    prev_regime_ = -1;
    prev_most_likely_ = -1;
    current_regime_.store(0, std::memory_order_relaxed);
    most_likely_next_.store(0, std::memory_order_relaxed);
    most_likely_prob_.store(0.0, std::memory_order_relaxed);
    total_.store(0, std::memory_order_relaxed);
}

void BiasRegimeTransitionMatrix::update_config(const Config& cfg) {
    std::lock_guard<std::mutex> lk(mutex_);
    cfg_ = cfg;
    int n = std::min(std::max(2, cfg_.n_regimes), kMaxRegimes);
    int w = std::max(2, cfg_.window);
    win_buf_.assign(static_cast<std::size_t>(w), Transition{});
    counts_.assign(static_cast<std::size_t>(n * n), 0);
    win_head_ = win_fill_ = 0;
    prev_regime_ = -1;
    prev_most_likely_ = -1;
}

} // namespace llmquant
