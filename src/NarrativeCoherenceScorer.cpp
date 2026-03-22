#include "NarrativeCoherenceScorer.h"

#include <algorithm>
#include <cmath>
#include <numeric>
#include <sstream>
#include <iomanip>

namespace llmquant {

NarrativeCoherenceScorer::NarrativeCoherenceScorer(Config cfg)
    : cfg_(std::move(cfg))
{
    int w = std::max(2, cfg_.window);
    buf_.assign(static_cast<std::size_t>(w), 0.0);
}

void NarrativeCoherenceScorer::record(double bias) {
    bool   fire_drop    = false;
    bool   fire_recover = false;
    double score        = 0.0;
    std::function<void(double)> drop_cb, rec_cb;

    {
        std::lock_guard<std::mutex> lk(mutex_);
        drop_cb = cfg_.on_coherence_drop;
        rec_cb  = cfg_.on_coherence_recover;
        total_.fetch_add(1, std::memory_order_relaxed);

        int w = static_cast<int>(buf_.size());
        buf_[static_cast<std::size_t>(head_)] = bias;
        head_ = (head_ + 1) % w;
        if (fill_ < w) ++fill_;

        if (fill_ < std::max(2, cfg_.min_samples)) {
            return;
        }

        // Compute mean and variance over filled portion.
        double sum = 0.0, sum2 = 0.0;
        for (int i = 0; i < fill_; ++i) {
            int idx = (head_ - 1 - i + w) % w;
            double v = buf_[static_cast<std::size_t>(idx)];
            sum  += v;
            sum2 += v * v;
        }
        double mu  = sum / fill_;
        double var = sum2 / fill_ - mu * mu;
        if (var < 0.0) var = 0.0;
        double sigma = std::sqrt(var);

        score = std::abs(mu) / (sigma + cfg_.epsilon);

        mean_.store(mu,    std::memory_order_relaxed);
        stddev_.store(sigma, std::memory_order_relaxed);
        coherence_.store(score, std::memory_order_relaxed);

        bool now_incoherent = score < cfg_.low_threshold;
        incoherent_.store(now_incoherent, std::memory_order_relaxed);

        if (now_incoherent && !prev_incoherent_) fire_drop = true;
        if (!now_incoherent && prev_incoherent_ && score >= cfg_.recover_threshold)
            fire_recover = true;
        prev_incoherent_ = now_incoherent;
    }

    if (fire_drop    && drop_cb) drop_cb(score);
    if (fire_recover && rec_cb)  rec_cb(score);
}

std::string NarrativeCoherenceScorer::to_stats_json() const {
    std::lock_guard<std::mutex> lk(mutex_);
    std::ostringstream ss;
    ss << std::fixed << std::setprecision(6);
    ss << "{"
       << "\"coherence\":"      << coherence_.load(std::memory_order_relaxed) << ","
       << "\"rolling_mean\":"   << mean_.load(std::memory_order_relaxed) << ","
       << "\"rolling_stddev\":" << stddev_.load(std::memory_order_relaxed) << ","
       << "\"is_incoherent\":"  << (incoherent_.load(std::memory_order_relaxed) ? "true" : "false") << ","
       << "\"total_records\":"  << total_.load(std::memory_order_relaxed)
       << "}";
    return ss.str();
}

void NarrativeCoherenceScorer::reset() {
    std::lock_guard<std::mutex> lk(mutex_);
    std::fill(buf_.begin(), buf_.end(), 0.0);
    head_ = fill_ = 0;
    prev_incoherent_ = false;
    coherence_.store(0.0, std::memory_order_relaxed);
    mean_.store(0.0, std::memory_order_relaxed);
    stddev_.store(0.0, std::memory_order_relaxed);
    incoherent_.store(false, std::memory_order_relaxed);
    total_.store(0, std::memory_order_relaxed);
}

void NarrativeCoherenceScorer::update_config(const Config& cfg) {
    std::lock_guard<std::mutex> lk(mutex_);
    cfg_ = cfg;
    int w = std::max(2, cfg_.window);
    buf_.assign(static_cast<std::size_t>(w), 0.0);
    head_ = fill_ = 0;
    prev_incoherent_ = false;
}

} // namespace llmquant
