#include "NarrativeConsistencyMeter.h"

#include <algorithm>
#include <cmath>
#include <iomanip>
#include <sstream>

namespace llmquant {

NarrativeConsistencyMeter::NarrativeConsistencyMeter(Config cfg)
    : cfg_(std::move(cfg))
{
    int w = std::max(4, cfg_.window);
    delta_buf_.resize(static_cast<std::size_t>(w), 0.0);
}

void NarrativeConsistencyMeter::record(double bias) {
    std::function<void(double)> incon_cb, recov_cb;
    bool fire_incon = false, fire_recov = false;
    double score_val = 1.0;

    {
        std::lock_guard<std::mutex> lk(mutex_);
        incon_cb = cfg_.on_inconsistent;
        recov_cb = cfg_.on_recovered;

        total_.fetch_add(1, std::memory_order_relaxed);

        if (!seeded_) {
            prev_bias_ = bias;
            seeded_    = true;
        } else {
            double delta = std::abs(bias - prev_bias_);
            prev_bias_   = bias;

            double old_val = delta_buf_[static_cast<std::size_t>(head_)];
            delta_sum_ -= old_val;
            delta_sum_ += delta;
            delta_buf_[static_cast<std::size_t>(head_)] = delta;
            head_ = (head_ + 1) % static_cast<int>(delta_buf_.size());
            if (fill_ < static_cast<int>(delta_buf_.size())) ++fill_;

            int min_obs = std::max(2, cfg_.min_samples);
            if (fill_ >= min_obs) {
                double aad_val = delta_sum_ / fill_;
                aad_val = std::max(0.0, aad_val);

                double norm = cfg_.normalizer > 1e-12 ? cfg_.normalizer : 1e-12;
                score_val = std::max(0.0, 1.0 - aad_val / norm);

                aad_.store(aad_val,    std::memory_order_relaxed);
                score_.store(score_val, std::memory_order_relaxed);

                bool now_incon = score_val < cfg_.consistency_threshold;
                inconsistent_.store(now_incon, std::memory_order_relaxed);

                if (now_incon && !prev_inconsistent_) {
                    fire_incon = true;
                    events_.fetch_add(1, std::memory_order_relaxed);
                } else if (!now_incon && prev_inconsistent_ &&
                           score_val > cfg_.consistency_threshold * cfg_.clear_hysteresis) {
                    fire_recov = true;
                }
                prev_inconsistent_ = now_incon;
            }
        }
    }

    if (fire_incon && incon_cb) incon_cb(score_val);
    if (fire_recov && recov_cb) recov_cb(score_val);
}

std::string NarrativeConsistencyMeter::to_stats_json() const {
    double score, aad_val;
    bool   incon;
    uint64_t events, total;
    {
        std::lock_guard<std::mutex> lk(mutex_);
        score   = score_.load(std::memory_order_relaxed);
        aad_val = aad_.load(std::memory_order_relaxed);
        incon   = inconsistent_.load(std::memory_order_relaxed);
        events  = events_.load(std::memory_order_relaxed);
        total   = total_.load(std::memory_order_relaxed);
    }
    std::ostringstream ss;
    ss << std::fixed << std::setprecision(6);
    ss << "{"
       << "\"consistency_score\":" << score   << ","
       << "\"aad\":"               << aad_val << ","
       << "\"is_inconsistent\":"   << (incon ? "true" : "false") << ","
       << "\"inconsistency_events\":" << events << ","
       << "\"total_records\":"     << total
       << "}";
    return ss.str();
}

void NarrativeConsistencyMeter::reset() {
    std::lock_guard<std::mutex> lk(mutex_);
    std::fill(delta_buf_.begin(), delta_buf_.end(), 0.0);
    head_ = fill_ = 0;
    delta_sum_         = 0.0;
    prev_bias_         = 0.0;
    seeded_            = false;
    prev_inconsistent_ = false;
    score_.store(1.0,   std::memory_order_relaxed);
    aad_.store(0.0,     std::memory_order_relaxed);
    inconsistent_.store(false, std::memory_order_relaxed);
    events_.store(0,    std::memory_order_relaxed);
    total_.store(0,     std::memory_order_relaxed);
}

void NarrativeConsistencyMeter::update_config(const Config& cfg) {
    std::lock_guard<std::mutex> lk(mutex_);
    cfg_ = cfg;
    int w = std::max(4, cfg_.window);
    delta_buf_.assign(static_cast<std::size_t>(w), 0.0);
    head_ = fill_ = 0;
    delta_sum_         = 0.0;
    prev_bias_         = 0.0;
    seeded_            = false;
    prev_inconsistent_ = false;
    score_.store(1.0,   std::memory_order_relaxed);
    aad_.store(0.0,     std::memory_order_relaxed);
    inconsistent_.store(false, std::memory_order_relaxed);
    events_.store(0,    std::memory_order_relaxed);
    total_.store(0,     std::memory_order_relaxed);
}

} // namespace llmquant
