#include "TokenImportanceDecayScheduler.h"

#include <algorithm>
#include <cmath>
#include <sstream>
#include <iomanip>

namespace llmquant {

TokenImportanceDecayScheduler::TokenImportanceDecayScheduler(Config cfg)
    : cfg_(std::move(cfg))
{}

void TokenImportanceDecayScheduler::record(const std::string& /*token*/,
                                            double weight,
                                            uint64_t timestamp_ns)
{
    std::function<void(double, double)> cb;
    bool   fire       = false;
    double old_sent   = 0.0;
    double new_sent   = 0.0;

    {
        std::lock_guard<std::mutex> lk(mutex_);
        cb = cfg_.on_sentiment_flip;

        evict_stale_locked(timestamp_ns);

        // Cap window size.
        if (static_cast<int>(window_.size()) >= cfg_.max_entries) {
            window_.erase(window_.begin());  // evict oldest
        }

        window_.push_back({weight, timestamp_ns});
        active_entries_.store(window_.size(), std::memory_order_relaxed);
        total_recorded_.fetch_add(1, std::memory_order_relaxed);

        // Compute current effective sentiment.
        double sum_w = 0.0, sum_abs = 0.0;
        double lambda = std::log(2.0) / std::max(cfg_.half_life_s, 1e-9);
        for (auto& e : window_) {
            double age_s = static_cast<double>(timestamp_ns - e.timestamp_ns) * 1e-9;
            double decay  = std::exp(-lambda * age_s);
            sum_w   += e.weight * decay;
            sum_abs += std::abs(e.weight) * decay;
        }
        double sent = (sum_abs > 0.0) ? (sum_w / sum_abs) : 0.0;

        if (sentiment_seeded_) {
            if ((prev_sentiment_ > 0.0) != (sent > 0.0) && std::abs(sent) > 1e-9) {
                old_sent = prev_sentiment_;
                new_sent = sent;
                fire     = true;
                flip_count_.fetch_add(1, std::memory_order_relaxed);
            }
        }
        prev_sentiment_   = sent;
        sentiment_seeded_ = true;
    }

    if (fire && cb) cb(old_sent, new_sent);
}

void TokenImportanceDecayScheduler::evict_stale_locked(uint64_t now_ns) {
    double max_age_ns = cfg_.max_age_s * 1e9;
    window_.erase(
        std::remove_if(window_.begin(), window_.end(),
            [&](const Entry& e) {
                return static_cast<double>(now_ns - e.timestamp_ns) > max_age_ns;
            }),
        window_.end());
}

void TokenImportanceDecayScheduler::evict_stale(uint64_t now_ns) {
    std::lock_guard<std::mutex> lk(mutex_);
    evict_stale_locked(now_ns);
    active_entries_.store(window_.size(), std::memory_order_relaxed);
}

double TokenImportanceDecayScheduler::effective_sentiment(uint64_t query_ns) const {
    std::lock_guard<std::mutex> lk(mutex_);
    if (window_.empty()) return 0.0;
    double sum_w = 0.0, sum_abs = 0.0;
    double lambda = std::log(2.0) / std::max(cfg_.half_life_s, 1e-9);
    for (auto& e : window_) {
        double age_s = static_cast<double>(query_ns - e.timestamp_ns) * 1e-9;
        if (age_s < 0.0) age_s = 0.0;
        double decay  = std::exp(-lambda * age_s);
        sum_w   += e.weight * decay;
        sum_abs += std::abs(e.weight) * decay;
    }
    return (sum_abs > 0.0) ? (sum_w / sum_abs) : 0.0;
}

std::string TokenImportanceDecayScheduler::to_stats_json(uint64_t query_ns) const {
    double sent   = effective_sentiment(query_ns);
    std::size_t ae;
    uint64_t tr, fc;
    {
        std::lock_guard<std::mutex> lk(mutex_);
        ae = active_entries_.load(std::memory_order_relaxed);
        tr = total_recorded_.load(std::memory_order_relaxed);
        fc = flip_count_.load(std::memory_order_relaxed);
    }
    std::ostringstream ss;
    ss << std::fixed << std::setprecision(6);
    ss << "{"
       << "\"effective_sentiment\":" << sent << ","
       << "\"active_entries\":"      << ae   << ","
       << "\"total_recorded\":"      << tr   << ","
       << "\"flip_count\":"          << fc
       << "}";
    return ss.str();
}

void TokenImportanceDecayScheduler::reset() {
    std::lock_guard<std::mutex> lk(mutex_);
    window_.clear();
    prev_sentiment_   = 0.0;
    sentiment_seeded_ = false;
    active_entries_.store(0, std::memory_order_relaxed);
    total_recorded_.store(0, std::memory_order_relaxed);
    flip_count_.store(0, std::memory_order_relaxed);
}

void TokenImportanceDecayScheduler::update_config(const Config& cfg) {
    std::lock_guard<std::mutex> lk(mutex_);
    cfg_ = cfg;
    window_.clear();
    prev_sentiment_   = 0.0;
    sentiment_seeded_ = false;
    active_entries_.store(0, std::memory_order_relaxed);
}

// Private helper declared in header but must be defined here.
// (evict_stale_locked is called from within mutex-held context)

} // namespace llmquant
