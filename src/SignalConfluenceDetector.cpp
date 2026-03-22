#include "SignalConfluenceDetector.h"

#include <algorithm>
#include <cmath>
#include <sstream>
#include <iomanip>

namespace llmquant {

SignalConfluenceDetector::SignalConfluenceDetector(Config cfg)
    : cfg_(std::move(cfg)) {
    buf_.resize(static_cast<size_t>(std::max(cfg_.window, 2)), 0);
}

void SignalConfluenceDetector::record(double bias) {
    double fire_score = 0.0;
    int    fire_dir   = 0;
    bool   do_fire    = false;

    {
        std::lock_guard<std::mutex> lk(mutex_);

        obs_count_.fetch_add(1, std::memory_order_relaxed);

        int dir = 0;
        if (bias > 0.0)      dir = +1;
        else if (bias < 0.0) dir = -1;

        int sz = static_cast<int>(buf_.size());
        buf_[static_cast<size_t>(head_)] = dir;
        head_ = (head_ + 1) % sz;
        if (fill_ < sz) ++fill_;

        int bulls = 0, bears = 0;
        for (int i = 0; i < fill_; ++i) {
            if (buf_[static_cast<size_t>(i)] > 0)       ++bulls;
            else if (buf_[static_cast<size_t>(i)] < 0)  ++bears;
        }

        int dominant_count  = std::max(bulls, bears);
        int minority_count  = std::min(bulls, bears);
        int dominant_dir    = (bulls >= bears) ? (bulls > 0 ? +1 : 0) : -1;

        double score = 0.0;
        if (fill_ > 0) {
            score = static_cast<double>(dominant_count - minority_count) /
                    static_cast<double>(fill_);
            if (bears > bulls) score = -score;
        }

        score_.store(score, std::memory_order_relaxed);
        dominant_.store(dominant_dir, std::memory_order_relaxed);

        bool now_confluent = (std::abs(score) >= cfg_.threshold);
        confluent_.store(now_confluent, std::memory_order_relaxed);

        if (now_confluent && !prev_confluent_ && cfg_.on_confluence) {
            fire_score = score;
            fire_dir   = dominant_dir;
            do_fire    = true;
        }
        prev_confluent_ = now_confluent;
    }

    if (do_fire) cfg_.on_confluence(fire_score, fire_dir);
}

std::string SignalConfluenceDetector::to_stats_json() const {
    std::lock_guard<std::mutex> lk(mutex_);
    std::ostringstream oss;
    oss << std::fixed << std::setprecision(4);
    oss << "{\"confluence_score\":" << score_.load(std::memory_order_relaxed)
        << ",\"dominant_dir\":" << dominant_.load(std::memory_order_relaxed)
        << ",\"is_confluent\":" << (confluent_.load(std::memory_order_relaxed) ? "true" : "false")
        << ",\"obs_count\":" << obs_count_.load(std::memory_order_relaxed)
        << "}";
    return oss.str();
}

void SignalConfluenceDetector::reset() {
    std::lock_guard<std::mutex> lk(mutex_);
    std::fill(buf_.begin(), buf_.end(), 0);
    head_ = 0; fill_ = 0; prev_confluent_ = false;
    score_.store(0.0, std::memory_order_relaxed);
    dominant_.store(0, std::memory_order_relaxed);
    confluent_.store(false, std::memory_order_relaxed);
    obs_count_.store(0, std::memory_order_relaxed);
}

void SignalConfluenceDetector::update_config(const Config& cfg) {
    std::lock_guard<std::mutex> lk(mutex_);
    cfg_ = cfg;
    buf_.resize(static_cast<size_t>(std::max(cfg_.window, 2)), 0);
}

} // namespace llmquant
