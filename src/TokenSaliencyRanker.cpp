#include "TokenSaliencyRanker.h"
#include <algorithm>
#include <cmath>
#include <sstream>
#include <iomanip>

namespace llmquant {

TokenSaliencyRanker::TokenSaliencyRanker(Config cfg)
    : cfg_(std::move(cfg))
    , stats_(kMaxTokens)
{}

void TokenSaliencyRanker::record(int token_id, double bias_contribution) {
    if (token_id < 0 || token_id >= kMaxTokens) return;

    bool changed = false;
    int  new_top = -1;
    double new_top_sal = 0.0;

    {
        std::lock_guard<std::mutex> lk(mutex_);
        auto& s = stats_[token_id];

        double a = cfg_.ema_alpha;
        double dev = bias_contribution - s.mean;
        s.mean += a * dev;
        s.var   = s.var * (1.0 - a) + dev * dev * a;
        ++s.count;

        // Find top saliency token (linear scan — bounded by kMaxTokens=512)
        int    best_id  = -1;
        double best_sal = -1e18;
        for (int i = 0; i < kMaxTokens; ++i) {
            if (stats_[i].count < cfg_.min_token_count) continue;
            double sal = stats_[i].mean / (std::sqrt(stats_[i].var) + 1e-8);
            if (sal > best_sal) { best_sal = sal; best_id = i; }
        }
        if (best_id >= 0 && best_id != prev_top_) {
            changed   = true;
            new_top   = best_id;
            new_top_sal = best_sal;
            prev_top_ = best_id;
        }
    }

    total_.fetch_add(1, std::memory_order_relaxed);
    if (changed && cfg_.on_top_token_change) cfg_.on_top_token_change(new_top, new_top_sal);
}

double TokenSaliencyRanker::saliency(int token_id) const {
    if (token_id < 0 || token_id >= kMaxTokens) return 0.0;
    std::lock_guard<std::mutex> lk(mutex_);
    const auto& s = stats_[token_id];
    if (s.count < cfg_.min_token_count) return 0.0;
    return s.mean / (std::sqrt(s.var) + 1e-8);
}

std::array<int, TokenSaliencyRanker::kTopK> TokenSaliencyRanker::top_bullish() const {
    std::lock_guard<std::mutex> lk(mutex_);
    std::array<int, kTopK> result{-1, -1, -1, -1, -1};
    std::array<double, kTopK> best_sal{};
    best_sal.fill(-1e18);
    for (int i = 0; i < kMaxTokens; ++i) {
        if (stats_[i].count < cfg_.min_token_count) continue;
        double sal = stats_[i].mean / (std::sqrt(stats_[i].var) + 1e-8);
        // insertion into top-K
        for (int k = 0; k < kTopK; ++k) {
            if (sal > best_sal[k]) {
                for (int j = kTopK - 1; j > k; --j) {
                    best_sal[j] = best_sal[j-1];
                    result[j]   = result[j-1];
                }
                best_sal[k] = sal;
                result[k]   = i;
                break;
            }
        }
    }
    return result;
}

std::array<int, TokenSaliencyRanker::kTopK> TokenSaliencyRanker::top_bearish() const {
    std::lock_guard<std::mutex> lk(mutex_);
    std::array<int, kTopK> result{-1, -1, -1, -1, -1};
    std::array<double, kTopK> worst_sal{};
    worst_sal.fill(1e18);
    for (int i = 0; i < kMaxTokens; ++i) {
        if (stats_[i].count < cfg_.min_token_count) continue;
        double sal = stats_[i].mean / (std::sqrt(stats_[i].var) + 1e-8);
        for (int k = 0; k < kTopK; ++k) {
            if (sal < worst_sal[k]) {
                for (int j = kTopK - 1; j > k; --j) {
                    worst_sal[j] = worst_sal[j-1];
                    result[j]    = result[j-1];
                }
                worst_sal[k] = sal;
                result[k]    = i;
                break;
            }
        }
    }
    return result;
}

std::string TokenSaliencyRanker::to_stats_json() const {
    std::lock_guard<std::mutex> lk(mutex_);
    auto top_bull = top_bullish();
    auto top_bear = top_bearish();
    std::ostringstream ss;
    ss << std::fixed << std::setprecision(4);
    ss << "{\"top_bullish\":[" << top_bull[0] << "," << top_bull[1] << "," << top_bull[2] << "]"
       << ",\"top_bearish\":[" << top_bear[0] << "," << top_bear[1] << "," << top_bear[2] << "]"
       << ",\"total_records\":" << total_.load(std::memory_order_relaxed)
       << "}";
    return ss.str();
}

void TokenSaliencyRanker::reset() {
    std::lock_guard<std::mutex> lk(mutex_);
    stats_.assign(kMaxTokens, TokenStats{});
    prev_top_ = -1;
    total_.store(0, std::memory_order_relaxed);
}

void TokenSaliencyRanker::update_config(const Config& cfg) {
    std::lock_guard<std::mutex> lk(mutex_);
    cfg_ = cfg;
}

} // namespace llmquant
