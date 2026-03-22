#include "TokenSentimentGraphBuilder.h"

#include <algorithm>
#include <sstream>
#include <iomanip>
#include <numeric>

namespace llmquant {

TokenSentimentGraphBuilder::TokenSentimentGraphBuilder(Config cfg)
    : cfg_(std::move(cfg))
{
    int w = std::max(4, cfg_.window);
    ring_.assign(static_cast<std::size_t>(w), {-1, 0.0});
    edge_sum_.assign(static_cast<std::size_t>(kMaxTokens * kMaxTokens), 0.0);
    edge_count_.assign(static_cast<std::size_t>(kMaxTokens * kMaxTokens), 0);
}

void TokenSentimentGraphBuilder::record(int token_id, double bias) {
    if (token_id < 0 || token_id >= kMaxTokens) return;

    bool   fire       = false;
    int    hub_id     = -1;
    double hub_weight = 0.0;
    std::function<void(int, double)> cb;

    {
        std::lock_guard<std::mutex> lk(mutex_);
        cb = cfg_.on_hub_detected;
        total_.fetch_add(1, std::memory_order_relaxed);

        int w = static_cast<int>(ring_.size());

        // Record directed edge prev→token_id
        if (prev_token_ >= 0) {
            int edge = prev_token_ * kMaxTokens + token_id;

            // Evict oldest entry if window full
            if (ring_fill_ == w) {
                Entry old = ring_[static_cast<std::size_t>(ring_head_)];
                if (old.token_id >= 0 && prev_token_ >= 0) {
                    // We need the from-token for the evicted pair, stored in ring
                    // We track full (from, to, bias) in ring for correct eviction
                    // Simplified: store the edge index and bias in ring
                }
                // Evict: we stored {to_token_id, bias}, but we need from_token too.
                // Redesign: store edge index + bias directly in ring.
                (void)old;  // placeholder — edge eviction handled below
            }

            // Simpler ring eviction: store edge index in ring as token_id field
            if (ring_fill_ == w) {
                Entry& old = ring_[static_cast<std::size_t>(ring_head_)];
                // old.token_id encodes (from*kMaxTokens + to) packed
                if (old.token_id >= 0) {
                    int eidx = old.token_id;
                    if (edge_count_[static_cast<std::size_t>(eidx)] > 0) {
                        --edge_count_[static_cast<std::size_t>(eidx)];
                        edge_sum_[static_cast<std::size_t>(eidx)] -= old.bias;
                    }
                }
            } else {
                ++ring_fill_;
            }
            ring_[static_cast<std::size_t>(ring_head_)] = {edge, bias};
            ring_head_ = (ring_head_ + 1) % w;

            edge_sum_[static_cast<std::size_t>(edge)]   += bias;
            edge_count_[static_cast<std::size_t>(edge)] += 1;

            // Check if token_id is now a hub (high in-degree weight)
            double in_w = 0.0;
            for (int from = 0; from < kMaxTokens; ++from) {
                int e   = from * kMaxTokens + token_id;
                int cnt = edge_count_[static_cast<std::size_t>(e)];
                if (cnt >= cfg_.min_edge_count)
                    in_w += edge_sum_[static_cast<std::size_t>(e)] /
                            static_cast<double>(cnt);
            }
            if (in_w > cfg_.hub_threshold) {
                hub_id     = token_id;
                hub_weight = in_w;
                fire       = true;
            }
        }

        prev_token_ = token_id;
    }

    if (fire && cb) cb(hub_id, hub_weight);
}

double TokenSentimentGraphBuilder::edge_weight(int from, int to) const {
    if (from < 0 || from >= kMaxTokens || to < 0 || to >= kMaxTokens) return 0.0;
    std::lock_guard<std::mutex> lk(mutex_);
    int e   = from * kMaxTokens + to;
    int cnt = edge_count_[static_cast<std::size_t>(e)];
    if (cnt < cfg_.min_edge_count) return 0.0;
    return edge_sum_[static_cast<std::size_t>(e)] / static_cast<double>(cnt);
}

double TokenSentimentGraphBuilder::in_degree_weight(int token_id) const {
    if (token_id < 0 || token_id >= kMaxTokens) return 0.0;
    std::lock_guard<std::mutex> lk(mutex_);
    double in_w = 0.0;
    for (int from = 0; from < kMaxTokens; ++from) {
        int e   = from * kMaxTokens + token_id;
        int cnt = edge_count_[static_cast<std::size_t>(e)];
        if (cnt >= cfg_.min_edge_count)
            in_w += edge_sum_[static_cast<std::size_t>(e)] / static_cast<double>(cnt);
    }
    return in_w;
}

std::array<int, TokenSentimentGraphBuilder::kTopK>
TokenSentimentGraphBuilder::top_hubs() const {
    std::lock_guard<std::mutex> lk(mutex_);
    std::array<double, kMaxTokens> in_weights{};
    for (int t = 0; t < kMaxTokens; ++t) {
        double w = 0.0;
        for (int from = 0; from < kMaxTokens; ++from) {
            int e   = from * kMaxTokens + t;
            int cnt = edge_count_[static_cast<std::size_t>(e)];
            if (cnt >= cfg_.min_edge_count)
                w += edge_sum_[static_cast<std::size_t>(e)] / static_cast<double>(cnt);
        }
        in_weights[static_cast<std::size_t>(t)] = w;
    }

    std::array<int, kTopK> result{};
    result.fill(-1);
    std::array<bool, kMaxTokens> used{};
    used.fill(false);
    for (int k = 0; k < kTopK; ++k) {
        int best = -1;
        double best_w = -1e18;
        for (int t = 0; t < kMaxTokens; ++t) {
            if (!used[static_cast<std::size_t>(t)] &&
                in_weights[static_cast<std::size_t>(t)] > best_w) {
                best_w = in_weights[static_cast<std::size_t>(t)];
                best   = t;
            }
        }
        if (best >= 0) {
            result[static_cast<std::size_t>(k)] = best;
            used[static_cast<std::size_t>(best)] = true;
        }
    }
    return result;
}

std::string TokenSentimentGraphBuilder::to_stats_json() const {
    std::lock_guard<std::mutex> lk(mutex_);
    // Count non-zero edges
    int edge_cnt = 0;
    for (int e = 0; e < kMaxTokens * kMaxTokens; ++e)
        if (edge_count_[static_cast<std::size_t>(e)] >= cfg_.min_edge_count) ++edge_cnt;

    std::ostringstream ss;
    ss << std::fixed << std::setprecision(6);
    ss << "{"
       << "\"active_edges\":"  << edge_cnt << ","
       << "\"total_records\":" << total_.load(std::memory_order_relaxed)
       << "}";
    return ss.str();
}

void TokenSentimentGraphBuilder::reset() {
    std::lock_guard<std::mutex> lk(mutex_);
    std::fill(edge_sum_.begin(),   edge_sum_.end(),   0.0);
    std::fill(edge_count_.begin(), edge_count_.end(), 0);
    std::fill(ring_.begin(),       ring_.end(),       Entry{-1, 0.0});
    ring_head_ = ring_fill_ = 0;
    prev_token_ = -1;
    total_.store(0, std::memory_order_relaxed);
}

void TokenSentimentGraphBuilder::update_config(const Config& cfg) {
    std::lock_guard<std::mutex> lk(mutex_);
    cfg_ = cfg;
    int w = std::max(4, cfg_.window);
    ring_.assign(static_cast<std::size_t>(w), {-1, 0.0});
    std::fill(edge_sum_.begin(),   edge_sum_.end(),   0.0);
    std::fill(edge_count_.begin(), edge_count_.end(), 0);
    ring_head_ = ring_fill_ = 0;
    prev_token_ = -1;
    total_.store(0, std::memory_order_relaxed);
}

} // namespace llmquant
