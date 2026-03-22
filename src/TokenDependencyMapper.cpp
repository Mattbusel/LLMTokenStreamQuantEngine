#include "TokenDependencyMapper.h"

#include <algorithm>
#include <sstream>
#include <iomanip>

namespace llmquant {

static uint64_t make_key(int a, int b) {
    int lo = std::min(a, b);
    int hi = std::max(a, b);
    return static_cast<uint64_t>(lo) * 100003ULL + static_cast<uint64_t>(hi);
}

TokenDependencyMapper::TokenDependencyMapper(Config cfg)
    : cfg_(std::move(cfg))
{
    int w = std::max(2, cfg_.window);
    win_buf_.assign(static_cast<std::size_t>(w), -1);
}

void TokenDependencyMapper::record_token(int token_id) {
    std::vector<std::vector<int>> new_clusters;
    {
        std::lock_guard<std::mutex> lk(mutex_);
        total_.fetch_add(1, std::memory_order_relaxed);

        // Update co-occurrence with all tokens currently in the window.
        int w = static_cast<int>(win_buf_.size());
        for (int i = 0; i < win_fill_; ++i) {
            int idx = (win_head_ - 1 - i + w) % w;
            int other = win_buf_[static_cast<std::size_t>(idx)];
            if (other >= 0 && other != token_id) {
                ++cooc_[make_key(token_id, other)];
            }
        }

        // Insert token into window.
        if (win_fill_ == w) {
            // Evict oldest element — no need to decrement counts (approximate window).
            // (Exact eviction would require re-scanning all pairs; approximation is fine.)
        } else {
            ++win_fill_;
        }
        win_buf_[static_cast<std::size_t>(win_head_)] = token_id;
        win_head_ = (win_head_ + 1) % w;

        ++insert_count_;
        int interval = std::max(1, cfg_.recompute_interval);
        if (insert_count_ % interval == 0) {
            find_clusters_locked(new_clusters);
        }
    }

    if (!new_clusters.empty()) {
        std::function<void(const std::vector<int>&)> cb;
        {
            std::lock_guard<std::mutex> lk(mutex_);
            cb = cfg_.on_cluster_detected;
        }
        for (auto& cl : new_clusters) {
            cluster_events_.fetch_add(1, std::memory_order_relaxed);
            if (cb) cb(cl);
        }
    }
}

void TokenDependencyMapper::find_clusters_locked(std::vector<std::vector<int>>& out) const {
    int min_cnt  = std::max(1, cfg_.min_count);
    int min_size = std::max(2, cfg_.min_cluster_size);

    // Collect tokens that have any edge above min_count.
    std::vector<int> candidates;
    for (auto& [key, cnt] : cooc_) {
        if (cnt >= min_cnt) {
            int lo = static_cast<int>(key / 100003ULL);
            int hi = static_cast<int>(key % 100003ULL);
            candidates.push_back(lo);
            candidates.push_back(hi);
        }
    }
    std::sort(candidates.begin(), candidates.end());
    candidates.erase(std::unique(candidates.begin(), candidates.end()), candidates.end());

    if (static_cast<int>(candidates.size()) < min_size) return;

    // Greedy clique scan: check all pairs in candidates for dense sub-sets.
    // O(|candidates|^2) — candidates set is bounded by practical vocabulary size.
    std::vector<int> cluster;
    for (int i = 0; i < static_cast<int>(candidates.size()); ++i) {
        cluster.clear();
        cluster.push_back(candidates[static_cast<std::size_t>(i)]);
        for (int j = i + 1; j < static_cast<int>(candidates.size()); ++j) {
            int b = candidates[static_cast<std::size_t>(j)];
            bool connected = true;
            for (int c : cluster) {
                auto it = cooc_.find(make_key(c, b));
                if (it == cooc_.end() || it->second < min_cnt) {
                    connected = false;
                    break;
                }
            }
            if (connected) cluster.push_back(b);
        }
        if (static_cast<int>(cluster.size()) >= min_size) {
            out.push_back(cluster);
        }
    }
}

int TokenDependencyMapper::cooccurrence(int id_a, int id_b) const {
    std::lock_guard<std::mutex> lk(mutex_);
    auto it = cooc_.find(make_key(id_a, id_b));
    return (it != cooc_.end()) ? it->second : 0;
}

std::string TokenDependencyMapper::to_stats_json() const {
    std::lock_guard<std::mutex> lk(mutex_);
    std::ostringstream ss;
    ss << "{"
       << "\"total_tokens\":"   << total_.load(std::memory_order_relaxed) << ","
       << "\"cluster_events\":" << cluster_events_.load(std::memory_order_relaxed) << ","
       << "\"edge_count\":"     << cooc_.size() << ","
       << "\"window_fill\":"    << win_fill_
       << "}";
    return ss.str();
}

void TokenDependencyMapper::reset() {
    std::lock_guard<std::mutex> lk(mutex_);
    int w = static_cast<int>(win_buf_.size());
    std::fill(win_buf_.begin(), win_buf_.end(), -1);
    win_head_ = win_fill_ = 0;
    cooc_.clear();
    insert_count_ = 0;
    total_.store(0, std::memory_order_relaxed);
    cluster_events_.store(0, std::memory_order_relaxed);
    (void)w;
}

void TokenDependencyMapper::update_config(const Config& cfg) {
    std::lock_guard<std::mutex> lk(mutex_);
    cfg_ = cfg;
    int w = std::max(2, cfg_.window);
    win_buf_.assign(static_cast<std::size_t>(w), -1);
    win_head_ = win_fill_ = 0;
    cooc_.clear();
    insert_count_ = 0;
}

} // namespace llmquant
