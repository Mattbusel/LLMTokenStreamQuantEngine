#include "SentimentTopologyMapper.h"

#include <algorithm>
#include <cmath>
#include <numeric>
#include <sstream>
#include <iomanip>

namespace llmquant {

SentimentTopologyMapper::SentimentTopologyMapper(Config cfg)
    : cfg_(std::move(cfg))
{
    int w = std::max(8, cfg_.window);
    win_buf_.assign(static_cast<std::size_t>(w), 0.0);
}

// Union-Find helpers (path-compressed, rank-based) on index arrays
static int uf_find(std::vector<int>& parent, int x) {
    while (parent[static_cast<std::size_t>(x)] != x) {
        parent[static_cast<std::size_t>(x)] = parent[static_cast<std::size_t>(parent[static_cast<std::size_t>(x)])];
        x = parent[static_cast<std::size_t>(x)];
    }
    return x;
}

static void uf_union(std::vector<int>& parent, std::vector<int>& rank, int a, int b) {
    a = uf_find(parent, a);
    b = uf_find(parent, b);
    if (a == b) return;
    if (rank[static_cast<std::size_t>(a)] < rank[static_cast<std::size_t>(b)]) std::swap(a, b);
    parent[static_cast<std::size_t>(b)] = a;
    if (rank[static_cast<std::size_t>(a)] == rank[static_cast<std::size_t>(b)])
        ++rank[static_cast<std::size_t>(a)];
}

void SentimentTopologyMapper::compute_persistence_locked() {
    int n = win_fill_;
    if (n < std::max(2, cfg_.min_samples)) return;

    int w = static_cast<int>(win_buf_.size());
    std::vector<double> vals(static_cast<std::size_t>(n));
    for (int i = 0; i < n; ++i) {
        int idx = (win_head_ - n + i + w * 2) % w;
        vals[static_cast<std::size_t>(i)] = win_buf_[static_cast<std::size_t>(idx)];
    }

    // Sort indices by value (sublevel-set: process smallest threshold first)
    std::vector<int> order(static_cast<std::size_t>(n));
    std::iota(order.begin(), order.end(), 0);
    std::sort(order.begin(), order.end(), [&](int a, int b) {
        return vals[static_cast<std::size_t>(a)] < vals[static_cast<std::size_t>(b)];
    });

    // Union-Find over positions; merge adjacent (|i - j| == 1) if both active
    std::vector<int> parent(static_cast<std::size_t>(n));
    std::iota(parent.begin(), parent.end(), 0);
    std::vector<int> rank(static_cast<std::size_t>(n), 0);
    std::vector<double> birth(static_cast<std::size_t>(n), 0.0);
    std::vector<bool> active(static_cast<std::size_t>(n), false);

    int components = 0;
    double total_pers = 0.0;

    for (int k = 0; k < n; ++k) {
        int idx = order[static_cast<std::size_t>(k)];
        double v = vals[static_cast<std::size_t>(idx)];

        active[static_cast<std::size_t>(idx)] = true;
        birth[static_cast<std::size_t>(idx)]  = v;
        ++components;

        // Check left neighbour
        if (idx > 0 && active[static_cast<std::size_t>(idx - 1)]) {
            int ra = uf_find(parent, idx);
            int rb = uf_find(parent, idx - 1);
            if (ra != rb) {
                // Elder rule: component with smaller birth survives
                double ba = birth[static_cast<std::size_t>(ra)];
                double bb = birth[static_cast<std::size_t>(rb)];
                double dying_birth = (ba > bb) ? ba : bb;
                double pers = v - dying_birth;
                if (pers > cfg_.min_persistence) total_pers += pers;
                uf_union(parent, rank, ra, rb);
                --components;
            }
        }
        // Check right neighbour
        if (idx < n - 1 && active[static_cast<std::size_t>(idx + 1)]) {
            int ra = uf_find(parent, idx);
            int rb = uf_find(parent, idx + 1);
            if (ra != rb) {
                double ba = birth[static_cast<std::size_t>(ra)];
                double bb = birth[static_cast<std::size_t>(rb)];
                double dying_birth = (ba > bb) ? ba : bb;
                double pers = v - dying_birth;
                if (pers > cfg_.min_persistence) total_pers += pers;
                uf_union(parent, rank, ra, rb);
                --components;
            }
        }
    }

    // Count persistent components (exclude the most persistent one = the whole signal)
    // Persistent components = those whose birth-to-max-value persistence > min_persistence
    // Simplified: count how many distinct root components survived all merges, minus 1
    // that is the global component. The number of persistent features = merges that
    // produced persistence > min_persistence = total merges that passed the filter.
    // components at end = 1 (one root). Persistent features fired above minus 1 for the
    // global component.  Use max(1, components) as proxy for Betti-0.
    // Actually let's count meaningful components from total merge events:
    // We track how many components_started = n, and each merge reduces by 1. The
    // number of "long-lived" components = those not yet merged = 1 at the end, but
    // also those that were killed with high persistence. Count them:
    int persistent_count = 1;  // always at least one
    // Re-do simpler: count distinct roots among active elements
    std::vector<bool> root_seen(static_cast<std::size_t>(n), false);
    for (int i = 0; i < n; ++i) {
        int r = uf_find(parent, i);
        if (!root_seen[static_cast<std::size_t>(r)]) {
            root_seen[static_cast<std::size_t>(r)] = true;
            // Check persistence of this component
            double pers = vals[order.back()] - birth[static_cast<std::size_t>(r)];
            if (pers > cfg_.min_persistence) ++persistent_count;
        }
    }
    // Subtract one (we started with 1 for the global)
    persistent_count = std::max(1, persistent_count - 1);

    components_.store(persistent_count, std::memory_order_relaxed);
    total_pers_.store(total_pers, std::memory_order_relaxed);
}

void SentimentTopologyMapper::record(double bias) {
    bool fire   = false;
    int old_cnt = 0, new_cnt = 0;
    std::function<void(int, int)> cb;

    {
        std::lock_guard<std::mutex> lk(mutex_);
        cb = cfg_.on_topology_change;
        total_.fetch_add(1, std::memory_order_relaxed);

        int w = static_cast<int>(win_buf_.size());
        if (win_fill_ < w) ++win_fill_;
        win_buf_[static_cast<std::size_t>(win_head_)] = bias;
        win_head_ = (win_head_ + 1) % w;

        compute_persistence_locked();

        int cnt = components_.load(std::memory_order_relaxed);
        if (prev_components_ >= 0 && cnt != prev_components_) {
            old_cnt = prev_components_;
            new_cnt = cnt;
            fire    = true;
            topo_events_.fetch_add(1, std::memory_order_relaxed);
        }
        prev_components_ = cnt;
    }

    if (fire && cb) cb(old_cnt, new_cnt);
}

std::string SentimentTopologyMapper::to_stats_json() const {
    std::lock_guard<std::mutex> lk(mutex_);
    std::ostringstream ss;
    ss << std::fixed << std::setprecision(6);
    ss << "{"
       << "\"component_count\":"   << components_.load(std::memory_order_relaxed) << ","
       << "\"total_persistence\":" << total_pers_.load(std::memory_order_relaxed) << ","
       << "\"topology_events\":"   << topo_events_.load(std::memory_order_relaxed) << ","
       << "\"total_records\":"     << total_.load(std::memory_order_relaxed)
       << "}";
    return ss.str();
}

void SentimentTopologyMapper::reset() {
    std::lock_guard<std::mutex> lk(mutex_);
    std::fill(win_buf_.begin(), win_buf_.end(), 0.0);
    win_head_ = win_fill_ = 0;
    prev_components_ = -1;
    components_.store(0, std::memory_order_relaxed);
    total_pers_.store(0.0, std::memory_order_relaxed);
    topo_events_.store(0, std::memory_order_relaxed);
    total_.store(0, std::memory_order_relaxed);
}

void SentimentTopologyMapper::update_config(const Config& cfg) {
    std::lock_guard<std::mutex> lk(mutex_);
    cfg_ = cfg;
    int w = std::max(8, cfg_.window);
    win_buf_.assign(static_cast<std::size_t>(w), 0.0);
    win_head_ = win_fill_ = 0;
    prev_components_ = -1;
    components_.store(0,   std::memory_order_relaxed);
    total_pers_.store(0.0, std::memory_order_relaxed);
    topo_events_.store(0,  std::memory_order_relaxed);
    total_.store(0,        std::memory_order_relaxed);
}

} // namespace llmquant
