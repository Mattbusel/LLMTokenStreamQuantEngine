#include "CrossAssetCorrelationMonitor.h"

#include <algorithm>
#include <cmath>
#include <sstream>

namespace llmquant {

CrossAssetCorrelationMonitor::CrossAssetCorrelationMonitor(Config config)
    : config_(std::move(config)) {}

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

int CrossAssetCorrelationMonitor::pair_index(int i, int j) const noexcept {
    // Map (i, j) with i < j to a unique index in the upper-triangular matrix.
    // pairs_ is sized to n*(n-1)/2 where n = kMaxAssets.
    return i * kMaxAssets - i * (i + 1) / 2 + (j - i - 1);
}

void CrossAssetCorrelationMonitor::recompute_pair(PairState& ps) noexcept {
    int n = ps.fill;
    if (n < config_.min_samples) {
        ps.rho = 0.0;
        return;
    }

    double inv_n = 1.0 / n;
    double mean_a = ps.sum_a * inv_n;
    double mean_b = ps.sum_b * inv_n;
    double var_a  = ps.sum_aa * inv_n - mean_a * mean_a;
    double var_b  = ps.sum_bb * inv_n - mean_b * mean_b;
    double cov_ab = ps.sum_ab * inv_n - mean_a * mean_b;

    double denom = std::sqrt(var_a * var_b);
    ps.rho = (denom > 1e-12) ? std::max(-1.0, std::min(1.0, cov_ab / denom)) : 0.0;
}

// ---------------------------------------------------------------------------
// Registration
// ---------------------------------------------------------------------------

int CrossAssetCorrelationMonitor::register_asset(const std::string& name) {
    std::lock_guard<std::mutex> lk(mutex_);
    if (asset_ids_.count(name)) return asset_ids_[name];

    int id = static_cast<int>(asset_names_.size());
    if (id >= kMaxAssets) return -1;

    asset_ids_[name] = id;
    asset_names_.push_back(name);
    n_assets_.store(id + 1, std::memory_order_relaxed);

    // Allocate pair states for all existing assets vs new one.
    int max_pairs = kMaxAssets * (kMaxAssets - 1) / 2;
    if (static_cast<int>(pairs_.size()) < max_pairs) {
        pairs_.resize(static_cast<size_t>(max_pairs));
        // Initialise ring for new pairs.
        for (int i = 0; i < id; ++i) {
            auto& ps = pairs_[static_cast<size_t>(pair_index(i, id))];
            ps.ring.assign(static_cast<size_t>(config_.window_size), {0.0, 0.0});
            ps.head = ps.fill = 0;
            ps.sum_a = ps.sum_b = ps.sum_aa = ps.sum_bb = ps.sum_ab = 0.0;
            ps.rho = 0.0;
        }
    }

    return id;
}

// ---------------------------------------------------------------------------
// Recording
// ---------------------------------------------------------------------------

bool CrossAssetCorrelationMonitor::record(const std::string& name, double value) noexcept {
    total_records_.fetch_add(1, std::memory_order_relaxed);

    // Collect (pair_index, old_rho, new_rho) to fire callbacks outside lock.
    struct Event { int pi; std::string na, nb; double rho; bool high; };
    std::vector<Event> events;
    Config cfg_copy;

    {
        std::lock_guard<std::mutex> lk(mutex_);
        cfg_copy = config_;

        auto it = asset_ids_.find(name);
        if (it == asset_ids_.end()) return false;

        int id = it->second;
        int n  = static_cast<int>(asset_names_.size());

        // Update all pairs that involve this asset.
        for (int other = 0; other < n; ++other) {
            if (other == id) continue;
            int i = std::min(id, other);
            int j = std::max(id, other);
            auto& ps = pairs_[static_cast<size_t>(pair_index(i, j))];

            if (ps.ring.empty()) continue;

            // Retrieve the most-recently stored value for the OTHER asset.
            // We store a single value per asset per slot; the last recorded
            // value for `other` is what we pair with `value`.
            // Strategy: we keep a per-asset "last value" cache.
            // (Stored below after the loop.)
        }

        // Last-value cache for pairwise update.
        if (static_cast<int>(last_value_.size()) <= id)
            last_value_.resize(static_cast<size_t>(id + 1), 0.0);
        last_value_[static_cast<size_t>(id)] = value;

        // Now update all pairs with the new joint observation.
        for (int other = 0; other < n; ++other) {
            if (other == id) continue;
            if (static_cast<int>(last_value_.size()) <= other) continue;

            int i = std::min(id, other);
            int j = std::max(id, other);
            auto& ps = pairs_[static_cast<size_t>(pair_index(i, j))];
            if (ps.ring.empty()) continue;

            double va = (id == i) ? value : last_value_[static_cast<size_t>(other)];
            double vb = (id == j) ? value : last_value_[static_cast<size_t>(other)];
            if (id == i) { va = value; vb = last_value_[static_cast<size_t>(other)]; }
            else         { va = last_value_[static_cast<size_t>(other)]; vb = value; }

            // Evict oldest if full.
            if (ps.fill == config_.window_size) {
                auto& old = ps.ring[static_cast<size_t>(ps.head)];
                ps.sum_a  -= old.first;
                ps.sum_b  -= old.second;
                ps.sum_aa -= old.first  * old.first;
                ps.sum_bb -= old.second * old.second;
                ps.sum_ab -= old.first  * old.second;
            }

            ps.ring[static_cast<size_t>(ps.head)] = {va, vb};
            ps.head = (ps.head + 1) % config_.window_size;
            if (ps.fill < config_.window_size) ++ps.fill;

            ps.sum_a  += va;
            ps.sum_b  += vb;
            ps.sum_aa += va * va;
            ps.sum_bb += vb * vb;
            ps.sum_ab += va * vb;

            double old_rho = ps.rho;
            recompute_pair(ps);

            // Check threshold transitions.
            bool now_high = (ps.rho >= config_.high_corr_threshold);
            bool now_low  = (ps.rho < config_.low_corr_threshold &&
                             ps.fill >= config_.min_samples);
            if (now_high && !ps.was_high) {
                ps.was_high = true;
                ps.was_low  = false;
                high_events_.fetch_add(1, std::memory_order_relaxed);
                events.push_back({pair_index(i, j),
                    asset_names_[static_cast<size_t>(i)],
                    asset_names_[static_cast<size_t>(j)],
                    ps.rho, true});
            } else if (now_low && !ps.was_low) {
                ps.was_low  = true;
                ps.was_high = false;
                low_events_.fetch_add(1, std::memory_order_relaxed);
                events.push_back({pair_index(i, j),
                    asset_names_[static_cast<size_t>(i)],
                    asset_names_[static_cast<size_t>(j)],
                    ps.rho, false});
            } else if (!now_high && !now_low) {
                ps.was_high = false;
                ps.was_low  = false;
            }
            (void)old_rho;
        }
    }

    // Fire callbacks outside lock.
    for (const auto& ev : events) {
        if (ev.high && cfg_copy.on_high_correlation)
            cfg_copy.on_high_correlation(ev.na, ev.nb, ev.rho);
        else if (!ev.high && cfg_copy.on_low_correlation)
            cfg_copy.on_low_correlation(ev.na, ev.nb, ev.rho);
    }

    return true;
}

// ---------------------------------------------------------------------------
// Query
// ---------------------------------------------------------------------------

double CrossAssetCorrelationMonitor::correlation(
    const std::string& a, const std::string& b) const noexcept {
    std::lock_guard<std::mutex> lk(mutex_);
    auto ia = asset_ids_.find(a);
    auto ib = asset_ids_.find(b);
    if (ia == asset_ids_.end() || ib == asset_ids_.end()) return 0.0;
    int i = std::min(ia->second, ib->second);
    int j = std::max(ia->second, ib->second);
    if (i == j) return 1.0;
    return pairs_[static_cast<size_t>(pair_index(i, j))].rho;
}

// ---------------------------------------------------------------------------
// Config / reset
// ---------------------------------------------------------------------------

void CrossAssetCorrelationMonitor::update_config(Config config) {
    std::lock_guard<std::mutex> lk(mutex_);
    config_ = std::move(config);
    for (auto& ps : pairs_) {
        ps.ring.assign(static_cast<size_t>(config_.window_size), {0.0, 0.0});
        ps.head = ps.fill = 0;
        ps.sum_a = ps.sum_b = ps.sum_aa = ps.sum_bb = ps.sum_ab = 0.0;
        ps.rho = 0.0;
        ps.was_high = ps.was_low = false;
    }
    last_value_.clear();
    total_records_.store(0, std::memory_order_relaxed);
    high_events_.store(0, std::memory_order_relaxed);
    low_events_.store(0, std::memory_order_relaxed);
}

void CrossAssetCorrelationMonitor::reset() {
    std::lock_guard<std::mutex> lk(mutex_);
    for (auto& ps : pairs_) {
        std::fill(ps.ring.begin(), ps.ring.end(), std::make_pair(0.0, 0.0));
        ps.head = ps.fill = 0;
        ps.sum_a = ps.sum_b = ps.sum_aa = ps.sum_bb = ps.sum_ab = 0.0;
        ps.rho = 0.0;
        ps.was_high = ps.was_low = false;
    }
    last_value_.assign(last_value_.size(), 0.0);
    total_records_.store(0, std::memory_order_relaxed);
    high_events_.store(0, std::memory_order_relaxed);
    low_events_.store(0, std::memory_order_relaxed);
}

std::string CrossAssetCorrelationMonitor::to_stats_json() const {
    std::lock_guard<std::mutex> lk(mutex_);
    std::ostringstream o;
    o << std::fixed;
    o.precision(4);
    o << "{\"asset_count\":"   << static_cast<int>(asset_names_.size())
      << ",\"total_records\":" << total_records_.load(std::memory_order_relaxed)
      << ",\"high_events\":"   << high_events_.load(std::memory_order_relaxed)
      << ",\"low_events\":"    << low_events_.load(std::memory_order_relaxed)
      << ",\"pairs\":[";

    bool first = true;
    int n = static_cast<int>(asset_names_.size());
    for (int i = 0; i < n; ++i) {
        for (int j = i + 1; j < n; ++j) {
            if (!first) o << ",";
            first = false;
            double rho = pairs_[static_cast<size_t>(pair_index(i, j))].rho;
            o << "{\"a\":\"" << asset_names_[static_cast<size_t>(i)] << "\""
              << ",\"b\":\"" << asset_names_[static_cast<size_t>(j)] << "\""
              << ",\"rho\":" << rho << "}";
        }
    }
    o << "]}";
    return o.str();
}

} // namespace llmquant
