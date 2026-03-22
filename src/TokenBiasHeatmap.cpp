#include "TokenBiasHeatmap.h"

#include <algorithm>
#include <sstream>

namespace llmquant {

void TokenBiasHeatmap::record(const std::string& token, double bias) {
    total_records_.fetch_add(1, std::memory_order_relaxed);

    std::lock_guard<std::mutex> lk(mtx_);

    auto it = map_.find(token);
    if (it == map_.end()) {
        // Evict oldest entry if at capacity.
        if (static_cast<uint32_t>(map_.size()) >= cfg_.max_tokens) {
            auto oldest = map_.begin();
            for (auto jt = map_.begin(); jt != map_.end(); ++jt) {
                if (jt->second.insert_order < oldest->second.insert_order)
                    oldest = jt;
            }
            map_.erase(oldest);
        }
        Entry e;
        e.insert_order    = insert_counter_++;
        e.total_bias      = bias;
        e.abs_contribution = std::abs(bias);
        e.count           = 1;
        map_.emplace(token, e);
    } else {
        it->second.total_bias       += bias;
        it->second.abs_contribution += std::abs(bias);
        it->second.count            += 1;
    }
}

std::vector<TokenBiasHeatmap::TokenStats>
TokenBiasHeatmap::snapshot_sorted(bool by_abs, bool ascending) const {
    std::vector<TokenStats> out;
    {
        std::lock_guard<std::mutex> lk(mtx_);
        out.reserve(map_.size());
        for (const auto& [tok, e] : map_) {
            TokenStats s;
            s.token            = tok;
            s.total_bias       = e.total_bias;
            s.abs_contribution = e.abs_contribution;
            s.count            = e.count;
            s.mean_bias        = (e.count > 0) ? (e.total_bias / static_cast<double>(e.count)) : 0.0;
            out.push_back(std::move(s));
        }
    }
    if (by_abs) {
        std::sort(out.begin(), out.end(), [ascending](const TokenStats& a, const TokenStats& b) {
            return ascending ? (a.abs_contribution < b.abs_contribution)
                             : (a.abs_contribution > b.abs_contribution);
        });
    } else {
        std::sort(out.begin(), out.end(), [ascending](const TokenStats& a, const TokenStats& b) {
            return ascending ? (a.total_bias < b.total_bias)
                             : (a.total_bias > b.total_bias);
        });
    }
    return out;
}

std::vector<TokenBiasHeatmap::TokenStats>
TokenBiasHeatmap::top_by_abs_contribution(uint32_t n) const {
    auto v = snapshot_sorted(true, false);
    if (static_cast<uint32_t>(v.size()) > n) v.resize(n);
    return v;
}

std::vector<TokenBiasHeatmap::TokenStats>
TokenBiasHeatmap::top_bullish(uint32_t n) const {
    auto v = snapshot_sorted(false, false);
    if (static_cast<uint32_t>(v.size()) > n) v.resize(n);
    return v;
}

std::vector<TokenBiasHeatmap::TokenStats>
TokenBiasHeatmap::top_bearish(uint32_t n) const {
    auto v = snapshot_sorted(false, true);
    if (static_cast<uint32_t>(v.size()) > n) v.resize(n);
    return v;
}

uint32_t TokenBiasHeatmap::distinct_tokens() const {
    std::lock_guard<std::mutex> lk(mtx_);
    return static_cast<uint32_t>(map_.size());
}

void TokenBiasHeatmap::reset() {
    std::lock_guard<std::mutex> lk(mtx_);
    map_.clear();
    insert_counter_ = 0;
    total_records_.store(0, std::memory_order_relaxed);
}

std::string TokenBiasHeatmap::to_stats_json() const {
    auto top = top_by_abs_contribution(10);
    std::ostringstream o;
    o << std::fixed;
    o.precision(4);
    o << "{\"total_records\":" << total_records()
      << ",\"distinct_tokens\":" << distinct_tokens()
      << ",\"top_tokens\":[";
    for (size_t i = 0; i < top.size(); ++i) {
        const auto& s = top[i];
        if (i > 0) o << ",";
        o << "{\"token\":\"" << s.token << "\""
          << ",\"total_bias\":"       << s.total_bias
          << ",\"abs_contribution\":" << s.abs_contribution
          << ",\"count\":"            << s.count
          << ",\"mean_bias\":"        << s.mean_bias
          << "}";
    }
    o << "]}";
    return o.str();
}

} // namespace llmquant
