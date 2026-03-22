#include "TokenNgramProfiler.h"

#include <algorithm>
#include <sstream>

namespace llmquant {

TokenNgramProfiler::TokenNgramProfiler(Config cfg) : cfg_(std::move(cfg)) {}

void TokenNgramProfiler::record_ngram(const std::string& key, int n) {
    // Must be called with mutex_ held.
    auto it = freq_.find(key);
    if (it == freq_.end()) {
        if (freq_.size() >= cfg_.max_ngrams) {
            // LRU eviction: remove oldest inserted entry.
            if (!insert_order_.empty()) {
                freq_.erase(insert_order_.front());
                insert_order_.erase(insert_order_.begin());
            }
        }
        freq_[key] = 1;
        insert_order_.push_back(key);
    } else {
        it->second++;
    }
    uint64_t cnt = freq_[key];
    if (cnt == cfg_.hot_threshold && cfg_.on_hot_ngram) {
        hot_events_.fetch_add(1, std::memory_order_relaxed);
        // Fire callback outside the lock – copy key/n/count first.
        // We'll fire after releasing the lock via a local copy.
    }
    (void)n;
}

void TokenNgramProfiler::push(const std::string& token) {
    std::vector<std::pair<std::string,int>> to_fire;
    {
        std::lock_guard<std::mutex> lk(mutex_);
        history_.push_back(token);
        // Keep history at trigram depth.
        while (history_.size() > 3) history_.pop_front();

        total_tokens_.fetch_add(1, std::memory_order_relaxed);

        if (cfg_.enable_bigrams && history_.size() >= 2) {
            std::string bg = history_[history_.size()-2] + " " + history_[history_.size()-1];
            auto it = freq_.find(bg);
            if (it == freq_.end()) {
                if (freq_.size() >= cfg_.max_ngrams && !insert_order_.empty()) {
                    freq_.erase(insert_order_.front());
                    insert_order_.erase(insert_order_.begin());
                }
                freq_[bg] = 1;
                insert_order_.push_back(bg);
            } else {
                it->second++;
            }
            if (freq_[bg] == cfg_.hot_threshold) {
                to_fire.push_back({bg, 2});
                hot_events_.fetch_add(1, std::memory_order_relaxed);
            }
        }
        if (cfg_.enable_trigrams && history_.size() == 3) {
            std::string tg = history_[0] + " " + history_[1] + " " + history_[2];
            auto it = freq_.find(tg);
            if (it == freq_.end()) {
                if (freq_.size() >= cfg_.max_ngrams && !insert_order_.empty()) {
                    freq_.erase(insert_order_.front());
                    insert_order_.erase(insert_order_.begin());
                }
                freq_[tg] = 1;
                insert_order_.push_back(tg);
            } else {
                it->second++;
            }
            if (freq_[tg] == cfg_.hot_threshold) {
                to_fire.push_back({tg, 3});
                hot_events_.fetch_add(1, std::memory_order_relaxed);
            }
        }
    }
    if (cfg_.on_hot_ngram) {
        for (auto& [ng, n] : to_fire)
            cfg_.on_hot_ngram(ng, cfg_.hot_threshold, n);
    }
}

std::vector<TokenNgramProfiler::NgramEntry>
TokenNgramProfiler::top_by_frequency(uint32_t n) const {
    std::lock_guard<std::mutex> lk(mutex_);
    std::vector<NgramEntry> result;
    result.reserve(freq_.size());
    for (const auto& [key, cnt] : freq_) {
        NgramEntry e;
        e.ngram = key;
        e.count = cnt;
        // Determine n from space count.
        int spaces = 0;
        for (char c : key) if (c == ' ') spaces++;
        e.n = spaces + 1;
        result.push_back(e);
    }
    std::partial_sort(result.begin(),
                      result.begin() + std::min(static_cast<size_t>(n), result.size()),
                      result.end(),
                      [](const NgramEntry& a, const NgramEntry& b) {
                          return a.count > b.count;
                      });
    if (result.size() > n) result.resize(n);
    return result;
}

uint64_t TokenNgramProfiler::frequency(const std::string& ngram) const {
    std::lock_guard<std::mutex> lk(mutex_);
    auto it = freq_.find(ngram);
    return it == freq_.end() ? 0 : it->second;
}

uint32_t TokenNgramProfiler::distinct_ngrams() const {
    std::lock_guard<std::mutex> lk(mutex_);
    return static_cast<uint32_t>(freq_.size());
}

void TokenNgramProfiler::reset() {
    std::lock_guard<std::mutex> lk(mutex_);
    history_.clear();
    freq_.clear();
    insert_order_.clear();
    total_tokens_.store(0, std::memory_order_relaxed);
    hot_events_.store(0, std::memory_order_relaxed);
}

void TokenNgramProfiler::update_config(const Config& cfg) {
    std::lock_guard<std::mutex> lk(mutex_);
    cfg_ = cfg;
    history_.clear();
    freq_.clear();
    insert_order_.clear();
}

std::string TokenNgramProfiler::to_stats_json() const {
    auto top = top_by_frequency(5);
    uint64_t tot = total_tokens_.load(std::memory_order_relaxed);
    uint64_t hot = hot_events_.load(std::memory_order_relaxed);
    std::ostringstream ss;
    ss << "{\"total_tokens\":" << tot
       << ",\"distinct_ngrams\":" << distinct_ngrams()
       << ",\"hot_events\":" << hot
       << ",\"top\":[";
    for (size_t i = 0; i < top.size(); ++i) {
        if (i > 0) ss << ",";
        ss << "{\"ngram\":\"" << top[i].ngram << "\""
           << ",\"count\":" << top[i].count
           << ",\"n\":" << top[i].n << "}";
    }
    ss << "]}";
    return ss.str();
}

} // namespace llmquant
