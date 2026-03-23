#include "PromptCache.h"
#include <algorithm>
#include <numeric>

namespace llmquant {

// FNV-1a 64-bit hash
CacheKey PromptCache::hash_prefix(const std::string& prefix) {
    uint64_t hash = 14695981039346656037ULL;
    for (unsigned char c : prefix) {
        hash ^= static_cast<uint64_t>(c);
        hash *= 1099511628211ULL;
    }
    return hash;
}

PromptCache::PromptCache(size_t max_entries, size_t max_kv_bytes)
    : max_entries_(max_entries), max_kv_bytes_(max_kv_bytes) {}

std::optional<std::pair<CacheKey, size_t>> PromptCache::find_prefix(const std::string& prompt) const {
    std::lock_guard<std::mutex> lock(mutex_);
    size_t best_len = 0;
    CacheKey best_key = 0;

    for (const auto& [key, entry] : cache_) {
        const auto& pfx = entry.prompt_prefix;
        if (pfx.size() > best_len && pfx.size() <= prompt.size() &&
            prompt.compare(0, pfx.size(), pfx) == 0) {
            best_len = pfx.size();
            best_key = key;
        }
    }

    if (best_len == 0) {
        misses_++;
        return std::nullopt;
    }
    hits_++;
    return std::make_pair(best_key, best_len);
}

void PromptCache::store(std::string prefix, std::vector<float> kv_data, size_t token_count) {
    std::lock_guard<std::mutex> lock(mutex_);
    size_t kv_bytes = kv_data.size() * sizeof(float);

    while ((cache_.size() >= max_entries_ || current_kv_bytes_ + kv_bytes > max_kv_bytes_)
           && !lru_order_.empty()) {
        evict_lru();
    }

    CacheKey key = hash_prefix(prefix);
    if (cache_.count(key)) return;  // already cached

    cache_.emplace(key, CacheEntry(key, std::move(prefix), std::move(kv_data), token_count));
    lru_order_.push_front(key);
    current_kv_bytes_ += kv_bytes;
}

std::optional<std::reference_wrapper<const CacheEntry>> PromptCache::get(CacheKey key) {
    std::lock_guard<std::mutex> lock(mutex_);
    auto it = cache_.find(key);
    if (it == cache_.end()) return std::nullopt;
    it->second.hit_count++;
    it->second.last_access = std::chrono::steady_clock::now();
    touch(key);
    return std::cref(it->second);
}

void PromptCache::evict(CacheKey key) {
    std::lock_guard<std::mutex> lock(mutex_);
    auto it = cache_.find(key);
    if (it == cache_.end()) return;
    current_kv_bytes_ -= it->second.kv_data.size() * sizeof(float);
    lru_order_.remove(key);
    cache_.erase(it);
}

void PromptCache::clear() {
    std::lock_guard<std::mutex> lock(mutex_);
    cache_.clear();
    lru_order_.clear();
    current_kv_bytes_ = 0;
}

PromptCache::CacheStats PromptCache::stats() const {
    std::lock_guard<std::mutex> lock(mutex_);
    uint64_t h = hits_.load(), m = misses_.load();
    double rate = (h + m) > 0 ? static_cast<double>(h) / (h + m) : 0.0;
    return {cache_.size(), current_kv_bytes_, h, m, rate};
}

void PromptCache::evict_lru() {
    if (lru_order_.empty()) return;
    CacheKey old_key = lru_order_.back();
    lru_order_.pop_back();
    auto it = cache_.find(old_key);
    if (it != cache_.end()) {
        current_kv_bytes_ -= it->second.kv_data.size() * sizeof(float);
        cache_.erase(it);
    }
}

void PromptCache::touch(CacheKey key) {
    lru_order_.remove(key);
    lru_order_.push_front(key);
}

} // namespace llmquant
