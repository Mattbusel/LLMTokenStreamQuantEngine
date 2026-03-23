// AttentionCache.cpp — implementation of KV-cache management.

#include "AttentionCache.h"

#include <algorithm>

namespace llmquant {

void AttentionCache::store(CacheEntry entry) {
    CacheKey key{entry.layer_id, entry.head_id, entry.position};
    entries_[key] = std::move(entry);
}

std::optional<CacheEntry> AttentionCache::retrieve(std::size_t layer_id,
                                                    std::size_t head_id,
                                                    std::size_t position) const {
    CacheKey key{layer_id, head_id, position};
    auto it = entries_.find(key);
    if (it == entries_.end()) {
        ++misses_;
        return std::nullopt;
    }
    ++hits_;
    return it->second;
}

void AttentionCache::invalidate_from(std::size_t pos) {
    for (auto it = entries_.begin(); it != entries_.end(); ) {
        if (it->first.position >= pos) {
            it = entries_.erase(it);
        } else {
            ++it;
        }
    }
}

std::size_t AttentionCache::size() const noexcept {
    return entries_.size();
}

std::size_t AttentionCache::memory_bytes() const noexcept {
    std::size_t total = 0;
    for (const auto& [key, entry] : entries_) {
        total += entry.key_values.size() * sizeof(float);
    }
    return total;
}

double AttentionCache::hit_rate() const noexcept {
    std::size_t total = hits_ + misses_;
    if (total == 0) return 0.0;
    return static_cast<double>(hits_) / static_cast<double>(total);
}

void AttentionCache::reset() {
    entries_.clear();
    hits_ = 0;
    misses_ = 0;
}

}  // namespace llmquant
