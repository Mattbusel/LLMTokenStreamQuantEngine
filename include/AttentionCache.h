#pragma once
// AttentionCache.h — KV-cache management for transformer attention layers.

#include <cstddef>
#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

namespace llmquant {

/// A single cached key-value entry for one (layer, head, position) triple.
struct CacheEntry {
    std::size_t layer_id{0};
    std::size_t head_id{0};
    std::size_t position{0};
    std::vector<float> key_values;
};

/// Key used to look up entries in the cache map.
struct CacheKey {
    std::size_t layer_id;
    std::size_t head_id;
    std::size_t position;

    bool operator==(const CacheKey& o) const noexcept {
        return layer_id == o.layer_id && head_id == o.head_id && position == o.position;
    }
};

struct CacheKeyHash {
    std::size_t operator()(const CacheKey& k) const noexcept {
        // FNV-1a-inspired combination.
        std::size_t h = 0xcbf29ce484222325ULL;
        auto mix = [&](std::size_t v) {
            h ^= v;
            h *= 0x100000001b3ULL;
        };
        mix(k.layer_id);
        mix(k.head_id);
        mix(k.position);
        return h;
    }
};

/// Attention KV-cache with hit-rate tracking and positional invalidation.
class AttentionCache {
public:
    AttentionCache() = default;
    ~AttentionCache() = default;

    // Non-copyable, movable.
    AttentionCache(const AttentionCache&) = delete;
    AttentionCache& operator=(const AttentionCache&) = delete;
    AttentionCache(AttentionCache&&) = default;
    AttentionCache& operator=(AttentionCache&&) = default;

    /// Insert or overwrite a cache entry.
    void store(CacheEntry entry);

    /// Look up an entry; returns std::nullopt on a miss.
    std::optional<CacheEntry> retrieve(std::size_t layer_id,
                                       std::size_t head_id,
                                       std::size_t position) const;

    /// Evict all entries whose position >= *pos*.
    void invalidate_from(std::size_t pos);

    /// Total number of entries currently cached.
    std::size_t size() const noexcept;

    /// Total bytes consumed by stored key_values vectors.
    std::size_t memory_bytes() const noexcept;

    /// Cache hit-rate = hits / (hits + misses), or 0.0 if no queries yet.
    double hit_rate() const noexcept;

    /// Evict all entries and reset statistics.
    void reset();

private:
    std::unordered_map<CacheKey, CacheEntry, CacheKeyHash> entries_;
    mutable std::size_t hits_{0};
    mutable std::size_t misses_{0};
};

}  // namespace llmquant
