#pragma once
#include <string>
#include <vector>
#include <unordered_map>
#include <list>
#include <mutex>
#include <atomic>
#include <optional>
#include <chrono>

namespace llmquant {

/// Cache key: hash of prompt prefix
using CacheKey = uint64_t;

/// Cached KV state for a prompt prefix
struct CacheEntry {
    CacheKey key;
    std::string prompt_prefix;
    std::vector<float> kv_data;  // Flattened KV cache data
    size_t prefix_token_count;
    std::chrono::steady_clock::time_point last_access;
    std::atomic<uint32_t> hit_count{0};

    CacheEntry() = default;
    CacheEntry(CacheKey k, std::string prefix, std::vector<float> data, size_t tokens)
        : key(k), prompt_prefix(std::move(prefix)), kv_data(std::move(data)),
          prefix_token_count(tokens), last_access(std::chrono::steady_clock::now()) {}

    CacheEntry(const CacheEntry& o)
        : key(o.key), prompt_prefix(o.prompt_prefix), kv_data(o.kv_data),
          prefix_token_count(o.prefix_token_count), last_access(o.last_access),
          hit_count(o.hit_count.load()) {}
};

/// LRU cache for prompt KV states (prefix sharing)
class PromptCache {
public:
    explicit PromptCache(size_t max_entries = 256, size_t max_kv_bytes = 1024 * 1024 * 512);
    ~PromptCache() = default;

    /// Hash a prompt prefix for lookup
    static CacheKey hash_prefix(const std::string& prefix);

    /// Find longest cached prefix of the given prompt
    /// Returns (cache_key, prefix_length) or nullopt if no match
    std::optional<std::pair<CacheKey, size_t>> find_prefix(const std::string& prompt) const;

    /// Store a new cache entry
    void store(std::string prefix, std::vector<float> kv_data, size_t token_count);

    /// Get cached KV data by key
    std::optional<std::reference_wrapper<const CacheEntry>> get(CacheKey key);

    /// Evict an entry
    void evict(CacheKey key);

    /// Clear all entries
    void clear();

    /// Statistics
    struct CacheStats {
        size_t entry_count;
        size_t total_kv_bytes;
        uint64_t hits;
        uint64_t misses;
        double hit_rate;
    };
    CacheStats stats() const;

private:
    size_t max_entries_;
    size_t max_kv_bytes_;
    mutable std::mutex mutex_;
    std::unordered_map<CacheKey, CacheEntry> cache_;
    std::list<CacheKey> lru_order_;  // front = most recently used
    mutable std::atomic<uint64_t> hits_{0};
    mutable std::atomic<uint64_t> misses_{0};
    size_t current_kv_bytes_{0};

    void evict_lru();
    void touch(CacheKey key);
};

} // namespace llmquant
