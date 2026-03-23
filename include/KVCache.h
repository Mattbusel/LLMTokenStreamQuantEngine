#pragma once

#include <cstddef>
#include <cstdint>
#include <functional>
#include <unordered_map>
#include <utility>
#include <vector>

namespace llmquant {

/// A single key-value cache entry for one attention layer/head/position.
struct KVEntry {
    int layer;
    int head;
    int position;
    std::vector<float> key;
    std::vector<float> value;
};

/// Composite key for indexing the KV cache.
struct KVKey {
    int layer;
    int head;
    int position;

    bool operator==(const KVKey& other) const noexcept {
        return layer == other.layer && head == other.head && position == other.position;
    }
};

}  // namespace llmquant

namespace std {
template <>
struct hash<llmquant::KVKey> {
    std::size_t operator()(const llmquant::KVKey& k) const noexcept {
        return static_cast<std::size_t>((k.layer * 31 + k.head) * 31 + k.position);
    }
};
}  // namespace std

namespace llmquant {

/**
 * @brief Key-Value cache for attention layers.
 *
 * Stores key and value vectors per (layer, head, position) triple.
 * Supports insertion, lookup, eviction of old positions, and memory estimation.
 */
class KVCache {
public:
    /**
     * @param max_seq_len  Maximum sequence length (informational; not enforced).
     * @param num_layers   Number of transformer layers.
     * @param num_heads    Number of attention heads per layer.
     * @param head_dim     Dimension of each key/value vector.
     */
    KVCache(int max_seq_len, int num_layers, int num_heads, int head_dim);

    /// Insert or overwrite a (layer, head, position) entry.
    void insert(int layer, int head, int position,
                const std::vector<float>& key,
                const std::vector<float>& value);

    /// Lookup by (layer, head, position). Returns {key, value} or two empty
    /// vectors if the entry does not exist.
    std::pair<std::vector<float>, std::vector<float>>
    lookup(int layer, int head, int position) const;

    /// Return true if the entry exists.
    bool has(int layer, int head, int position) const;

    /// Remove all entries whose position < given value.
    void evict_positions_before(int position);

    /// Remove all entries.
    void clear();

    /// Number of cached entries.
    std::size_t size() const;

    /// Estimated memory in bytes: entries * (key_size + value_size) * 4.
    std::size_t memory_bytes() const;

    /// Sorted list of cached positions for a given (layer, head).
    std::vector<int> cached_positions(int layer, int head) const;

private:
    int max_seq_len_;
    int num_layers_;
    int num_heads_;
    int head_dim_;

    std::unordered_map<KVKey, KVEntry> store_;
};

}  // namespace llmquant
