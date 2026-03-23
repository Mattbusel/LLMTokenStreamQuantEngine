#include "KVCache.h"

#include <algorithm>
#include <stdexcept>

namespace llmquant {

KVCache::KVCache(int max_seq_len, int num_layers, int num_heads, int head_dim)
    : max_seq_len_(max_seq_len)
    , num_layers_(num_layers)
    , num_heads_(num_heads)
    , head_dim_(head_dim)
{
    if (max_seq_len <= 0)
        throw std::invalid_argument("KVCache: max_seq_len must be > 0");
    if (num_layers <= 0)
        throw std::invalid_argument("KVCache: num_layers must be > 0");
    if (num_heads <= 0)
        throw std::invalid_argument("KVCache: num_heads must be > 0");
    if (head_dim <= 0)
        throw std::invalid_argument("KVCache: head_dim must be > 0");
}

void KVCache::insert(int layer, int head, int position,
                     const std::vector<float>& key,
                     const std::vector<float>& value)
{
    KVKey k{layer, head, position};
    store_[k] = KVEntry{layer, head, position, key, value};
}

std::pair<std::vector<float>, std::vector<float>>
KVCache::lookup(int layer, int head, int position) const
{
    auto it = store_.find(KVKey{layer, head, position});
    if (it == store_.end()) {
        return {{}, {}};
    }
    return {it->second.key, it->second.value};
}

bool KVCache::has(int layer, int head, int position) const
{
    return store_.count(KVKey{layer, head, position}) > 0;
}

void KVCache::evict_positions_before(int position)
{
    auto it = store_.begin();
    while (it != store_.end()) {
        if (it->first.position < position) {
            it = store_.erase(it);
        } else {
            ++it;
        }
    }
}

void KVCache::clear()
{
    store_.clear();
}

std::size_t KVCache::size() const
{
    return store_.size();
}

std::size_t KVCache::memory_bytes() const
{
    if (store_.empty()) return 0;
    // Estimate: each entry holds key_size + value_size floats (4 bytes each)
    std::size_t total = 0;
    for (const auto& [key, entry] : store_) {
        total += (entry.key.size() + entry.value.size()) * sizeof(float);
    }
    return total;
}

std::vector<int> KVCache::cached_positions(int layer, int head) const
{
    std::vector<int> positions;
    for (const auto& [key, entry] : store_) {
        if (key.layer == layer && key.head == head) {
            positions.push_back(key.position);
        }
    }
    std::sort(positions.begin(), positions.end());
    return positions;
}

}  // namespace llmquant
