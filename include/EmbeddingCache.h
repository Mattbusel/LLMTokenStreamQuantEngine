#pragma once

#include <cstddef>
#include <cstdint>
#include <functional>
#include <unordered_map>
#include <vector>

namespace llmquant {

/// A single cache entry for a token embedding.
struct EmbeddingEntry {
    int token_id;
    std::vector<float> embedding;
    uint64_t access_count;
    uint64_t last_access_ms;
};

/**
 * @brief LRU embedding cache for token vectors.
 *
 * Stores dense float embeddings keyed by token id.
 * Supports lookup, nearest-neighbour search (cosine similarity), and
 * automatic LRU eviction when capacity is exceeded.
 */
class EmbeddingCache {
public:
    EmbeddingCache(size_t capacity, size_t embedding_dim);

    /// Store an embedding (overwrites any existing entry for token_id).
    void store(int token_id, const std::vector<float>& embedding);

    /// Lookup: returns the embedding or an empty vector if not cached.
    std::vector<float> lookup(int token_id) const;

    bool contains(int token_id) const;

    /// Evict the least-recently-accessed entry.
    void evict_lru();

    size_t size() const;
    size_t capacity() const;

    float cosine_similarity(
        const std::vector<float>& a, const std::vector<float>& b) const;

    /// Brute-force top-k nearest neighbours by cosine similarity.
    std::vector<int> nearest_neighbors(
        const std::vector<float>& query, size_t k) const;

    void clear();

    /// Eagerly populate the cache by calling embed_fn for each token id.
    void prefetch(
        const std::vector<int>& token_ids,
        std::function<std::vector<float>(int)> embed_fn);

private:
    std::unordered_map<int, EmbeddingEntry> cache_;
    size_t capacity_;
    size_t embedding_dim_;
    mutable uint64_t clock_ms_ = 0;  ///< Monotonic logical clock.

    uint64_t tick() const;
};

}  // namespace llmquant
