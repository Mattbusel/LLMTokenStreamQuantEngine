#include "EmbeddingCache.h"

#include <algorithm>
#include <cmath>
#include <numeric>
#include <stdexcept>

namespace llmquant {

EmbeddingCache::EmbeddingCache(size_t capacity, size_t embedding_dim)
    : capacity_(capacity), embedding_dim_(embedding_dim)
{
    if (capacity == 0)
        throw std::invalid_argument("EmbeddingCache: capacity must be > 0");
}

uint64_t EmbeddingCache::tick() const {
    return ++clock_ms_;
}

// ---------------------------------------------------------------------------
// Store
// ---------------------------------------------------------------------------

void EmbeddingCache::store(int token_id, const std::vector<float>& embedding) {
    if (cache_.size() >= capacity_ && cache_.find(token_id) == cache_.end()) {
        evict_lru();
    }
    uint64_t now = tick();
    EmbeddingEntry& entry = cache_[token_id];
    entry.token_id     = token_id;
    entry.embedding    = embedding;
    entry.last_access_ms = now;
    entry.access_count  = (entry.access_count > 0) ? entry.access_count + 1 : 1;
}

// ---------------------------------------------------------------------------
// Lookup
// ---------------------------------------------------------------------------

std::vector<float> EmbeddingCache::lookup(int token_id) const {
    auto it = cache_.find(token_id);
    if (it == cache_.end()) return {};
    it->second.last_access_ms = tick();
    it->second.access_count++;
    return it->second.embedding;
}

bool EmbeddingCache::contains(int token_id) const {
    return cache_.find(token_id) != cache_.end();
}

// ---------------------------------------------------------------------------
// Eviction
// ---------------------------------------------------------------------------

void EmbeddingCache::evict_lru() {
    if (cache_.empty()) return;
    auto oldest = cache_.begin();
    for (auto it = cache_.begin(); it != cache_.end(); ++it) {
        if (it->second.last_access_ms < oldest->second.last_access_ms) {
            oldest = it;
        }
    }
    cache_.erase(oldest);
}

// ---------------------------------------------------------------------------
// Sizing
// ---------------------------------------------------------------------------

size_t EmbeddingCache::size() const     { return cache_.size(); }
size_t EmbeddingCache::capacity() const { return capacity_; }

// ---------------------------------------------------------------------------
// Cosine similarity
// ---------------------------------------------------------------------------

float EmbeddingCache::cosine_similarity(
    const std::vector<float>& a, const std::vector<float>& b) const
{
    if (a.empty() || b.empty() || a.size() != b.size()) return 0.0f;
    float dot = 0.0f, norm_a = 0.0f, norm_b = 0.0f;
    for (size_t i = 0; i < a.size(); ++i) {
        dot    += a[i] * b[i];
        norm_a += a[i] * a[i];
        norm_b += b[i] * b[i];
    }
    float denom = std::sqrt(norm_a) * std::sqrt(norm_b);
    if (denom < 1e-12f) return 0.0f;
    return dot / denom;
}

// ---------------------------------------------------------------------------
// Nearest neighbours (brute force)
// ---------------------------------------------------------------------------

std::vector<int> EmbeddingCache::nearest_neighbors(
    const std::vector<float>& query, size_t k) const
{
    // Collect (similarity, token_id) pairs
    std::vector<std::pair<float, int>> scored;
    scored.reserve(cache_.size());
    for (const auto& [id, entry] : cache_) {
        float sim = cosine_similarity(query, entry.embedding);
        scored.emplace_back(sim, id);
    }
    // Partial sort: top-k descending
    size_t take = std::min(k, scored.size());
    std::partial_sort(
        scored.begin(), scored.begin() + static_cast<std::ptrdiff_t>(take),
        scored.end(),
        [](const auto& x, const auto& y) { return x.first > y.first; });

    std::vector<int> result;
    result.reserve(take);
    for (size_t i = 0; i < take; ++i) {
        result.push_back(scored[i].second);
    }
    return result;
}

// ---------------------------------------------------------------------------
// Clear / prefetch
// ---------------------------------------------------------------------------

void EmbeddingCache::clear() { cache_.clear(); }

void EmbeddingCache::prefetch(
    const std::vector<int>& token_ids,
    std::function<std::vector<float>(int)> embed_fn)
{
    for (int id : token_ids) {
        if (!contains(id)) {
            store(id, embed_fn(id));
        }
    }
}

}  // namespace llmquant
