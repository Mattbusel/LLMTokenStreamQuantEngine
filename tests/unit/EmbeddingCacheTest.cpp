#include <gtest/gtest.h>
#include "EmbeddingCache.h"

#include <cmath>
#include <vector>

using namespace llmquant;

static std::vector<float> make_embedding(float val, size_t dim = 4) {
    return std::vector<float>(dim, val);
}

// ---------------------------------------------------------------------------
// Construction
// ---------------------------------------------------------------------------

TEST(EmbeddingCache, ConstructValid) {
    EXPECT_NO_THROW(EmbeddingCache(128, 64));
}

TEST(EmbeddingCache, ZeroCapacityThrows) {
    EXPECT_THROW(EmbeddingCache(0, 64), std::invalid_argument);
}

// ---------------------------------------------------------------------------
// Store / Lookup / Contains
// ---------------------------------------------------------------------------

TEST(EmbeddingCache, StoreAndLookupReturnsEmbedding) {
    EmbeddingCache cache(10, 4);
    auto emb = make_embedding(1.0f);
    cache.store(42, emb);
    auto result = cache.lookup(42);
    EXPECT_EQ(result, emb);
}

TEST(EmbeddingCache, LookupMissingReturnsEmpty) {
    EmbeddingCache cache(10, 4);
    EXPECT_TRUE(cache.lookup(999).empty());
}

TEST(EmbeddingCache, ContainsTrueAfterStore) {
    EmbeddingCache cache(10, 4);
    cache.store(1, make_embedding(0.5f));
    EXPECT_TRUE(cache.contains(1));
}

TEST(EmbeddingCache, ContainsFalseForMissing) {
    EmbeddingCache cache(10, 4);
    EXPECT_FALSE(cache.contains(99));
}

// ---------------------------------------------------------------------------
// Size / Capacity
// ---------------------------------------------------------------------------

TEST(EmbeddingCache, SizeIncreasesOnStore) {
    EmbeddingCache cache(10, 4);
    cache.store(1, make_embedding(1.0f));
    cache.store(2, make_embedding(2.0f));
    EXPECT_EQ(cache.size(), 2u);
}

TEST(EmbeddingCache, CapacityReturnsConstructorValue) {
    EmbeddingCache cache(64, 8);
    EXPECT_EQ(cache.capacity(), 64u);
}

// ---------------------------------------------------------------------------
// Eviction
// ---------------------------------------------------------------------------

TEST(EmbeddingCache, EvictLruReducesSize) {
    EmbeddingCache cache(10, 4);
    cache.store(1, make_embedding(1.0f));
    cache.store(2, make_embedding(2.0f));
    cache.evict_lru();
    EXPECT_EQ(cache.size(), 1u);
}

TEST(EmbeddingCache, StoreEvictsWhenFull) {
    EmbeddingCache cache(2, 4);
    cache.store(1, make_embedding(1.0f));
    cache.store(2, make_embedding(2.0f));
    // Lookup 2 to make 1 the LRU
    cache.lookup(2);
    // Store 3 — should evict 1
    cache.store(3, make_embedding(3.0f));
    EXPECT_EQ(cache.size(), 2u);
    EXPECT_TRUE(cache.contains(3));
}

// ---------------------------------------------------------------------------
// Cosine similarity
// ---------------------------------------------------------------------------

TEST(EmbeddingCache, CosineSimilarityIdentical) {
    EmbeddingCache cache(10, 4);
    auto a = make_embedding(1.0f);
    float sim = cache.cosine_similarity(a, a);
    EXPECT_NEAR(sim, 1.0f, 1e-5f);
}

TEST(EmbeddingCache, CosineSimilarityOrthogonal) {
    EmbeddingCache cache(10, 4);
    std::vector<float> a = {1.0f, 0.0f, 0.0f, 0.0f};
    std::vector<float> b = {0.0f, 1.0f, 0.0f, 0.0f};
    float sim = cache.cosine_similarity(a, b);
    EXPECT_NEAR(sim, 0.0f, 1e-5f);
}

TEST(EmbeddingCache, CosineSimilarityEmptyReturnsZero) {
    EmbeddingCache cache(10, 4);
    std::vector<float> empty;
    EXPECT_NEAR(cache.cosine_similarity(empty, empty), 0.0f, 1e-5f);
}

// ---------------------------------------------------------------------------
// Nearest neighbours
// ---------------------------------------------------------------------------

TEST(EmbeddingCache, NearestNeighborsReturnsTopK) {
    EmbeddingCache cache(10, 4);
    cache.store(1, {1.0f, 0.0f, 0.0f, 0.0f});
    cache.store(2, {0.0f, 1.0f, 0.0f, 0.0f});
    cache.store(3, {0.0f, 0.0f, 1.0f, 0.0f});
    // Query nearest to token 1
    std::vector<float> query = {1.0f, 0.0f, 0.0f, 0.0f};
    auto nn = cache.nearest_neighbors(query, 1);
    ASSERT_EQ(nn.size(), 1u);
    EXPECT_EQ(nn[0], 1);
}

// ---------------------------------------------------------------------------
// Clear / Prefetch
// ---------------------------------------------------------------------------

TEST(EmbeddingCache, ClearEmptiesCache) {
    EmbeddingCache cache(10, 4);
    cache.store(1, make_embedding(1.0f));
    cache.clear();
    EXPECT_EQ(cache.size(), 0u);
}

TEST(EmbeddingCache, PrefetchPopulatesCache) {
    EmbeddingCache cache(10, 4);
    auto embed_fn = [](int id) -> std::vector<float> {
        return std::vector<float>(4, static_cast<float>(id));
    };
    cache.prefetch({10, 20, 30}, embed_fn);
    EXPECT_TRUE(cache.contains(10));
    EXPECT_TRUE(cache.contains(20));
    EXPECT_TRUE(cache.contains(30));
}
