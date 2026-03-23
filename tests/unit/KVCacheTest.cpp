#include <gtest/gtest.h>
#include "KVCache.h"

#include <algorithm>
#include <vector>

using namespace llmquant;

// ---------------------------------------------------------------------------
// Construction
// ---------------------------------------------------------------------------

TEST(KVCache, ConstructValid) {
    EXPECT_NO_THROW(KVCache(512, 12, 8, 64));
}

TEST(KVCache, ConstructInvalidMaxSeqLen) {
    EXPECT_THROW(KVCache(0, 12, 8, 64), std::invalid_argument);
}

TEST(KVCache, ConstructInvalidNumLayers) {
    EXPECT_THROW(KVCache(512, 0, 8, 64), std::invalid_argument);
}

TEST(KVCache, ConstructInvalidHeadDim) {
    EXPECT_THROW(KVCache(512, 12, 8, 0), std::invalid_argument);
}

// ---------------------------------------------------------------------------
// Insert & Lookup
// ---------------------------------------------------------------------------

TEST(KVCache, InsertAndLookupReturnsCorrectVectors) {
    KVCache cache(512, 4, 4, 8);
    std::vector<float> k = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f};
    std::vector<float> v = {8.0f, 7.0f, 6.0f, 5.0f, 4.0f, 3.0f, 2.0f, 1.0f};
    cache.insert(0, 0, 10, k, v);

    auto [rk, rv] = cache.lookup(0, 0, 10);
    EXPECT_EQ(rk, k);
    EXPECT_EQ(rv, v);
}

TEST(KVCache, LookupMissReturnsEmptyVectors) {
    KVCache cache(512, 4, 4, 8);
    auto [rk, rv] = cache.lookup(0, 0, 999);
    EXPECT_TRUE(rk.empty());
    EXPECT_TRUE(rv.empty());
}

TEST(KVCache, HasReturnsTrueAfterInsert) {
    KVCache cache(512, 4, 4, 8);
    cache.insert(1, 2, 5, {1.0f}, {2.0f});
    EXPECT_TRUE(cache.has(1, 2, 5));
}

TEST(KVCache, HasReturnsFalseForMissing) {
    KVCache cache(512, 4, 4, 8);
    EXPECT_FALSE(cache.has(0, 0, 0));
}

TEST(KVCache, SizeIncreasesAfterInsert) {
    KVCache cache(512, 4, 4, 8);
    EXPECT_EQ(cache.size(), 0u);
    cache.insert(0, 0, 0, {1.0f}, {2.0f});
    EXPECT_EQ(cache.size(), 1u);
    cache.insert(0, 0, 1, {1.0f}, {2.0f});
    EXPECT_EQ(cache.size(), 2u);
}

TEST(KVCache, InsertOverwritesExistingEntry) {
    KVCache cache(512, 4, 4, 8);
    cache.insert(0, 0, 5, {1.0f}, {1.0f});
    cache.insert(0, 0, 5, {9.9f}, {8.8f});
    EXPECT_EQ(cache.size(), 1u);
    auto [rk, rv] = cache.lookup(0, 0, 5);
    EXPECT_FLOAT_EQ(rk[0], 9.9f);
}

// ---------------------------------------------------------------------------
// Eviction
// ---------------------------------------------------------------------------

TEST(KVCache, EvictPositionsBeforeRemovesOldEntries) {
    KVCache cache(512, 4, 4, 8);
    cache.insert(0, 0, 0, {1.0f}, {1.0f});
    cache.insert(0, 0, 1, {2.0f}, {2.0f});
    cache.insert(0, 0, 2, {3.0f}, {3.0f});
    cache.insert(0, 0, 5, {4.0f}, {4.0f});

    cache.evict_positions_before(2);

    EXPECT_FALSE(cache.has(0, 0, 0));
    EXPECT_FALSE(cache.has(0, 0, 1));
    EXPECT_TRUE(cache.has(0, 0, 2));
    EXPECT_TRUE(cache.has(0, 0, 5));
    EXPECT_EQ(cache.size(), 2u);
}

TEST(KVCache, ClearRemovesAllEntries) {
    KVCache cache(512, 4, 4, 8);
    cache.insert(0, 0, 0, {1.0f}, {1.0f});
    cache.insert(1, 1, 1, {2.0f}, {2.0f});
    cache.clear();
    EXPECT_EQ(cache.size(), 0u);
}

// ---------------------------------------------------------------------------
// Memory estimate
// ---------------------------------------------------------------------------

TEST(KVCache, MemoryBytesIsPositiveAfterInsert) {
    KVCache cache(512, 4, 4, 8);
    EXPECT_EQ(cache.memory_bytes(), 0u);
    std::vector<float> k(8, 1.0f);
    std::vector<float> v(8, 2.0f);
    cache.insert(0, 0, 0, k, v);
    // 8 + 8 = 16 floats * 4 bytes = 64 bytes
    EXPECT_EQ(cache.memory_bytes(), 64u);
}

TEST(KVCache, MemoryBytesScalesWithEntries) {
    KVCache cache(512, 4, 4, 8);
    std::vector<float> k(8, 1.0f);
    std::vector<float> v(8, 2.0f);
    cache.insert(0, 0, 0, k, v);
    cache.insert(0, 0, 1, k, v);
    EXPECT_EQ(cache.memory_bytes(), 128u);
}

// ---------------------------------------------------------------------------
// Cached positions
// ---------------------------------------------------------------------------

TEST(KVCache, CachedPositionsReturnsSortedList) {
    KVCache cache(512, 4, 4, 8);
    cache.insert(0, 0, 5, {1.0f}, {1.0f});
    cache.insert(0, 0, 2, {1.0f}, {1.0f});
    cache.insert(0, 0, 8, {1.0f}, {1.0f});
    cache.insert(0, 1, 3, {1.0f}, {1.0f});  // different head

    auto positions = cache.cached_positions(0, 0);
    ASSERT_EQ(positions.size(), 3u);
    EXPECT_EQ(positions[0], 2);
    EXPECT_EQ(positions[1], 5);
    EXPECT_EQ(positions[2], 8);
}

TEST(KVCache, CachedPositionsEmptyForUnknownLayerHead) {
    KVCache cache(512, 4, 4, 8);
    auto positions = cache.cached_positions(99, 99);
    EXPECT_TRUE(positions.empty());
}
