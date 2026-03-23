// test_AttentionCache.cpp — GTest unit tests for AttentionCache.

#include "AttentionCache.h"
#include <gtest/gtest.h>

using namespace llmquant;

// -------------------------------------------------------------------------
// Helpers
// -------------------------------------------------------------------------

static CacheEntry make_entry(std::size_t layer, std::size_t head,
                              std::size_t pos,
                              std::vector<float> kv = {1.0f, 2.0f, 3.0f}) {
    return CacheEntry{layer, head, pos, std::move(kv)};
}

// -------------------------------------------------------------------------
// Tests
// -------------------------------------------------------------------------

TEST(AttentionCache, InitialSizeIsZero) {
    AttentionCache cache;
    EXPECT_EQ(cache.size(), 0u);
}

TEST(AttentionCache, StoreAndRetrieve) {
    AttentionCache cache;
    cache.store(make_entry(0, 0, 5));
    auto result = cache.retrieve(0, 0, 5);
    ASSERT_TRUE(result.has_value());
    EXPECT_EQ(result->position, 5u);
}

TEST(AttentionCache, RetrieveMissReturnsNullopt) {
    AttentionCache cache;
    auto result = cache.retrieve(0, 0, 99);
    EXPECT_FALSE(result.has_value());
}

TEST(AttentionCache, SizeIncreasesOnStore) {
    AttentionCache cache;
    cache.store(make_entry(0, 0, 1));
    cache.store(make_entry(0, 0, 2));
    EXPECT_EQ(cache.size(), 2u);
}

TEST(AttentionCache, OverwriteExistingEntry) {
    AttentionCache cache;
    cache.store(make_entry(0, 0, 1, {1.0f}));
    cache.store(make_entry(0, 0, 1, {99.0f}));
    EXPECT_EQ(cache.size(), 1u);
    auto result = cache.retrieve(0, 0, 1);
    ASSERT_TRUE(result.has_value());
    EXPECT_FLOAT_EQ(result->key_values.front(), 99.0f);
}

TEST(AttentionCache, InvalidateFromRemovesCorrectEntries) {
    AttentionCache cache;
    cache.store(make_entry(0, 0, 1));
    cache.store(make_entry(0, 0, 5));
    cache.store(make_entry(0, 0, 10));
    cache.invalidate_from(5);
    EXPECT_EQ(cache.size(), 1u);
    EXPECT_TRUE(cache.retrieve(0, 0, 1).has_value());
    EXPECT_FALSE(cache.retrieve(0, 0, 5).has_value());
    EXPECT_FALSE(cache.retrieve(0, 0, 10).has_value());
}

TEST(AttentionCache, InvalidateFromZeroEvictsAll) {
    AttentionCache cache;
    cache.store(make_entry(0, 0, 0));
    cache.store(make_entry(0, 0, 1));
    cache.invalidate_from(0);
    EXPECT_EQ(cache.size(), 0u);
}

TEST(AttentionCache, MemoryBytesCorrect) {
    AttentionCache cache;
    cache.store(make_entry(0, 0, 1, {1.0f, 2.0f}));  // 2 floats = 8 bytes
    EXPECT_EQ(cache.memory_bytes(), 2 * sizeof(float));
}

TEST(AttentionCache, HitRateZeroWhenNoQueries) {
    AttentionCache cache;
    EXPECT_DOUBLE_EQ(cache.hit_rate(), 0.0);
}

TEST(AttentionCache, HitRateOneAfterAllHits) {
    AttentionCache cache;
    cache.store(make_entry(0, 0, 1));
    cache.retrieve(0, 0, 1);
    cache.retrieve(0, 0, 1);
    EXPECT_DOUBLE_EQ(cache.hit_rate(), 1.0);
}

TEST(AttentionCache, HitRateMixed) {
    AttentionCache cache;
    cache.store(make_entry(0, 0, 1));
    cache.retrieve(0, 0, 1);  // hit
    cache.retrieve(0, 0, 2);  // miss
    EXPECT_DOUBLE_EQ(cache.hit_rate(), 0.5);
}

TEST(AttentionCache, ResetClearsAllStateAndStats) {
    AttentionCache cache;
    cache.store(make_entry(0, 0, 1));
    cache.retrieve(0, 0, 1);
    cache.reset();
    EXPECT_EQ(cache.size(), 0u);
    EXPECT_EQ(cache.memory_bytes(), 0u);
    EXPECT_DOUBLE_EQ(cache.hit_rate(), 0.0);
}

TEST(AttentionCache, MultipleLayersAndHeads) {
    AttentionCache cache;
    cache.store(make_entry(0, 0, 1));
    cache.store(make_entry(0, 1, 1));
    cache.store(make_entry(1, 0, 1));
    EXPECT_EQ(cache.size(), 3u);
    EXPECT_TRUE(cache.retrieve(0, 0, 1).has_value());
    EXPECT_TRUE(cache.retrieve(0, 1, 1).has_value());
    EXPECT_TRUE(cache.retrieve(1, 0, 1).has_value());
}
