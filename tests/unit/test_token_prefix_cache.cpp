#include <gtest/gtest.h>

#ifdef LLMQUANT_TOKEN_PREFIX_CACHE_ENABLED

#include "TokenPrefixCache.h"

#include <vector>

using namespace llmquant;

// ============================================================
// Lookup miss on empty cache
// ============================================================

TEST(TokenPrefixCache, LookupMissOnEmptyCache) {
    TokenPrefixCache cache;
    auto hit = cache.lookup({1, 2, 3});
    EXPECT_FALSE(hit.is_hit());
    EXPECT_EQ(hit.matched_length, 0u);
    EXPECT_EQ(hit.kv_block_ptr, nullptr);
}

TEST(TokenPrefixCache, StatsInitialState) {
    TokenPrefixCache cache;
    auto s = cache.stats();
    EXPECT_EQ(s.hit_count,  0u);
    EXPECT_EQ(s.miss_count, 0u);
    EXPECT_NEAR(s.hit_rate, 0.0, 1e-9);
}

// ============================================================
// Insert + lookup hit
// ============================================================

TEST(TokenPrefixCache, InsertThenLookupFullMatch) {
    TokenPrefixCache cache;
    int fake_block = 42;
    cache.insert({1, 2, 3}, &fake_block);

    auto hit = cache.lookup({1, 2, 3});
    EXPECT_TRUE(hit.is_hit());
    EXPECT_EQ(hit.kv_block_ptr, &fake_block);
    EXPECT_EQ(hit.matched_length, 3u);
}

TEST(TokenPrefixCache, InsertThenLookupPrefixMatch) {
    TokenPrefixCache cache;
    int block_a = 10;
    cache.insert({1, 2, 3}, &block_a);

    // Query longer sequence: first 3 tokens should match.
    auto hit = cache.lookup({1, 2, 3, 4, 5});
    EXPECT_TRUE(hit.is_hit());
    EXPECT_EQ(hit.matched_length, 3u);
    EXPECT_EQ(hit.kv_block_ptr, &block_a);
}

TEST(TokenPrefixCache, LookupReturnsBestKvBlock) {
    TokenPrefixCache cache;
    int block_2 = 2;
    int block_4 = 4;
    cache.insert({1, 2},       &block_2);
    cache.insert({1, 2, 3, 4}, &block_4);

    // Query of length 3: best match is [1,2] (only [1,2] has a stored block
    // that is a prefix; [1,2,3,4] requires all 4 tokens).
    auto hit3 = cache.lookup({1, 2, 3});
    EXPECT_TRUE(hit3.is_hit());
    EXPECT_EQ(hit3.kv_block_ptr, &block_2);
    EXPECT_EQ(hit3.matched_length, 2u);  // matched [1,2]

    // Query of length 4: matches [1,2,3,4].
    auto hit4 = cache.lookup({1, 2, 3, 4});
    EXPECT_TRUE(hit4.is_hit());
    EXPECT_EQ(hit4.kv_block_ptr, &block_4);
    EXPECT_EQ(hit4.matched_length, 4u);
}

// ============================================================
// Prefix matching returns correct length
// ============================================================

TEST(TokenPrefixCache, PrefixMatchReturnedLength) {
    TokenPrefixCache cache;
    int block = 99;
    cache.insert({10, 20, 30}, &block);

    // Exact match → length = 3.
    EXPECT_EQ(cache.lookup({10, 20, 30}).matched_length, 3u);

    // Shorter query → partial trie walk, no hit (no block at [10]).
    auto h1 = cache.lookup({10});
    EXPECT_EQ(h1.matched_length, 1u);
    EXPECT_FALSE(h1.is_hit());  // no block stored at depth 1

    // Different prefix → miss.
    auto h2 = cache.lookup({99, 1, 2});
    EXPECT_FALSE(h2.is_hit());
    EXPECT_EQ(h2.matched_length, 0u);
}

// ============================================================
// Eviction reduces node count
// ============================================================

TEST(TokenPrefixCache, EvictionReducesNodeCount) {
    TokenPrefixCache cache;
    int b1 = 1, b2 = 2, b3 = 3;
    cache.insert({1},    &b1);
    cache.insert({2},    &b2);
    cache.insert({3},    &b3);

    // Each insert of a length-1 sequence adds 1 node (+ root = total 4).
    EXPECT_GE(cache.stats().node_count, 3u);

    size_t before = cache.stats().node_count;
    cache.evict_lru(1);  // evict down to 1 node (root only, if possible)
    size_t after = cache.stats().node_count;
    EXPECT_LT(after, before);
}

TEST(TokenPrefixCache, EvictionDoesNotRemoveRoot) {
    TokenPrefixCache cache;
    cache.evict_lru(0);
    EXPECT_GE(cache.stats().node_count, 1u);  // root always remains
}

// ============================================================
// Stats counters
// ============================================================

TEST(TokenPrefixCache, HitMissCounters) {
    TokenPrefixCache cache;
    int b = 7;
    cache.insert({5, 6, 7}, &b);

    cache.lookup({5, 6, 7});  // hit
    cache.lookup({1, 2, 3});  // miss
    cache.lookup({5, 6, 7});  // hit

    auto s = cache.stats();
    EXPECT_EQ(s.hit_count,  2u);
    EXPECT_EQ(s.miss_count, 1u);
    EXPECT_NEAR(s.hit_rate, 2.0 / 3.0, 1e-9);
}

TEST(TokenPrefixCache, NodeCountAfterInserts) {
    TokenPrefixCache cache;
    int b = 0;
    // Insert 5-token sequence: creates root + 5 child nodes = 6 total.
    cache.insert({1, 2, 3, 4, 5}, &b);
    EXPECT_EQ(cache.stats().node_count, 6u);

    // Insert overlapping prefix [1,2] — only 0 new nodes since [1] and [1,2] exist.
    int b2 = 1;
    cache.insert({1, 2}, &b2);
    EXPECT_EQ(cache.stats().node_count, 6u);  // no new nodes

    // Insert branching sequence [1, 99]: one new node.
    int b3 = 2;
    cache.insert({1, 99}, &b3);
    EXPECT_EQ(cache.stats().node_count, 7u);
}

#endif // LLMQUANT_TOKEN_PREFIX_CACHE_ENABLED
