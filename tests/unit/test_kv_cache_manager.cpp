#include <gtest/gtest.h>

#ifdef LLMQUANT_KV_CACHE_MANAGER_ENABLED

#include "KVCacheManager.h"

#include <algorithm>
#include <set>
#include <vector>

using namespace llmquant;

namespace {

// ============================================================
// Construction
// ============================================================

TEST(KVCacheManager, DefaultConstruction) {
    KVCacheManager mgr(4, 16, 32, 1024);
    EXPECT_EQ(mgr.num_layers(), 4);
    EXPECT_EQ(mgr.block_size(), 16);
    EXPECT_EQ(mgr.max_blocks(), 32);
    EXPECT_EQ(mgr.bytes_per_block(), 1024u);

    auto s = mgr.stats();
    EXPECT_EQ(s.total_blocks, 32);
    EXPECT_EQ(s.used_blocks, 0);
    EXPECT_EQ(s.free_blocks, 32);
    EXPECT_NEAR(s.fragmentation_ratio, 1.0f, 1e-5f);
}

// ============================================================
// Allocate and free
// ============================================================

TEST(KVCacheManager, AllocateSingleBlock) {
    KVCacheManager mgr(2, 8, 4, 256);
    int blk = mgr.allocate_block(1, 0);
    EXPECT_GE(blk, 0);

    auto* b = mgr.get_block(blk);
    ASSERT_NE(b, nullptr);
    EXPECT_FALSE(b->is_free);
    EXPECT_EQ(b->seq_id, 1);
    EXPECT_EQ(b->layer_id, 0);

    auto s = mgr.stats();
    EXPECT_EQ(s.used_blocks, 1);
    EXPECT_EQ(s.free_blocks, 3);
}

TEST(KVCacheManager, FreeBlock) {
    KVCacheManager mgr(2, 8, 4, 64);
    int blk = mgr.allocate_block(5, 1);
    ASSERT_GE(blk, 0);

    mgr.free_block(blk);
    auto* b = mgr.get_block(blk);
    ASSERT_NE(b, nullptr);
    EXPECT_TRUE(b->is_free);
    EXPECT_EQ(b->seq_id, -1);

    auto s = mgr.stats();
    EXPECT_EQ(s.used_blocks, 0);
}

TEST(KVCacheManager, FreeAndReallocate) {
    KVCacheManager mgr(1, 4, 2, 64);
    int b0 = mgr.allocate_block(1, 0);
    int b1 = mgr.allocate_block(2, 0);
    EXPECT_GE(b0, 0);
    EXPECT_GE(b1, 0);
    EXPECT_NE(b0, b1);

    mgr.free_block(b0);
    int b2 = mgr.allocate_block(3, 0);
    EXPECT_GE(b2, 0);
    EXPECT_EQ(b2, b0); // should reuse the freed slot
}

// ============================================================
// Sequence cleanup
// ============================================================

TEST(KVCacheManager, FreeSequenceReleasesAllBlocks) {
    KVCacheManager mgr(4, 8, 16, 128);

    // Allocate 4 blocks for seq 10 across different layers.
    std::vector<int> seq_blocks;
    for (int layer = 0; layer < 4; ++layer) {
        int blk = mgr.allocate_block(10, layer);
        ASSERT_GE(blk, 0);
        seq_blocks.push_back(blk);
    }

    // Allocate 2 blocks for a different sequence.
    int other0 = mgr.allocate_block(20, 0);
    int other1 = mgr.allocate_block(20, 1);
    EXPECT_GE(other0, 0);
    EXPECT_GE(other1, 0);

    EXPECT_EQ(mgr.stats().used_blocks, 6);

    mgr.free_sequence(10);

    // Seq 10 blocks should now be free.
    for (int blk : seq_blocks) {
        auto* b = mgr.get_block(blk);
        ASSERT_NE(b, nullptr);
        EXPECT_TRUE(b->is_free);
    }

    // Seq 20 blocks should still be in use.
    EXPECT_EQ(mgr.stats().used_blocks, 2);
}

// ============================================================
// Full cache returns -1
// ============================================================

TEST(KVCacheManager, FullCacheReturnsMinusOne) {
    KVCacheManager mgr(1, 4, 3, 64);
    EXPECT_GE(mgr.allocate_block(1, 0), 0);
    EXPECT_GE(mgr.allocate_block(2, 0), 0);
    EXPECT_GE(mgr.allocate_block(3, 0), 0);

    // Cache is now full.
    int blk = mgr.allocate_block(4, 0);
    EXPECT_EQ(blk, -1);

    auto s = mgr.stats();
    EXPECT_EQ(s.used_blocks, 3);
    EXPECT_EQ(s.free_blocks, 0);
}

// ============================================================
// Stats correctness
// ============================================================

TEST(KVCacheManager, StatsCorrect) {
    KVCacheManager mgr(2, 8, 8, 64);
    for (int i = 0; i < 5; ++i) {
        mgr.allocate_block(i, 0);
    }
    auto s = mgr.stats();
    EXPECT_EQ(s.total_blocks, 8);
    EXPECT_EQ(s.used_blocks, 5);
    EXPECT_EQ(s.free_blocks, 3);
    EXPECT_NEAR(s.fragmentation_ratio, 3.0f / 8.0f, 1e-4f);
}

// ============================================================
// get_sequence_blocks sorted by layer
// ============================================================

TEST(KVCacheManager, GetSequenceBlocksSortedByLayer) {
    KVCacheManager mgr(4, 8, 16, 64);

    // Allocate layers in reverse order.
    std::vector<int> allocated;
    for (int layer = 3; layer >= 0; --layer) {
        int blk = mgr.allocate_block(42, layer);
        ASSERT_GE(blk, 0);
        allocated.push_back(blk);
    }

    auto blocks = mgr.get_sequence_blocks(42);
    ASSERT_EQ(blocks.size(), 4u);

    // Verify layer order.
    for (std::size_t i = 0; i + 1 < blocks.size(); ++i) {
        int layer_a = mgr.get_block(blocks[i])->layer_id;
        int layer_b = mgr.get_block(blocks[i + 1])->layer_id;
        EXPECT_LT(layer_a, layer_b);
    }
}

TEST(KVCacheManager, GetSequenceBlocksEmptyForUnknownSeq) {
    KVCacheManager mgr(2, 8, 4, 64);
    auto blocks = mgr.get_sequence_blocks(999);
    EXPECT_TRUE(blocks.empty());
}

// ============================================================
// Out-of-bounds accessors
// ============================================================

TEST(KVCacheManager, GetBlockOutOfBoundsReturnsNullptr) {
    KVCacheManager mgr(1, 4, 4, 64);
    EXPECT_EQ(mgr.get_block(-1), nullptr);
    EXPECT_EQ(mgr.get_block(4), nullptr);
    EXPECT_EQ(mgr.get_block(100), nullptr);
}

TEST(KVCacheManager, FreeBlockOutOfBoundsIsNoop) {
    KVCacheManager mgr(1, 4, 4, 64);
    // Should not crash.
    EXPECT_NO_THROW(mgr.free_block(-1));
    EXPECT_NO_THROW(mgr.free_block(4));
}

} // namespace

#else  // LLMQUANT_KV_CACHE_MANAGER_ENABLED

TEST(KVCacheManagerDisabled, FeatureFlagOff) {
    SUCCEED() << "KVCacheManager feature flag is OFF — tests skipped.";
}

#endif // LLMQUANT_KV_CACHE_MANAGER_ENABLED
