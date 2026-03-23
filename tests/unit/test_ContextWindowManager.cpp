#include <gtest/gtest.h>
#include "ContextWindowManager.h"

using namespace llmquant;

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

static ContextSegment make_seg(
    uint64_t    id,
    std::size_t tokens,
    int         priority = 2,
    uint32_t    hash     = 0)
{
    return ContextSegment{id, tokens, priority, hash};
}

// ---------------------------------------------------------------------------
// Construction
// ---------------------------------------------------------------------------

TEST(ContextWindowManager, InitialStateEmpty) {
    ContextWindowManager mgr(8192);
    EXPECT_TRUE(mgr.empty());
    EXPECT_EQ(mgr.total_tokens(), 0u);
    EXPECT_EQ(mgr.capacity(), 8192u);
    EXPECT_EQ(mgr.free_tokens(), 8192u);
    EXPECT_EQ(mgr.segment_count(), 0u);
    EXPECT_NEAR(mgr.utilization(), 0.0, 1e-12);
}

// ---------------------------------------------------------------------------
// add_segment
// ---------------------------------------------------------------------------

TEST(ContextWindowManager, AddSegmentFits) {
    ContextWindowManager mgr(1000);
    EXPECT_TRUE(mgr.add_segment(make_seg(1, 100)));
    EXPECT_EQ(mgr.total_tokens(), 100u);
    EXPECT_EQ(mgr.segment_count(), 1u);
}

TEST(ContextWindowManager, AddSegmentFillsExactly) {
    ContextWindowManager mgr(100);
    EXPECT_TRUE(mgr.add_segment(make_seg(1, 100)));
    EXPECT_TRUE(mgr.full());
}

TEST(ContextWindowManager, AddSegmentDoesNotFit) {
    ContextWindowManager mgr(100);
    EXPECT_FALSE(mgr.add_segment(make_seg(1, 101)));
    EXPECT_TRUE(mgr.empty());
}

TEST(ContextWindowManager, AddSegmentDoesNotFitAfterPartialFill) {
    ContextWindowManager mgr(100);
    mgr.add_segment(make_seg(1, 80));
    EXPECT_FALSE(mgr.add_segment(make_seg(2, 30)));
    EXPECT_EQ(mgr.total_tokens(), 80u);
}

TEST(ContextWindowManager, AddMultipleSegments) {
    ContextWindowManager mgr(1000);
    mgr.add_segment(make_seg(1, 100));
    mgr.add_segment(make_seg(2, 200));
    mgr.add_segment(make_seg(3, 300));
    EXPECT_EQ(mgr.total_tokens(), 600u);
    EXPECT_EQ(mgr.segment_count(), 3u);
}

// ---------------------------------------------------------------------------
// remove_segment
// ---------------------------------------------------------------------------

TEST(ContextWindowManager, RemoveSegmentFound) {
    ContextWindowManager mgr(1000);
    mgr.add_segment(make_seg(42, 200));
    EXPECT_TRUE(mgr.remove_segment(42));
    EXPECT_EQ(mgr.total_tokens(), 0u);
    EXPECT_TRUE(mgr.empty());
}

TEST(ContextWindowManager, RemoveSegmentNotFound) {
    ContextWindowManager mgr(1000);
    EXPECT_FALSE(mgr.remove_segment(999));
}

TEST(ContextWindowManager, RemoveSegmentUpdatesCount) {
    ContextWindowManager mgr(1000);
    mgr.add_segment(make_seg(1, 100));
    mgr.add_segment(make_seg(2, 200));
    mgr.remove_segment(1);
    EXPECT_EQ(mgr.segment_count(), 1u);
    EXPECT_EQ(mgr.total_tokens(), 200u);
}

// ---------------------------------------------------------------------------
// evict_to_fit — basic
// ---------------------------------------------------------------------------

TEST(ContextWindowManager, EvictToFitSingleSegment) {
    ContextWindowManager mgr(1000);
    mgr.add_segment(make_seg(1, 800, 2));
    std::size_t evicted = mgr.evict_to_fit(900);
    EXPECT_EQ(evicted, 1u);
    EXPECT_EQ(mgr.total_tokens(), 0u);
    EXPECT_GE(mgr.free_tokens(), 900u);
}

TEST(ContextWindowManager, EvictToFitNothingNeeded) {
    ContextWindowManager mgr(1000);
    mgr.add_segment(make_seg(1, 100, 2));
    std::size_t evicted = mgr.evict_to_fit(800);
    EXPECT_EQ(evicted, 0u);  // already 900 free — no eviction needed
}

TEST(ContextWindowManager, EvictToFitMultipleSegments) {
    ContextWindowManager mgr(1000);
    mgr.add_segment(make_seg(1, 300, 2));
    mgr.add_segment(make_seg(2, 300, 2));
    mgr.add_segment(make_seg(3, 300, 2));
    // Used = 900, free = 100.  Need 500 free → must evict at least 2 segs.
    std::size_t evicted = mgr.evict_to_fit(500);
    EXPECT_GE(evicted, 2u);
    EXPECT_GE(mgr.free_tokens(), 500u);
}

// ---------------------------------------------------------------------------
// evict_to_fit — priority ordering
// ---------------------------------------------------------------------------

TEST(ContextWindowManager, EvictHighPriorityFirstBeforeLow) {
    // priority 2 (normal) should be evicted before priority 1 (important).
    ContextWindowManager mgr(1000);
    mgr.add_segment(make_seg(1, 400, 1));  // important
    mgr.add_segment(make_seg(2, 400, 2));  // normal
    // Need 500 free tokens; total used = 800.  Must evict ~300+.
    mgr.evict_to_fit(500);
    // The normal-priority segment (id=2) should be evicted first.
    EXPECT_EQ(mgr.find_segment(1) != nullptr, true);   // important kept
    EXPECT_EQ(mgr.find_segment(2), nullptr);            // normal evicted
}

TEST(ContextWindowManager, SystemSegmentsNeverEvicted) {
    ContextWindowManager mgr(1000);
    mgr.add_segment(make_seg(1, 900, 0));  // system — never evict
    mgr.add_segment(make_seg(2, 50,  2));  // normal
    mgr.evict_to_fit(500);
    // System segment must remain.
    EXPECT_NE(mgr.find_segment(1), nullptr);
}

TEST(ContextWindowManager, FIFOWithinSamePriority) {
    ContextWindowManager mgr(1000);
    mgr.add_segment(make_seg(10, 300, 2));  // oldest normal
    mgr.add_segment(make_seg(11, 300, 2));  // newer normal
    mgr.add_segment(make_seg(12, 300, 2));  // newest normal
    // Need to evict exactly one; oldest (id=10) should go first.
    mgr.evict_to_fit(400);
    EXPECT_EQ(mgr.find_segment(10), nullptr);  // oldest evicted
    EXPECT_NE(mgr.find_segment(11), nullptr);
    EXPECT_NE(mgr.find_segment(12), nullptr);
}

// ---------------------------------------------------------------------------
// find_segment
// ---------------------------------------------------------------------------

TEST(ContextWindowManager, FindSegmentExists) {
    ContextWindowManager mgr(1000);
    mgr.add_segment(make_seg(77, 128, 1, 0xDEAD));
    const ContextSegment* s = mgr.find_segment(77);
    ASSERT_NE(s, nullptr);
    EXPECT_EQ(s->segment_id,   77u);
    EXPECT_EQ(s->token_count,  128u);
    EXPECT_EQ(s->priority,     1);
    EXPECT_EQ(s->content_hash, 0xDEADu);
}

TEST(ContextWindowManager, FindSegmentNotExists) {
    ContextWindowManager mgr(1000);
    EXPECT_EQ(mgr.find_segment(42), nullptr);
}

// ---------------------------------------------------------------------------
// resize_segment
// ---------------------------------------------------------------------------

TEST(ContextWindowManager, ResizeSegmentShrink) {
    ContextWindowManager mgr(1000);
    mgr.add_segment(make_seg(1, 500));
    EXPECT_TRUE(mgr.resize_segment(1, 200));
    EXPECT_EQ(mgr.total_tokens(), 200u);
}

TEST(ContextWindowManager, ResizeSegmentGrow) {
    ContextWindowManager mgr(1000);
    mgr.add_segment(make_seg(1, 100));
    EXPECT_TRUE(mgr.resize_segment(1, 400));
    EXPECT_EQ(mgr.total_tokens(), 400u);
}

TEST(ContextWindowManager, ResizeSegmentExceedsCapacity) {
    ContextWindowManager mgr(500);
    mgr.add_segment(make_seg(1, 100));
    EXPECT_FALSE(mgr.resize_segment(1, 600));
    EXPECT_EQ(mgr.total_tokens(), 100u);
}

TEST(ContextWindowManager, ResizeSegmentNotFound) {
    ContextWindowManager mgr(1000);
    EXPECT_FALSE(mgr.resize_segment(999, 100));
}

// ---------------------------------------------------------------------------
// clear
// ---------------------------------------------------------------------------

TEST(ContextWindowManager, ClearEmptiesManager) {
    ContextWindowManager mgr(1000);
    mgr.add_segment(make_seg(1, 100));
    mgr.add_segment(make_seg(2, 200));
    mgr.clear();
    EXPECT_TRUE(mgr.empty());
    EXPECT_EQ(mgr.total_tokens(), 0u);
    EXPECT_EQ(mgr.segment_count(), 0u);
}

// ---------------------------------------------------------------------------
// utilization
// ---------------------------------------------------------------------------

TEST(ContextWindowManager, UtilizationHalfFull) {
    ContextWindowManager mgr(1000);
    mgr.add_segment(make_seg(1, 500));
    EXPECT_NEAR(mgr.utilization(), 0.5, 1e-9);
}

TEST(ContextWindowManager, UtilizationFull) {
    ContextWindowManager mgr(100);
    mgr.add_segment(make_seg(1, 100));
    EXPECT_NEAR(mgr.utilization(), 1.0, 1e-9);
}

// ---------------------------------------------------------------------------
// format_summary
// ---------------------------------------------------------------------------

TEST(ContextWindowManager, FormatSummaryContainsFields) {
    ContextWindowManager mgr(8192);
    mgr.add_segment(make_seg(1, 512));
    std::string s = mgr.format_summary();
    EXPECT_NE(s.find("segments="), std::string::npos);
    EXPECT_NE(s.find("tokens="),   std::string::npos);
    EXPECT_NE(s.find("util="),     std::string::npos);
    EXPECT_NE(s.find("free="),     std::string::npos);
}
