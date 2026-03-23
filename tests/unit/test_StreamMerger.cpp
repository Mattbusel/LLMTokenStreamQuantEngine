#include "gtest/gtest.h"
#include "StreamMerger.h"

using namespace llmquant;

// Helper factory.
static StreamEntry make_entry(
    const std::string& stream_id,
    const std::string& token,
    uint64_t timestamp_ms,
    int priority = 0)
{
    return StreamEntry{stream_id, token, timestamp_ms, priority};
}

// ---------------------------------------------------------------------------
// StreamMerger unit tests
// ---------------------------------------------------------------------------

TEST(StreamMergerTest, test_initial_state_empty) {
    StreamMerger merger;
    EXPECT_EQ(merger.total_entries(), 0u);
    EXPECT_EQ(merger.stream_count(), 0u);
}

TEST(StreamMergerTest, test_add_entry_increments_total) {
    StreamMerger merger;
    merger.add_entry(make_entry("s1", "tok", 1000));
    EXPECT_EQ(merger.total_entries(), 1u);
}

TEST(StreamMergerTest, test_stream_count_counts_unique_streams) {
    StreamMerger merger;
    merger.add_entry(make_entry("s1", "a", 1000));
    merger.add_entry(make_entry("s1", "b", 2000));
    merger.add_entry(make_entry("s2", "c", 3000));
    EXPECT_EQ(merger.stream_count(), 2u);
}

TEST(StreamMergerTest, test_merge_by_priority_descending) {
    StreamMerger merger;
    merger.add_entry(make_entry("s1", "low",  1000, 1));
    merger.add_entry(make_entry("s2", "high", 2000, 5));
    merger.add_entry(make_entry("s3", "mid",  3000, 3));

    auto result = merger.merge_by_priority();
    ASSERT_EQ(result.size(), 3u);
    EXPECT_EQ(result[0].token, "high");
    EXPECT_EQ(result[1].token, "mid");
    EXPECT_EQ(result[2].token, "low");
}

TEST(StreamMergerTest, test_merge_by_priority_tie_broken_by_timestamp) {
    StreamMerger merger;
    merger.add_entry(make_entry("s1", "later",  2000, 5));
    merger.add_entry(make_entry("s2", "earlier", 1000, 5));

    auto result = merger.merge_by_priority();
    ASSERT_EQ(result.size(), 2u);
    EXPECT_EQ(result[0].token, "earlier");
    EXPECT_EQ(result[1].token, "later");
}

TEST(StreamMergerTest, test_merge_by_timestamp_ascending) {
    StreamMerger merger;
    merger.add_entry(make_entry("s1", "t3", 3000));
    merger.add_entry(make_entry("s2", "t1", 1000));
    merger.add_entry(make_entry("s3", "t2", 2000));

    auto result = merger.merge_by_timestamp();
    ASSERT_EQ(result.size(), 3u);
    EXPECT_EQ(result[0].token, "t1");
    EXPECT_EQ(result[1].token, "t2");
    EXPECT_EQ(result[2].token, "t3");
}

TEST(StreamMergerTest, test_merge_by_timestamp_preserves_all_entries) {
    StreamMerger merger;
    for (int i = 0; i < 10; ++i) {
        merger.add_entry(make_entry("s1", "tok" + std::to_string(i), static_cast<uint64_t>(i * 100)));
    }
    auto result = merger.merge_by_timestamp();
    EXPECT_EQ(result.size(), 10u);
}

TEST(StreamMergerTest, test_merge_round_robin_interleaves) {
    StreamMerger merger;
    merger.add_entry(make_entry("s1", "a1", 1000));
    merger.add_entry(make_entry("s1", "a2", 2000));
    merger.add_entry(make_entry("s2", "b1", 1500));
    merger.add_entry(make_entry("s2", "b2", 2500));

    auto result = merger.merge_round_robin({"s1", "s2"});
    ASSERT_GE(result.size(), 4u);
    // First entry from s1, then s2, alternating.
    EXPECT_EQ(result[0].stream_id, "s1");
    EXPECT_EQ(result[1].stream_id, "s2");
    EXPECT_EQ(result[2].stream_id, "s1");
    EXPECT_EQ(result[3].stream_id, "s2");
}

TEST(StreamMergerTest, test_merge_round_robin_unequal_streams) {
    StreamMerger merger;
    merger.add_entry(make_entry("s1", "a1", 100));
    merger.add_entry(make_entry("s1", "a2", 200));
    merger.add_entry(make_entry("s1", "a3", 300));
    merger.add_entry(make_entry("s2", "b1", 150));

    auto result = merger.merge_round_robin({"s1", "s2"});
    // s1 has 3, s2 has 1 → [a1, b1, a2, a3]
    ASSERT_EQ(result.size(), 4u);
    EXPECT_EQ(result[0].token, "a1");
    EXPECT_EQ(result[1].token, "b1");
    EXPECT_EQ(result[2].token, "a2");
    EXPECT_EQ(result[3].token, "a3");
}

TEST(StreamMergerTest, test_merge_round_robin_unknown_stream_id_skipped) {
    StreamMerger merger;
    merger.add_entry(make_entry("s1", "a1", 100));
    // "ghost" does not exist in buffer.
    auto result = merger.merge_round_robin({"ghost", "s1"});
    EXPECT_EQ(result.size(), 1u);
    EXPECT_EQ(result[0].token, "a1");
}

TEST(StreamMergerTest, test_merge_round_robin_empty_stream_ids) {
    StreamMerger merger;
    merger.add_entry(make_entry("s1", "a", 100));
    // No IDs listed → unlisted entries are appended.
    auto result = merger.merge_round_robin({});
    EXPECT_EQ(result.size(), 1u);
}

TEST(StreamMergerTest, test_clear_resets_state) {
    StreamMerger merger;
    merger.add_entry(make_entry("s1", "a", 100));
    merger.add_entry(make_entry("s2", "b", 200));
    merger.clear();
    EXPECT_EQ(merger.total_entries(), 0u);
    EXPECT_EQ(merger.stream_count(), 0u);
}

TEST(StreamMergerTest, test_merge_by_priority_empty_buffer) {
    StreamMerger merger;
    EXPECT_TRUE(merger.merge_by_priority().empty());
}

TEST(StreamMergerTest, test_merge_by_timestamp_empty_buffer) {
    StreamMerger merger;
    EXPECT_TRUE(merger.merge_by_timestamp().empty());
}

TEST(StreamMergerTest, test_merge_round_robin_empty_buffer) {
    StreamMerger merger;
    auto result = merger.merge_round_robin({"s1", "s2"});
    EXPECT_TRUE(result.empty());
}

TEST(StreamMergerTest, test_total_entries_after_multiple_adds) {
    StreamMerger merger;
    for (int i = 0; i < 7; ++i) {
        merger.add_entry(make_entry("s" + std::to_string(i % 3), "t", static_cast<uint64_t>(i)));
    }
    EXPECT_EQ(merger.total_entries(), 7u);
}
